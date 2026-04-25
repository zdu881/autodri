#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute p1 20s-window metrics from gaze + wheel frame-level CSVs.

Expected inputs:
1) windows CSV (from build_p1_schedule_windows.py), containing at least:
   - window_uid, video_path, window_start_sec, window_end_sec
2) gaze map CSV:
   - video_path, gaze_csv
3) wheel map CSV:
   - video_path, wheel_csv

Metrics per window:
1) % time off-path (Non-Forward + In-Car)
2) Glance location count/rate (entry counts for Forward/Non-Forward/In-Car)
3) Off-path episodes >= 1.6s
4) Off-path episodes >= 2.0s
5) Wheel ON/OFF ratio within each glance location
6) Overall wheel ON/OFF ratio
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from autodri.common.paths import resolve_workspace_or_repo_path


LOCATION_CLASSES = ("Forward", "Non-Forward", "In-Car")
OFF_PATH_CLASSES = {"Non-Forward", "In-Car"}
WHEEL_VALID = {"ON", "OFF"}
STATE_ON = "ON"
STATE_OFF = "OFF"
STATE_UNCERTAIN = "UNCERTAIN"
SUPPORTED_GAZE_FPS = (25.0, 30.0)


@dataclass
class WindowMetric:
    row: Dict[str, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute p1 window metrics from gaze/wheel frame CSVs")
    p.add_argument("--windows-csv", required=True)
    p.add_argument("--gaze-map-csv", required=True, help="CSV with columns: video_path,gaze_csv")
    p.add_argument("--wheel-map-csv", required=True, help="CSV with columns: video_path,wheel_csv")
    p.add_argument("--out-csv", required=True)
    p.add_argument(
        "--out-qc-csv",
        default="",
        help="Optional QC CSV for windows with zero/low gaze coverage. Default derives from --out-csv.",
    )
    p.add_argument(
        "--nearest-wheel-max-gap",
        type=float,
        default=0.35,
        help="Max allowed |t_gaze - t_wheel| when mapping wheel state to a gaze frame",
    )
    p.add_argument(
        "--resolve-uncertain",
        choices=["keep", "split"],
        default="keep",
        help=(
            "How to handle wheel UNCERTAIN states. "
            "'keep' preserves them; 'split' force-assigns them to ON/OFF using temporal rules."
        ),
    )
    p.add_argument(
        "--uncertain-bridge-on-sec",
        type=float,
        default=2.0,
        help=(
            "When --resolve-uncertain=split, a UNCERTAIN run bounded by ON/ON is reassigned "
            "to ON only if its duration is no longer than this threshold."
        ),
    )
    p.add_argument(
        "--uncertain-bridge-mixed-sec",
        type=float,
        default=1.0,
        help=(
            "When --resolve-uncertain=split, a UNCERTAIN run between conflicting states "
            "(or with only one neighbor) is bridged only if its duration is no longer than this threshold."
        ),
    )
    p.add_argument(
        "--uncertain-default-state",
        choices=["ON", "OFF"],
        default="OFF",
        help=(
            "Fallback state for unresolved/long UNCERTAIN runs when --resolve-uncertain=split. "
            "Default OFF is conservative because most UNCERTAIN frames have no hand evidence."
        ),
    )
    p.add_argument(
        "--gaze-coverage-threshold",
        type=float,
        default=0.98,
        help="Minimum gaze_rows / expected_gaze_rows required for a window to be used in final summaries.",
    )
    return p.parse_args()


def canon_path(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    return Path(t).as_posix()


def load_map(path: Path, value_col: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "video_path" not in df.columns or value_col not in df.columns:
        raise ValueError(f"{path} missing required columns: video_path,{value_col}")
    by_video: Dict[str, str] = {}
    by_segment: Dict[str, str] = {}
    for _, r in df.iterrows():
        k = canon_path(str(r["video_path"]))
        raw_v = str(r[value_col]).strip()
        v = str(resolve_workspace_or_repo_path(raw_v)) if raw_v else ""
        if k and v:
            by_video[k] = v
        if "segment_uid" in df.columns:
            sid = str(r.get("segment_uid", "")).strip()
            if sid and v:
                by_segment[sid] = v
    return by_video, by_segment


def load_gaze_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "Gaze_Class" not in df.columns:
        raise ValueError(f"{path} missing Gaze_Class")
    time_col = "Video_Timestamp" if "Video_Timestamp" in df.columns else "Timestamp"
    if time_col not in df.columns:
        raise ValueError(f"{path} missing Timestamp/Video_Timestamp")
    x = pd.DataFrame(
        {
            "t": pd.to_numeric(df[time_col], errors="coerce"),
            "gaze": df["Gaze_Class"].astype(str).str.strip(),
        }
    ).dropna(subset=["t"])
    x = x.sort_values("t").reset_index(drop=True)
    x.attrs["nominal_gaze_fps"] = infer_nominal_gaze_fps(x["t"].to_numpy(dtype=float))
    return x


def infer_nominal_gaze_fps(times: np.ndarray) -> float:
    if times.size < 2:
        return 25.0
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 25.0
    raw_fps = float(1.0 / np.median(diffs))
    return float(min(SUPPORTED_GAZE_FPS, key=lambda fps: abs(float(fps) - raw_fps)))


def expected_gaze_rows_for_window(window_sec: float, nominal_gaze_fps: float) -> int:
    return max(1, int(round(float(window_sec) * float(nominal_gaze_fps))))


def normalize_wheel_state_from_row(row: pd.Series) -> str:
    if "stable_state" in row and isinstance(row["stable_state"], str):
        t = row["stable_state"].strip().upper()
        if t in {"ON", "OFF", "UNCERTAIN"}:
            return t
    if "stable_hand_on_wheel" in row:
        try:
            v = int(float(row["stable_hand_on_wheel"]))
            if v == 1:
                return "ON"
            if v == 0:
                return "OFF"
            return "UNCERTAIN"
        except Exception:
            pass
    return "UNCERTAIN"


def resolve_uncertain_wheel_states(
    times: np.ndarray,
    states: np.ndarray,
    bridge_on_sec: float,
    bridge_mixed_sec: float,
    default_state: str,
) -> np.ndarray:
    """Force-assign UNCERTAIN wheel states to ON/OFF using conservative temporal rules.

    Policy:
    1) OFF ... UNCERTAIN ... OFF  -> OFF
    2) ON  ... UNCERTAIN ... ON   -> ON only for short gaps (<= bridge_on_sec)
    3) Conflicting neighbors or single-sided gaps:
       - use previous/available state only for short gaps (<= bridge_mixed_sec)
       - otherwise fall back to default_state
    """
    if states.size == 0:
        return states

    out = states.astype(object).copy()
    positive_diffs = np.diff(times)
    positive_diffs = positive_diffs[positive_diffs > 0]
    dt_med = float(np.median(positive_diffs)) if positive_diffs.size else (1.0 / 25.0)

    n = int(states.size)
    i = 0
    while i < n:
        if str(states[i]) != STATE_UNCERTAIN:
            i += 1
            continue

        j = i
        while j + 1 < n and str(states[j + 1]) == STATE_UNCERTAIN:
            j += 1

        dur_sec = float(times[j] - times[i] + dt_med) if j >= i else dt_med
        left = str(out[i - 1]) if i - 1 >= 0 and str(out[i - 1]) != STATE_UNCERTAIN else ""
        right = str(states[j + 1]) if j + 1 < n and str(states[j + 1]) != STATE_UNCERTAIN else ""

        fill = default_state
        if left == STATE_OFF and right == STATE_OFF:
            fill = STATE_OFF
        elif left == STATE_ON and right == STATE_ON:
            fill = STATE_ON if dur_sec <= float(bridge_on_sec) else default_state
        elif left and right and left != right:
            fill = left if dur_sec <= float(bridge_mixed_sec) else default_state
        elif left or right:
            neighbor = left or right
            if neighbor == STATE_OFF:
                fill = STATE_OFF
            elif dur_sec <= float(bridge_mixed_sec):
                fill = neighbor
            else:
                fill = default_state

        out[i : j + 1] = fill
        i = j + 1

    return out


def load_wheel_csv(
    path: Path,
    resolve_uncertain: str = "keep",
    uncertain_bridge_on_sec: float = 2.0,
    uncertain_bridge_mixed_sec: float = 1.0,
    uncertain_default_state: str = "OFF",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    time_col = "video_time_sec" if "video_time_sec" in df.columns else "time_sec"
    if time_col not in df.columns:
        raise ValueError(f"{path} missing time_sec/video_time_sec")

    t = pd.to_numeric(df[time_col], errors="coerce")
    st = np.array([normalize_wheel_state_from_row(r) for _, r in df.iterrows()], dtype=object)
    x = pd.DataFrame({"t": t, "wheel_raw": st}).dropna(subset=["t"])
    x = x.sort_values("t").reset_index(drop=True)
    if str(resolve_uncertain).strip().lower() == "split":
        resolved = resolve_uncertain_wheel_states(
            times=x["t"].to_numpy(dtype=float),
            states=x["wheel_raw"].to_numpy(dtype=object),
            bridge_on_sec=float(uncertain_bridge_on_sec),
            bridge_mixed_sec=float(uncertain_bridge_mixed_sec),
            default_state=str(uncertain_default_state).strip().upper(),
        )
        x["wheel"] = resolved
    else:
        x["wheel"] = x["wheel_raw"]
    return x


def nearest_wheel_state(
    gaze_t: np.ndarray, wheel_t: np.ndarray, wheel_state: np.ndarray, max_gap: float
) -> np.ndarray:
    if gaze_t.size == 0 or wheel_t.size == 0:
        return np.array([""] * gaze_t.size, dtype=object)

    idx = np.searchsorted(wheel_t, gaze_t, side="left")
    out = np.array([""] * gaze_t.size, dtype=object)
    for i, j in enumerate(idx):
        cand = []
        if j < wheel_t.size:
            cand.append(j)
        if j - 1 >= 0:
            cand.append(j - 1)
        if not cand:
            continue
        best = min(cand, key=lambda k: abs(float(wheel_t[k]) - float(gaze_t[i])))
        if abs(float(wheel_t[best]) - float(gaze_t[i])) <= max_gap:
            out[i] = str(wheel_state[best])
    return out


def weighted_offpath_ratio(times: np.ndarray, classes: np.ndarray) -> float:
    if times.size == 0:
        return float("nan")
    if times.size == 1:
        return 100.0 if classes[0] in OFF_PATH_CLASSES else 0.0

    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    dt_med = float(np.median(diffs)) if diffs.size else (1.0 / 30.0)
    dt = np.diff(times, append=times[-1] + dt_med)

    is_loc = np.isin(classes, LOCATION_CLASSES)
    is_off = np.isin(classes, list(OFF_PATH_CLASSES))
    total = float(np.sum(dt[is_loc]))
    if total <= 1e-9:
        return float("nan")
    off = float(np.sum(dt[np.logical_and(is_loc, is_off)]))
    return off / total * 100.0


def weighted_location_ratios(times: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    """Return per-location time ratios (%) within valid gaze-location time only."""
    out = {k: float("nan") for k in LOCATION_CLASSES}
    if times.size == 0:
        return out
    if times.size == 1:
        cls = str(classes[0])
        if cls in LOCATION_CLASSES:
            out[cls] = 100.0
        return out

    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    dt_med = float(np.median(diffs)) if diffs.size else (1.0 / 30.0)
    dt = np.diff(times, append=times[-1] + dt_med)

    is_loc = np.isin(classes, LOCATION_CLASSES)
    total = float(np.sum(dt[is_loc]))
    if total <= 1e-9:
        return out

    for loc in LOCATION_CLASSES:
        mask = np.logical_and(is_loc, classes == loc)
        out[loc] = float(np.sum(dt[mask])) / total * 100.0
    return out


def glance_entry_counts(classes: np.ndarray) -> Dict[str, int]:
    counts = {k: 0 for k in LOCATION_CLASSES}
    prev = ""
    for c in classes:
        if c not in LOCATION_CLASSES:
            prev = ""
            continue
        if c != prev:
            counts[c] += 1
        prev = c
    return counts


def offpath_episode_counts(times: np.ndarray, classes: np.ndarray) -> Tuple[int, int]:
    if times.size == 0:
        return 0, 0
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    dt_med = float(np.median(diffs)) if diffs.size else (1.0 / 30.0)

    is_off = np.array([c in OFF_PATH_CLASSES for c in classes], dtype=bool)
    n16 = 0
    n20 = 0
    i = 0
    n = len(is_off)
    while i < n:
        if not is_off[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and is_off[j + 1]:
            j += 1
        if j + 1 < n:
            dur = float(times[j + 1] - times[i])
        else:
            dur = float(times[j] - times[i] + dt_med)
        if dur >= 1.6:
            n16 += 1
        if dur >= 2.0:
            n20 += 1
        i = j + 1
    return n16, n20


def ratio_on_off(states: np.ndarray) -> Tuple[float, float, int, int]:
    on = int(np.sum(states == "ON"))
    off = int(np.sum(states == "OFF"))
    den = on + off
    if den <= 0:
        return float("nan"), float("nan"), on, off
    return on / den, off / den, on, off


def compute_one_window(
    video_path: str,
    w0: float,
    w1: float,
    gaze_df: pd.DataFrame,
    wheel_df: pd.DataFrame,
    max_gap: float,
    gaze_coverage_threshold: float,
) -> Dict[str, str]:
    g = gaze_df[(gaze_df["t"] >= w0) & (gaze_df["t"] < w1)].copy()
    wh = wheel_df[(wheel_df["t"] >= w0) & (wheel_df["t"] < w1)].copy()

    g_times = g["t"].to_numpy(dtype=float) if not g.empty else np.array([], dtype=float)
    g_cls = g["gaze"].to_numpy(dtype=object) if not g.empty else np.array([], dtype=object)
    w_times = wh["t"].to_numpy(dtype=float) if not wh.empty else np.array([], dtype=float)
    w_state = wh["wheel"].to_numpy(dtype=object) if not wh.empty else np.array([], dtype=object)

    mapped_wheel = nearest_wheel_state(g_times, w_times, w_state, max_gap=max_gap)

    off_ratio = weighted_offpath_ratio(g_times, g_cls)
    loc_ratio = weighted_location_ratios(g_times, g_cls)
    glance_cnt = glance_entry_counts(g_cls)
    glance_total = int(sum(glance_cnt.values()))
    win_sec = max(1e-9, float(w1 - w0))
    win_min = win_sec / 60.0
    glance_rate = float(glance_total) / win_min
    glance_rate_forward = float(glance_cnt["Forward"]) / win_min
    glance_rate_nonforward = float(glance_cnt["Non-Forward"]) / win_min
    glance_rate_incar = float(glance_cnt["In-Car"]) / win_min
    off16, off20 = offpath_episode_counts(g_times, g_cls)

    # Ratios of wheel ON/OFF under each gaze location.
    per_loc = {}
    for loc in LOCATION_CLASSES:
        mask = np.logical_and(g_cls == loc, np.isin(mapped_wheel, ["ON", "OFF"]))
        on_r, off_r, on_n, off_n = ratio_on_off(mapped_wheel[mask])
        per_loc[loc] = (on_r, off_r, on_n, off_n)

    # Overall wheel ratio from wheel stream itself.
    overall_on_r, overall_off_r, overall_on_n, overall_off_n = ratio_on_off(w_state)

    nominal_gaze_fps = float(
        gaze_df.attrs.get("nominal_gaze_fps", infer_nominal_gaze_fps(gaze_df["t"].to_numpy(dtype=float)))
    )
    expected_gaze_rows = int(expected_gaze_rows_for_window(win_sec, nominal_gaze_fps))
    coverage_ratio = float(g_times.size) / float(expected_gaze_rows) if expected_gaze_rows > 0 else float("nan")
    coverage_ok = bool(expected_gaze_rows > 0 and coverage_ratio >= float(gaze_coverage_threshold))
    if g_times.size <= 0:
        gaze_qc_reason = "zero_gaze_rows"
    elif coverage_ok:
        gaze_qc_reason = ""
    else:
        gaze_qc_reason = "low_gaze_coverage"

    def fmt(v: float) -> str:
        if v is None or not np.isfinite(float(v)):
            return ""
        return f"{float(v):.6f}"

    row = {
        "video_path": video_path,
        "window_start_sec": f"{w0:.3f}",
        "window_end_sec": f"{w1:.3f}",
        "window_duration_sec": f"{win_sec:.3f}",
        "nominal_gaze_fps": fmt(nominal_gaze_fps),
        "gaze_rows": str(int(g_times.size)),
        "expected_gaze_rows": str(int(expected_gaze_rows)),
        "gaze_coverage_ratio": fmt(coverage_ratio),
        "gaze_coverage_threshold": fmt(float(gaze_coverage_threshold)),
        "gaze_coverage_ok": "1" if coverage_ok else "0",
        "gaze_qc_reason": gaze_qc_reason,
        "wheel_rows": str(int(w_times.size)),
        "pct_time_off_path": fmt(off_ratio),
        "pct_time_forward": fmt(loc_ratio["Forward"]),
        "pct_time_nonforward": fmt(loc_ratio["Non-Forward"]),
        "pct_time_incar": fmt(loc_ratio["In-Car"]),
        "glance_count_total": str(glance_total),
        "glance_rate_per_min": fmt(glance_rate),
        "glance_count_forward": str(int(glance_cnt["Forward"])),
        "glance_rate_forward_per_min": fmt(glance_rate_forward),
        "glance_count_nonforward": str(int(glance_cnt["Non-Forward"])),
        "glance_rate_nonforward_per_min": fmt(glance_rate_nonforward),
        "glance_count_incar": str(int(glance_cnt["In-Car"])),
        "glance_rate_incar_per_min": fmt(glance_rate_incar),
        "offpath_count_ge_1p6s": str(int(off16)),
        "offpath_count_ge_2p0s": str(int(off20)),
        "wheel_on_ratio_forward": fmt(per_loc["Forward"][0]),
        "wheel_off_ratio_forward": fmt(per_loc["Forward"][1]),
        "wheel_on_n_forward": str(int(per_loc["Forward"][2])),
        "wheel_off_n_forward": str(int(per_loc["Forward"][3])),
        "wheel_on_ratio_nonforward": fmt(per_loc["Non-Forward"][0]),
        "wheel_off_ratio_nonforward": fmt(per_loc["Non-Forward"][1]),
        "wheel_on_n_nonforward": str(int(per_loc["Non-Forward"][2])),
        "wheel_off_n_nonforward": str(int(per_loc["Non-Forward"][3])),
        "wheel_on_ratio_incar": fmt(per_loc["In-Car"][0]),
        "wheel_off_ratio_incar": fmt(per_loc["In-Car"][1]),
        "wheel_on_n_incar": str(int(per_loc["In-Car"][2])),
        "wheel_off_n_incar": str(int(per_loc["In-Car"][3])),
        "wheel_on_ratio_overall": fmt(overall_on_r),
        "wheel_off_ratio_overall": fmt(overall_off_r),
        "wheel_on_n_overall": str(int(overall_on_n)),
        "wheel_off_n_overall": str(int(overall_off_n)),
    }
    return row


def main() -> None:
    args = parse_args()
    windows_csv = Path(args.windows_csv)
    gaze_map_csv = Path(args.gaze_map_csv)
    wheel_map_csv = Path(args.wheel_map_csv)
    out_csv = Path(args.out_csv)
    out_qc_csv = Path(args.out_qc_csv) if str(args.out_qc_csv).strip() else out_csv.with_name(f"{out_csv.stem}.gaze_qc.csv")

    if not windows_csv.exists():
        raise FileNotFoundError(windows_csv)

    wdf = pd.read_csv(windows_csv)
    required = {"window_uid", "video_path", "window_start_sec", "window_end_sec"}
    miss = required - set(wdf.columns)
    if miss:
        raise ValueError(f"{windows_csv} missing columns: {sorted(miss)}")

    gaze_by_video, gaze_by_segment = load_map(gaze_map_csv, value_col="gaze_csv")
    wheel_by_video, wheel_by_segment = load_map(wheel_map_csv, value_col="wheel_csv")

    gaze_cache: Dict[str, pd.DataFrame] = {}
    wheel_cache: Dict[str, pd.DataFrame] = {}
    out_rows: List[Dict[str, str]] = []

    n_no_gaze = 0
    n_no_wheel = 0
    n_missing_gaze_file = 0
    n_missing_wheel_file = 0
    n_load_err = 0

    for _, wr in wdf.iterrows():
        video_path = canon_path(str(wr["video_path"]))
        segment_uid = str(wr.get("segment_uid", "")).strip()
        window_uid = str(wr["window_uid"])
        w0 = float(wr["window_start_sec"])
        w1 = float(wr["window_end_sec"])

        gaze_csv = gaze_by_segment.get(segment_uid, "") if segment_uid else ""
        wheel_csv = wheel_by_segment.get(segment_uid, "") if segment_uid else ""
        map_mode = "segment" if (gaze_csv or wheel_csv) else "video"
        if not gaze_csv:
            gaze_csv = gaze_by_video.get(video_path, "")
        if not wheel_csv:
            wheel_csv = wheel_by_video.get(video_path, "")
        if not gaze_csv:
            n_no_gaze += 1
        if not wheel_csv:
            n_no_wheel += 1

        row = dict((k, str(v)) for k, v in wr.to_dict().items())
        row["video_path"] = video_path
        row["window_uid"] = window_uid
        row["segment_uid"] = segment_uid
        row["map_mode"] = map_mode
        row["gaze_csv"] = gaze_csv
        row["wheel_csv"] = wheel_csv
        row["wheel_uncertain_policy"] = str(args.resolve_uncertain)
        row["error"] = ""

        if not gaze_csv or not wheel_csv:
            row["status"] = "missing_csv_map"
            out_rows.append(row)
            continue

        gaze_path = Path(gaze_csv)
        wheel_path = Path(wheel_csv)
        gaze_exists = gaze_path.exists()
        wheel_exists = wheel_path.exists()
        if not gaze_exists:
            n_missing_gaze_file += 1
        if not wheel_exists:
            n_missing_wheel_file += 1
        if not gaze_exists or not wheel_exists:
            if (not gaze_exists) and (not wheel_exists):
                row["status"] = "missing_gaze_wheel_file"
            elif not gaze_exists:
                row["status"] = "missing_gaze_file"
            else:
                row["status"] = "missing_wheel_file"
            out_rows.append(row)
            continue

        try:
            if gaze_csv not in gaze_cache:
                gaze_cache[gaze_csv] = load_gaze_csv(gaze_path)
            if wheel_csv not in wheel_cache:
                wheel_cache[wheel_csv] = load_wheel_csv(
                    wheel_path,
                    resolve_uncertain=str(args.resolve_uncertain),
                    uncertain_bridge_on_sec=float(args.uncertain_bridge_on_sec),
                    uncertain_bridge_mixed_sec=float(args.uncertain_bridge_mixed_sec),
                    uncertain_default_state=str(args.uncertain_default_state),
                )
        except Exception as exc:
            n_load_err += 1
            row["status"] = "csv_load_error"
            row["error"] = str(exc)
            out_rows.append(row)
            continue

        met = compute_one_window(
            video_path=video_path,
            w0=w0,
            w1=w1,
            gaze_df=gaze_cache[gaze_csv],
            wheel_df=wheel_cache[wheel_csv],
            max_gap=float(args.nearest_wheel_max_gap),
            gaze_coverage_threshold=float(args.gaze_coverage_threshold),
        )
        row.update(met)
        row["status"] = "ok"
        out_rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        fields = sorted({k for r in out_rows for k in r.keys()})
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
    else:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["empty"])

    qc_rows = [
        r
        for r in out_rows
        if r.get("status") == "ok" and str(r.get("gaze_coverage_ok", "0")).strip() != "1"
    ]
    out_qc_csv.parent.mkdir(parents=True, exist_ok=True)
    if qc_rows:
        fields = sorted({k for r in qc_rows for k in r.keys()})
        with out_qc_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(qc_rows)
    else:
        with out_qc_csv.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["empty"])

    n_ok = sum(1 for r in out_rows if r.get("status") == "ok")
    n_coverage_ok = sum(
        1 for r in out_rows if r.get("status") == "ok" and str(r.get("gaze_coverage_ok", "0")).strip() == "1"
    )
    print(f"Windows total={len(out_rows)} ok={n_ok}")
    print(f"Coverage ok windows={n_coverage_ok} threshold={float(args.gaze_coverage_threshold):.3f}")
    print(f"Missing gaze map: {n_no_gaze}, missing wheel map: {n_no_wheel}")
    print(
        "Missing files: "
        f"gaze={n_missing_gaze_file}, wheel={n_missing_wheel_file}, load_error={n_load_err}"
    )
    print(f"Output: {out_csv}")
    print(f"QC Output: {out_qc_csv}")


if __name__ == "__main__":
    main()
