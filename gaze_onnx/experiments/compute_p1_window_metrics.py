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


LOCATION_CLASSES = ("Forward", "Non-Forward", "In-Car")
OFF_PATH_CLASSES = {"Non-Forward", "In-Car"}
WHEEL_VALID = {"ON", "OFF"}


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
        "--nearest-wheel-max-gap",
        type=float,
        default=0.35,
        help="Max allowed |t_gaze - t_wheel| when mapping wheel state to a gaze frame",
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
        v = str(r[value_col]).strip()
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
    return x


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


def load_wheel_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    time_col = "video_time_sec" if "video_time_sec" in df.columns else "time_sec"
    if time_col not in df.columns:
        raise ValueError(f"{path} missing time_sec/video_time_sec")

    t = pd.to_numeric(df[time_col], errors="coerce")
    st = [normalize_wheel_state_from_row(r) for _, r in df.iterrows()]
    x = pd.DataFrame({"t": t, "wheel": st}).dropna(subset=["t"])
    x = x.sort_values("t").reset_index(drop=True)
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
) -> Dict[str, str]:
    g = gaze_df[(gaze_df["t"] >= w0) & (gaze_df["t"] < w1)].copy()
    wh = wheel_df[(wheel_df["t"] >= w0) & (wheel_df["t"] < w1)].copy()

    g_times = g["t"].to_numpy(dtype=float) if not g.empty else np.array([], dtype=float)
    g_cls = g["gaze"].to_numpy(dtype=object) if not g.empty else np.array([], dtype=object)
    w_times = wh["t"].to_numpy(dtype=float) if not wh.empty else np.array([], dtype=float)
    w_state = wh["wheel"].to_numpy(dtype=object) if not wh.empty else np.array([], dtype=object)

    mapped_wheel = nearest_wheel_state(g_times, w_times, w_state, max_gap=max_gap)

    off_ratio = weighted_offpath_ratio(g_times, g_cls)
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

    def fmt(v: float) -> str:
        if v is None or not np.isfinite(float(v)):
            return ""
        return f"{float(v):.6f}"

    row = {
        "video_path": video_path,
        "window_start_sec": f"{w0:.3f}",
        "window_end_sec": f"{w1:.3f}",
        "window_duration_sec": f"{win_sec:.3f}",
        "gaze_rows": str(int(g_times.size)),
        "wheel_rows": str(int(w_times.size)),
        "pct_time_off_path": fmt(off_ratio),
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
                wheel_cache[wheel_csv] = load_wheel_csv(wheel_path)
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

    n_ok = sum(1 for r in out_rows if r.get("status") == "ok")
    print(f"Windows total={len(out_rows)} ok={n_ok}")
    print(f"Missing gaze map: {n_no_gaze}, missing wheel map: {n_no_wheel}")
    print(
        "Missing files: "
        f"gaze={n_missing_gaze_file}, wheel={n_missing_wheel_file}, load_error={n_load_err}"
    )
    print(f"Output: {out_csv}")


if __name__ == "__main__":
    main()
