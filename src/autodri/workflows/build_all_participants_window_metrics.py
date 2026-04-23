#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build one all-participants window-metrics table.

This script reuses the same metric logic as compute_p1_window_metrics.py and
applies it to every participant with available windows/gaze_map/wheel_map
inputs. It writes:

1) one long-form CSV with one row per 20s window
2) one per-participant summary CSV
3) one optional XLSX workbook with both sheets
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from autodri.common.paths import participant_analysis_dir, reports_root
from autodri.workflows.compute_p1_window_metrics import (
    canon_path,
    compute_one_window,
    load_gaze_csv,
    load_map,
    load_wheel_csv,
)


PARTICIPANTS = [
    "p1",
    "p2",
    "p4",
    "p6",
    "p7",
    "p8",
    "p9",
    "p10",
    "p11",
    "p12",
    "p13",
    "p14",
    "p15",
    "p16",
    "p17",
    "p18",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build all-participants window metrics table")
    p.add_argument(
        "--out-csv",
        default=str(reports_root() / "all_participants_window_metrics.current.csv"),
        help="Long-form output CSV",
    )
    p.add_argument(
        "--out-summary-csv",
        default=str(reports_root() / "all_participants_window_metrics_summary.current.csv"),
        help="Per-participant coverage summary CSV",
    )
    p.add_argument(
        "--out-xlsx",
        default=str(reports_root() / "all_participants_window_metrics.current.xlsx"),
        help="Workbook with all_windows + participant_summary sheets",
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
        help="How to handle wheel UNCERTAIN states before metric extraction",
    )
    p.add_argument(
        "--uncertain-bridge-on-sec",
        type=float,
        default=2.0,
        help="Short-gap threshold for ON...UNCERTAIN...ON bridging",
    )
    p.add_argument(
        "--uncertain-bridge-mixed-sec",
        type=float,
        default=1.0,
        help="Short-gap threshold for mixed-neighbor UNCERTAIN bridging",
    )
    p.add_argument(
        "--uncertain-default-state",
        choices=["ON", "OFF"],
        default="OFF",
        help="Fallback state for unresolved/long UNCERTAIN runs",
    )
    p.add_argument(
        "--participants",
        nargs="*",
        default=PARTICIPANTS,
        help="Participants to include (default: current study set)",
    )
    p.add_argument(
        "--skip-xlsx",
        action="store_true",
        help="Skip XLSX export and only write CSV files",
    )
    return p.parse_args()


def participant_paths(participant: str) -> Dict[str, Path]:
    if participant == "p1":
        analysis_dir = participant_analysis_dir("p1")
        manual_gmap = analysis_dir / "p1_gaze_map.segment.manual_roi_20260414.csv"
        manual_wmap = analysis_dir / "p1_wheel_map.segment.manual_roi_20260414.csv"
        return {
            "windows": analysis_dir / "p1_windows.20s.csv",
            "gaze_map": manual_gmap if manual_gmap.exists() else (analysis_dir / "p1_gaze_map.segment.csv"),
            "wheel_map": manual_wmap if manual_wmap.exists() else (analysis_dir / "p1_wheel_map.segment.csv"),
        }
    analysis_dir = participant_analysis_dir(participant)
    return {
        "windows": analysis_dir / f"{participant}_windows.20s.current.csv",
        "gaze_map": analysis_dir / f"{participant}_gaze_map.current.csv",
        "wheel_map": analysis_dir / f"{participant}_wheel_map.current.csv",
    }


def compute_rows_for_participant(
    participant: str,
    windows_csv: Path,
    gaze_map_csv: Path,
    wheel_map_csv: Path,
    max_gap: float,
    resolve_uncertain: str,
    uncertain_bridge_on_sec: float,
    uncertain_bridge_mixed_sec: float,
    uncertain_default_state: str,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    summary: Dict[str, object] = {
        "participant": participant,
        "windows_csv": str(windows_csv),
        "gaze_map_csv": str(gaze_map_csv),
        "wheel_map_csv": str(wheel_map_csv),
        "windows_total": 0,
        "ok_windows": 0,
        "missing_csv_map_windows": 0,
        "missing_gaze_file_windows": 0,
        "missing_wheel_file_windows": 0,
        "missing_both_files_windows": 0,
        "csv_load_error_windows": 0,
        "input_status": "ok",
        "wheel_uncertain_policy": resolve_uncertain,
        "uncertain_bridge_on_sec": uncertain_bridge_on_sec,
        "uncertain_bridge_mixed_sec": uncertain_bridge_mixed_sec,
        "uncertain_default_state": uncertain_default_state,
    }

    missing_inputs = [
        name for name, path in [("windows", windows_csv), ("gaze_map", gaze_map_csv), ("wheel_map", wheel_map_csv)]
        if not path.exists()
    ]
    if missing_inputs:
        summary["input_status"] = "missing_inputs:" + ",".join(missing_inputs)
        return [], summary

    wdf = pd.read_csv(windows_csv)
    required = {"window_uid", "video_path", "window_start_sec", "window_end_sec"}
    miss = required - set(wdf.columns)
    if miss:
        summary["input_status"] = "invalid_windows_csv:" + ",".join(sorted(miss))
        return [], summary

    summary["windows_total"] = int(len(wdf))

    gaze_by_video, gaze_by_segment = load_map(gaze_map_csv, value_col="gaze_csv")
    wheel_by_video, wheel_by_segment = load_map(wheel_map_csv, value_col="wheel_csv")

    gaze_cache: Dict[str, pd.DataFrame] = {}
    wheel_cache: Dict[str, pd.DataFrame] = {}
    out_rows: List[Dict[str, str]] = []

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

        row = {k: str(v) for k, v in wr.to_dict().items()}
        row["participant"] = str(row.get("participant", participant) or participant)
        row["video_path"] = video_path
        row["window_uid"] = window_uid
        row["segment_uid"] = segment_uid
        row["map_mode"] = map_mode
        row["gaze_csv"] = gaze_csv
        row["wheel_csv"] = wheel_csv
        row["wheel_uncertain_policy"] = resolve_uncertain
        row["error"] = ""

        if not gaze_csv or not wheel_csv:
            row["status"] = "missing_csv_map"
            summary["missing_csv_map_windows"] = int(summary["missing_csv_map_windows"]) + 1
            out_rows.append(row)
            continue

        gaze_path = Path(gaze_csv)
        wheel_path = Path(wheel_csv)
        gaze_exists = gaze_path.exists()
        wheel_exists = wheel_path.exists()
        if not gaze_exists or not wheel_exists:
            if (not gaze_exists) and (not wheel_exists):
                row["status"] = "missing_gaze_wheel_file"
                summary["missing_both_files_windows"] = int(summary["missing_both_files_windows"]) + 1
            elif not gaze_exists:
                row["status"] = "missing_gaze_file"
                summary["missing_gaze_file_windows"] = int(summary["missing_gaze_file_windows"]) + 1
            else:
                row["status"] = "missing_wheel_file"
                summary["missing_wheel_file_windows"] = int(summary["missing_wheel_file_windows"]) + 1
            out_rows.append(row)
            continue

        try:
            if gaze_csv not in gaze_cache:
                gaze_cache[gaze_csv] = load_gaze_csv(gaze_path)
            if wheel_csv not in wheel_cache:
                wheel_cache[wheel_csv] = load_wheel_csv(
                    wheel_path,
                    resolve_uncertain=resolve_uncertain,
                    uncertain_bridge_on_sec=uncertain_bridge_on_sec,
                    uncertain_bridge_mixed_sec=uncertain_bridge_mixed_sec,
                    uncertain_default_state=uncertain_default_state,
                )
        except Exception as exc:
            row["status"] = "csv_load_error"
            row["error"] = str(exc)
            summary["csv_load_error_windows"] = int(summary["csv_load_error_windows"]) + 1
            out_rows.append(row)
            continue

        row.update(
            compute_one_window(
                video_path=video_path,
                w0=w0,
                w1=w1,
                gaze_df=gaze_cache[gaze_csv],
                wheel_df=wheel_cache[wheel_csv],
                max_gap=max_gap,
            )
        )
        row["status"] = "ok"
        summary["ok_windows"] = int(summary["ok_windows"]) + 1
        out_rows.append(row)

    return out_rows, summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["empty"])
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    all_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, object]] = []

    for participant in args.participants:
        paths = participant_paths(participant)
        rows, summary = compute_rows_for_participant(
            participant=participant,
            windows_csv=paths["windows"],
            gaze_map_csv=paths["gaze_map"],
            wheel_map_csv=paths["wheel_map"],
            max_gap=float(args.nearest_wheel_max_gap),
            resolve_uncertain=str(args.resolve_uncertain),
            uncertain_bridge_on_sec=float(args.uncertain_bridge_on_sec),
            uncertain_bridge_mixed_sec=float(args.uncertain_bridge_mixed_sec),
            uncertain_default_state=str(args.uncertain_default_state),
        )
        all_rows.extend(rows)
        summary_rows.append(summary)

    out_csv = Path(args.out_csv).expanduser()
    out_summary_csv = Path(args.out_summary_csv).expanduser()
    write_csv(out_csv, all_rows)
    write_csv(out_summary_csv, summary_rows)

    if not args.skip_xlsx:
        out_xlsx = Path(args.out_xlsx).expanduser()
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        windows_df = pd.DataFrame(all_rows)
        summary_df = pd.DataFrame(summary_rows)
        with pd.ExcelWriter(out_xlsx) as writer:
            windows_df.to_excel(writer, sheet_name="all_windows", index=False)
            summary_df.to_excel(writer, sheet_name="participant_summary", index=False)

    ok_total = sum(int(r.get("ok_windows", 0) or 0) for r in summary_rows)
    total_windows = sum(int(r.get("windows_total", 0) or 0) for r in summary_rows)
    print(f"participants={len(summary_rows)}")
    print(f"windows_total={total_windows}")
    print(f"windows_ok={ok_total}")
    print(f"out_csv={out_csv}")
    print(f"out_summary_csv={out_summary_csv}")
    if not args.skip_xlsx:
        print(f"out_xlsx={Path(args.out_xlsx).expanduser()}")


if __name__ == "__main__":
    main()
