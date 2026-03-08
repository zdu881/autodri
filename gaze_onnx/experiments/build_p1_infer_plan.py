#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build per-video inference plan/map from p1 windows + ROI manifests."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build p1 inference plan from windows CSV")
    p.add_argument("--windows-csv", required=True)
    p.add_argument(
        "--gaze-roi-csv",
        default="gaze_onnx/experiments/manifests/p1_gaze_domains_fixed_roiB.small.csv",
    )
    p.add_argument(
        "--wheel-roi-csv",
        default="data/natural_driving_p1/p1_wheel_domains_fixed_roiA.csv",
    )
    p.add_argument("--out-dir", default="data/natural_driving_p1/infer_p1_windows")
    p.add_argument("--plan-csv", default="data/natural_driving_p1/analysis/p1_infer_plan.csv")
    p.add_argument("--gaze-map-csv", default="data/natural_driving_p1/analysis/p1_gaze_map.csv")
    p.add_argument("--wheel-map-csv", default="data/natural_driving_p1/analysis/p1_wheel_map.csv")
    p.add_argument(
        "--group-by",
        choices=["segment", "video"],
        default="segment",
        help="segment: one infer task per effective segment (recommended); video: merge windows per video",
    )
    return p.parse_args()


def canon_path(s: str) -> str:
    return Path(str(s).strip()).as_posix()


def safe_slug(text: str) -> str:
    t = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))
    t = t.strip("._-")
    return t or "v"


def load_roi_map(path: Path) -> Dict[str, Tuple[int, int, int, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    req = {"video", "roi_x1", "roi_y1", "roi_x2", "roi_y2"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")
    out: Dict[str, Tuple[int, int, int, int]] = {}
    for _, r in df.iterrows():
        key = canon_path(r["video"])
        out[key] = (int(r["roi_x1"]), int(r["roi_y1"]), int(r["roi_x2"]), int(r["roi_y2"]))
    return out


def main() -> None:
    args = parse_args()
    windows_csv = Path(args.windows_csv)
    out_dir = Path(args.out_dir)
    plan_csv = Path(args.plan_csv)
    gaze_map_csv = Path(args.gaze_map_csv)
    wheel_map_csv = Path(args.wheel_map_csv)

    if not windows_csv.exists():
        raise FileNotFoundError(windows_csv)
    wdf = pd.read_csv(windows_csv)
    required = {"video_path", "window_start_sec", "window_end_sec", "video_folder_name"}
    miss = required - set(wdf.columns)
    if miss:
        raise ValueError(f"{windows_csv} missing columns: {sorted(miss)}")
    if args.group_by == "segment" and "segment_uid" not in wdf.columns:
        raise ValueError(f"{windows_csv} missing segment_uid required by --group-by segment")

    gaze_roi_map = load_roi_map(Path(args.gaze_roi_csv))
    wheel_roi_map = load_roi_map(Path(args.wheel_roi_csv))

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_rows: List[Dict[str, str]] = []
    gaze_map_rows: List[Dict[str, str]] = []
    wheel_map_rows: List[Dict[str, str]] = []

    if args.group_by == "video":
        grouped = [(f"video::{k}", g) for k, g in wdf.groupby("video_path")]
    else:
        grouped = [(str(k), g) for k, g in wdf.groupby("segment_uid")]

    for group_key, g in grouped:
        video_path = str(g["video_path"].iloc[0])
        vp = canon_path(video_path)
        folder_name = str(g["video_folder_name"].iloc[0])
        seg_uid = str(g["segment_uid"].iloc[0]) if "segment_uid" in g.columns else ""
        start = float(g["window_start_sec"].min())
        end = float(g["window_end_sec"].max())
        duration = max(0.0, end - start)
        slug = safe_slug(folder_name)
        if args.group_by == "segment" and seg_uid:
            slug = f"{slug}__{safe_slug(seg_uid)}"

        gaze_roi = gaze_roi_map.get(vp)
        wheel_roi = wheel_roi_map.get(vp)
        status = "ok" if (gaze_roi is not None and wheel_roi is not None) else "missing_roi"

        gaze_csv = str((out_dir / f"{slug}.gaze.csv").as_posix())
        gaze_video = str((out_dir / f"{slug}.gaze.mp4").as_posix())
        wheel_csv = str((out_dir / f"{slug}.wheel.csv").as_posix())
        wheel_video = str((out_dir / f"{slug}.wheel.mp4").as_posix())

        plan_rows.append(
            {
                "group_key": group_key,
                "group_mode": args.group_by,
                "segment_uid": seg_uid,
                "video_path": vp,
                "video_folder_name": folder_name,
                "status": status,
                "start_sec": f"{start:.3f}",
                "end_sec": f"{end:.3f}",
                "duration_sec": f"{duration:.3f}",
                "gaze_roi_x1": "" if gaze_roi is None else str(gaze_roi[0]),
                "gaze_roi_y1": "" if gaze_roi is None else str(gaze_roi[1]),
                "gaze_roi_x2": "" if gaze_roi is None else str(gaze_roi[2]),
                "gaze_roi_y2": "" if gaze_roi is None else str(gaze_roi[3]),
                "wheel_roi_x1": "" if wheel_roi is None else str(wheel_roi[0]),
                "wheel_roi_y1": "" if wheel_roi is None else str(wheel_roi[1]),
                "wheel_roi_x2": "" if wheel_roi is None else str(wheel_roi[2]),
                "wheel_roi_y2": "" if wheel_roi is None else str(wheel_roi[3]),
                "gaze_csv": gaze_csv,
                "gaze_video": gaze_video,
                "wheel_csv": wheel_csv,
                "wheel_video": wheel_video,
            }
        )

        gaze_map_rows.append(
            {"group_key": group_key, "segment_uid": seg_uid, "video_path": vp, "gaze_csv": gaze_csv}
        )
        wheel_map_rows.append(
            {"group_key": group_key, "segment_uid": seg_uid, "video_path": vp, "wheel_csv": wheel_csv}
        )

    def write(path: Path, rows: List[Dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["empty"])
            return
        fields = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write(plan_csv, plan_rows)
    write(gaze_map_csv, gaze_map_rows)
    write(wheel_map_csv, wheel_map_rows)

    n_ok = sum(1 for r in plan_rows if r.get("status") == "ok")
    print(f"Plan rows: {len(plan_rows)}  ok={n_ok}  missing_roi={len(plan_rows)-n_ok}")
    print(f"Plan CSV:      {plan_csv}")
    print(f"Gaze map CSV:  {gaze_map_csv}")
    print(f"Wheel map CSV: {wheel_map_csv}")


if __name__ == "__main__":
    main()
