#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare a dual-ROI review pack for p1.

The pack contains one item per unique p1 video with:
- raw frame
- grid frame
- current gaze ROI
- current wheel ROI

Used together with serve_dual_roi_review.py for manual ROI correction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare p1 dual ROI review pack")
    p.add_argument(
        "--plan-csv",
        default="data/natural_driving_p1/analysis/p1_infer_plan.segment.csv",
        help="p1 infer plan with current gaze/wheel ROI columns",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--grid-step", type=int, default=220)
    p.add_argument("--sample-position", choices=("first", "middle"), default="middle")
    return p.parse_args()


def pick_frame_index(total_frames: int, mode: str) -> int:
    if total_frames <= 1:
        return 0
    if mode == "middle":
        return max(0, (total_frames - 1) // 2)
    return 0


def draw_grid(img, step: int) -> None:
    h, w = img.shape[:2]
    step = max(40, int(step))
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), (255, 255, 255), 1)
        cv2.putText(img, str(x), (x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), (255, 255, 255), 1)
        cv2.putText(img, str(y), (4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 255, 255), 2)


def safe_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = Path(text).stem
    stem = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    stem = stem[:48] if stem else "video"
    return f"{stem}__{digest}"


def as_text_int(v) -> str:
    if pd.isna(v):
        return ""
    return str(int(float(v)))


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    plan_csv = root / args.plan_csv
    out_dir = Path(args.out_dir).resolve()
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    if not plan_csv.exists():
        raise FileNotFoundError(plan_csv)

    df = pd.read_csv(plan_csv)
    if "video_path" not in df.columns:
        raise ValueError(f"{plan_csv} missing video_path")

    rows_out: List[Dict[str, str]] = []
    uniq = df.drop_duplicates(subset=["video_path"]).copy()

    for _, r in uniq.iterrows():
        video_rel = str(r["video_path"]).strip()
        if not video_rel:
            continue
        video = (root / video_rel).resolve() if not Path(video_rel).is_absolute() else Path(video_rel)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[WARN] skip unreadable video: {video}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        idx = pick_frame_index(total, args.sample_position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"[WARN] skip unreadable frame: {video}")
            continue

        rel = f"p1/{video.parent.name}/{video.name}"
        sid = safe_id(rel)
        raw_name = f"{sid}__raw.jpg"
        grid_name = f"{sid}__grid.jpg"
        raw_path = refs_dir / raw_name
        grid_path = refs_dir / grid_name

        raw = frame.copy()
        grid = frame.copy()
        draw_grid(grid, int(args.grid_step))
        cv2.imwrite(str(raw_path), raw)
        cv2.imwrite(str(grid_path), grid)

        rows_out.append(
            {
                "video_rel": rel,
                "video_abs": str(video),
                "ref_raw": str(raw_path.relative_to(out_dir)),
                "ref_grid": str(grid_path.relative_to(out_dir)),
                "frame_idx": str(idx),
                "timestamp_sec": f"{(idx / fps) if fps > 0 else 0.0:.3f}",
                "width": str(w),
                "height": str(h),
                "gaze_roi_x1": as_text_int(r.get("gaze_roi_x1")),
                "gaze_roi_y1": as_text_int(r.get("gaze_roi_y1")),
                "gaze_roi_x2": as_text_int(r.get("gaze_roi_x2")),
                "gaze_roi_y2": as_text_int(r.get("gaze_roi_y2")),
                "wheel_roi_x1": as_text_int(r.get("wheel_roi_x1")),
                "wheel_roi_y1": as_text_int(r.get("wheel_roi_y1")),
                "wheel_roi_x2": as_text_int(r.get("wheel_roi_x2")),
                "wheel_roi_y2": as_text_int(r.get("wheel_roi_y2")),
                "roi_note": "from_p1_infer_plan_segment_current",
            }
        )

    manifest = out_dir / "roi_label_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows_out[0].keys()) if rows_out else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    (out_dir / "README.txt").write_text(
        "P1 dual ROI review pack.\n"
        "Edit gaze ROI and wheel ROI for each video in the web UI.\n",
        encoding="utf-8",
    )

    print(f"rows={len(rows_out)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
