#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare one combined gaze ROI review pack from current infer plans."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, List

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare gaze ROI review pack")
    p.add_argument("--participant", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--grid-step", type=int, default=220)
    p.add_argument("--sample-position", choices=("first", "middle"), default="middle")
    return p.parse_args()


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
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = Path(text).stem
    stem = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    stem = stem[:48] if stem else "video"
    return f"{stem}__{h}"


def pick_frame_index(total_frames: int, mode: str) -> int:
    if total_frames <= 1:
        return 0
    if mode == "middle":
        return max(0, (total_frames - 1) // 2)
    return 0


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    out_dir = Path(args.out_dir).resolve()
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    if args.participant == "p1":
        plan = root / "data/natural_driving_p1/analysis/p1_infer_plan.segment.csv"
    else:
        plan = root / f"data/natural_driving/{args.participant}/analysis/{args.participant}_infer_plan.current.csv"

    rows = list(csv.DictReader(plan.open("r", encoding="utf-8-sig", newline="")))
    seen = set()
    out_rows: List[Dict[str, str]] = []

    for r in rows:
        video = str(r.get("video_path", "")).strip()
        if not video or video in seen:
            continue
        seen.add(video)
        cap = cv2.VideoCapture(video)
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
            continue

        rel = f"{args.participant}/{Path(video).parent.name}/{Path(video).name}"
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

        out_rows.append(
            {
                "video_rel": rel,
                "video_abs": video,
                "ref_raw": str(raw_path.relative_to(out_dir)),
                "ref_grid": str(grid_path.relative_to(out_dir)),
                "frame_idx": str(idx),
                "timestamp_sec": f"{(idx / fps) if fps > 0 else 0.0:.3f}",
                "width": str(w),
                "height": str(h),
                "roi_x1": str(r["gaze_roi_x1"]),
                "roi_y1": str(r["gaze_roi_y1"]),
                "roi_x2": str(r["gaze_roi_x2"]),
                "roi_y2": str(r["gaze_roi_y2"]),
                "roi_note": "current_gaze_roi",
            }
        )

    manifest = out_dir / "roi_label_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        fields = list(out_rows[0].keys()) if out_rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    with (out_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write("Gaze ROI review pack.\n")
        f.write("Current ROI values are prefilled.\n")

    print(f"rows={len(out_rows)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
