#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate ROI reference images with coordinate grid and segment IDs.

Use this to quickly provide approximate ROI coordinates (x1 y1 x2 y2).

Examples:
  python gaze_onnx/experiments/make_roi_reference.py \
    --video "6月1日.mp4" \
    --out-dir gaze_onnx/experiments/roi_refs/domain_a

  python gaze_onnx/experiments/make_roi_reference.py \
    --video "12月23日(1).mp4" \
    --out-dir gaze_onnx/experiments/roi_refs/domain_b \
    --num-frames 4 --grid-step 240 --segments-x 4 --segments-y 3
"""

import argparse
import os
from pathlib import Path
from typing import List

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ROI reference frames")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--num-frames", type=int, default=3, help="How many evenly spaced frames to export")
    p.add_argument("--grid-step", type=int, default=200, help="Pixel spacing of coordinate grid")
    p.add_argument("--segments-x", type=int, default=3, help="Segment columns")
    p.add_argument("--segments-y", type=int, default=3, help="Segment rows")
    p.add_argument("--line-thickness", type=int, default=2)
    return p.parse_args()


def sample_indices(n_total: int, n_pick: int) -> List[int]:
    if n_total <= 0:
        return [0]
    n_pick = max(1, int(n_pick))
    if n_pick == 1:
        return [0]
    out = []
    for i in range(n_pick):
        idx = int(round(i * (n_total - 1) / (n_pick - 1)))
        out.append(max(0, min(n_total - 1, idx)))
    return sorted(set(out))


def draw_coordinate_grid(img, step: int, thickness: int) -> None:
    h, w = img.shape[:2]
    step = max(40, int(step))

    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), (255, 255, 255), 1)
        cv2.putText(img, str(x), (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), (255, 255, 255), 1)
        cv2.putText(img, str(y), (4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 255, 255), max(1, thickness))


def draw_segments(img, n_x: int, n_y: int, thickness: int) -> None:
    h, w = img.shape[:2]
    n_x = max(1, int(n_x))
    n_y = max(1, int(n_y))
    col_w = w / n_x
    row_h = h / n_y

    for i in range(1, n_x):
        x = int(round(i * col_w))
        cv2.line(img, (x, 0), (x, h - 1), (0, 255, 255), max(1, thickness))

    for j in range(1, n_y):
        y = int(round(j * row_h))
        cv2.line(img, (0, y), (w - 1, y), (0, 255, 255), max(1, thickness))

    for j in range(n_y):
        for i in range(n_x):
            x1 = int(round(i * col_w))
            y1 = int(round(j * row_h))
            x2 = int(round((i + 1) * col_w))
            y2 = int(round((j + 1) * row_h))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = f"R{j+1}C{i+1}"
            cv2.putText(img, label, (max(4, cx - 35), max(22, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    picks = sample_indices(total, args.num_frames)
    manifest_path = out_dir / "roi_reference_manifest.csv"

    rows = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        ts = (idx / fps) if fps > 0 else 0.0
        raw_name = f"frame_{idx:06d}_raw.jpg"
        grid_name = f"frame_{idx:06d}_grid.jpg"

        raw = frame.copy()
        grid = frame.copy()
        draw_coordinate_grid(grid, step=args.grid_step, thickness=args.line_thickness)
        draw_segments(grid, n_x=args.segments_x, n_y=args.segments_y, thickness=args.line_thickness)

        cv2.putText(grid, f"frame={idx}  t={ts:.2f}s  size={width}x{height}", (16, height - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / raw_name), raw)
        cv2.imwrite(str(out_dir / grid_name), grid)

        rows.append((idx, f"{ts:.3f}", raw_name, grid_name, width, height))

    cap.release()

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        f.write("FrameID,Timestamp,RawImage,GridImage,Width,Height\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    readme = out_dir / "README.txt"
    with readme.open("w", encoding="utf-8") as f:
        f.write("ROI reference generated.\n")
        f.write("Please provide approximate ROI coordinates: x1 y1 x2 y2\n")
        f.write("Use *_grid.jpg with coordinate lines and segment IDs (R#C#).\n")

    print(f"Video: {args.video}")
    print(f"Frames exported: {len(rows)}")
    print(f"Out dir: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
