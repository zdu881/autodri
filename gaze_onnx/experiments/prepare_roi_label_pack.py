#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare ROI reference images and a fill-in manifest for manual ROI labeling.

This script is designed for step-1 of the workflow:
1) Human labels model ROI windows for each video.
2) Human labels few-shot samples.
3) Run model-assisted labeling/inference.
"""

import argparse
import csv
import hashlib
from pathlib import Path
from typing import List, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ROI label pack from a folder of videos")
    p.add_argument("--videos-root", required=True, help="Root directory containing videos recursively")
    p.add_argument("--out-dir", required=True, help="Output folder for refs and manifest")
    p.add_argument("--glob", default="*.mp4", help="Video glob pattern")
    p.add_argument("--grid-step", type=int, default=200, help="Grid spacing in pixels")
    p.add_argument("--line-thickness", type=int, default=2)
    p.add_argument(
        "--sample-position",
        choices=("first", "middle"),
        default="first",
        help="Which frame to export as ROI reference",
    )
    return p.parse_args()


def find_videos(root: Path, pattern: str) -> List[Path]:
    out = sorted(root.rglob(pattern))
    return [p for p in out if p.is_file()]


def draw_grid(img, step: int, thickness: int) -> None:
    h, w = img.shape[:2]
    step = max(40, int(step))

    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), (255, 255, 255), 1)
        cv2.putText(img, str(x), (x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), (255, 255, 255), 1)
        cv2.putText(img, str(y), (4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 255, 255), max(1, thickness))


def safe_id(rel_path: str) -> str:
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:10]
    stem = Path(rel_path).stem
    stem = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    stem = stem[:60] if stem else "video"
    return f"{stem}__{h}"


def pick_frame_index(total_frames: int, mode: str) -> int:
    if total_frames <= 1:
        return 0
    if mode == "middle":
        return max(0, (total_frames - 1) // 2)
    return 0


def main() -> None:
    args = parse_args()
    videos_root = Path(args.videos_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(videos_root, args.glob)
    if not videos:
        raise SystemExit(f"No videos found in {videos_root} with pattern {args.glob}")

    rows: List[dict] = []
    for v in videos:
        rel = str(v.relative_to(videos_root))
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"[WARN] skip (open failed): {rel}")
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
            print(f"[WARN] skip (read failed): {rel}")
            continue

        ts = (idx / fps) if fps > 0 else 0.0
        vid = safe_id(rel)
        raw_name = f"{vid}__raw.jpg"
        grid_name = f"{vid}__grid.jpg"
        raw_path = refs_dir / raw_name
        grid_path = refs_dir / grid_name

        raw = frame.copy()
        grid = frame.copy()
        draw_grid(grid, step=args.grid_step, thickness=args.line_thickness)
        cv2.putText(
            grid,
            f"{rel} | frame={idx} t={ts:.2f}s size={w}x{h}",
            (16, max(28, h - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(raw_path), raw)
        cv2.imwrite(str(grid_path), grid)

        rows.append(
            {
                "video_rel": rel,
                "video_abs": str(v),
                "ref_raw": str(raw_path.relative_to(out_dir)),
                "ref_grid": str(grid_path.relative_to(out_dir)),
                "frame_idx": idx,
                "timestamp_sec": f"{ts:.3f}",
                "width": w,
                "height": h,
                "roi_x1": "",
                "roi_y1": "",
                "roi_x2": "",
                "roi_y2": "",
                "roi_note": "",
            }
        )

    manifest = out_dir / "roi_label_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "video_rel",
            "video_abs",
            "ref_raw",
            "ref_grid",
            "frame_idx",
            "timestamp_sec",
            "width",
            "height",
            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2",
            "roi_note",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    readme = out_dir / "README.txt"
    with readme.open("w", encoding="utf-8") as f:
        f.write("ROI label pack ready.\n")
        f.write("1) Open refs/*__grid.jpg in VS Code.\n")
        f.write("2) Fill roi_x1,roi_y1,roi_x2,roi_y2 in roi_label_manifest.csv.\n")
        f.write("3) Keep coordinates in original frame pixel space.\n")

    print(f"videos_total={len(videos)}")
    print(f"rows_written={len(rows)}")
    print(f"out_dir={out_dir}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
