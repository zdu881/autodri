#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rebuild an existing annotation pack with the same sampled frames but new ROI crops.

Use this when an old annotation pack was built on the wrong ROI but the sampled
frames and labels are still valuable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild an annotation pack using new ROI coordinates")
    p.add_argument("--src-pack", required=True, help="Existing annotation pack directory")
    p.add_argument("--roi-csv", required=True, help="CSV with columns video,roi_x1,roi_y1,roi_x2,roi_y2")
    p.add_argument("--out-pack", required=True, help="Output pack directory")
    p.add_argument("--jpeg-quality", type=int, default=95)
    return p.parse_args()


def _to_int(x: str, d: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return d


def read_roi_map(path: Path) -> Dict[str, Tuple[int, int, int, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, Tuple[int, int, int, int]] = {}
    for r in rows:
        video = str(r.get("video", "")).strip()
        if not video:
            continue
        out[video] = (
            _to_int(r.get("roi_x1", "0")),
            _to_int(r.get("roi_y1", "0")),
            _to_int(r.get("roi_x2", "0")),
            _to_int(r.get("roi_y2", "0")),
        )
    return out


def clamp_roi(roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), max(0, w - 1)))
    y1 = max(0, min(int(y1), max(0, h - 1)))
    x2 = max(1, min(int(x2), max(1, w)))
    y2 = max(1, min(int(y2), max(1, h)))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    src_pack = Path(args.src_pack)
    out_pack = Path(args.out_pack)
    out_img = out_pack / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    manifest_path = src_pack / "manifest.csv"
    labels_path = src_pack / "labels.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    roi_map = read_roi_map(Path(args.roi_csv))

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    video_caps: Dict[str, cv2.VideoCapture] = {}
    rebuilt_rows: List[dict] = []
    saved = 0

    try:
        for r in manifest_rows:
            video = str(r.get("Video", "")).strip()
            img_rel = str(r.get("img", "")).strip()
            frame_id = _to_int(r.get("FrameID", "0"))
            if not video or not img_rel:
                continue
            roi = roi_map.get(video)
            if roi is None:
                raise KeyError(f"Missing ROI for video: {video}")

            if video not in video_caps:
                cap = cv2.VideoCapture(video)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open video: {video}")
                video_caps[video] = cap
            cap = video_caps[video]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame {frame_id} from {video}")

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = clamp_roi(roi, w, h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raise RuntimeError(f"Empty crop for frame {frame_id} from {video}")

            out_img_path = out_pack / img_rel
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            ok_write = cv2.imwrite(str(out_img_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
            if not ok_write:
                raise RuntimeError(f"Failed to write {out_img_path}")

            row2 = dict(r)
            row2["ROI_X1"] = str(x1)
            row2["ROI_Y1"] = str(y1)
            row2["ROI_X2"] = str(x2)
            row2["ROI_Y2"] = str(y2)
            rebuilt_rows.append(row2)
            saved += 1
    finally:
        for cap in video_caps.values():
            cap.release()

    # copy labels.csv exactly to preserve human labels on the same sampled frames
    labels_bytes = labels_path.read_bytes()
    (out_pack / "labels.csv").write_bytes(labels_bytes)

    # rewrite manifest with updated ROI columns
    fieldnames = list(rebuilt_rows[0].keys()) if rebuilt_rows else []
    with (out_pack / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rebuilt_rows)

    with (out_pack / "README.txt").open("w", encoding="utf-8") as f:
        f.write("Rebuilt annotation pack with the same sampled frames and preserved human labels.\n")
        f.write("Only the ROI crop was replaced using the provided ROI CSV.\n")

    print(f"saved_images={saved}")
    print(f"out_pack={out_pack}")


if __name__ == "__main__":
    main()
