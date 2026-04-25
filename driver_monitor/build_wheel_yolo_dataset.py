#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a YOLO detection dataset from hand_on_wheel.py --det-csv outputs.

The source detection CSV stores sampled-frame detections from GroundingDINO.
This script extracts frame images and writes YOLO txt labels for classes:
  0 = hand
  1 = steering_wheel
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd


@dataclass
class Det:
    class_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class FrameSample:
    video_path: str
    video_frame: int
    roi: Tuple[int, int, int, int]
    dets: List[Det]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build wheel YOLO dataset from det-csv files")
    p.add_argument(
        "--det-csv",
        action="append",
        required=True,
        help="Det CSV path or glob pattern. Can be passed multiple times.",
    )
    p.add_argument("--out-dir", required=True, help="Output dataset root")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--include-negatives",
        action="store_true",
        help="Include frames with no detections as empty-label images",
    )
    p.add_argument(
        "--neg-keep-prob",
        type=float,
        default=0.15,
        help="Keep probability for negative frames (only when --include-negatives)",
    )
    p.add_argument(
        "--use-roi-crop",
        action="store_true",
        help="Crop to ROI before writing images/labels (recommended).",
    )
    p.add_argument(
        "--min-box-size",
        type=float,
        default=3.0,
        help="Minimum pixel width/height after clipping to keep a box.",
    )
    return p.parse_args()


def expand_csv_patterns(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            out.extend(Path(x) for x in hits)
        else:
            p = Path(pat)
            if p.exists():
                out.append(p)
    dedup = sorted(set(out))
    if not dedup:
        raise FileNotFoundError("No det-csv files matched --det-csv patterns")
    return dedup


def safe_stem(video_path: str) -> str:
    p = Path(video_path)
    stem = p.stem
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{h}"


def collect_samples(csv_paths: List[Path]) -> List[FrameSample]:
    need_cols = {
        "video_path",
        "video_frame",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p)
        miss = need_cols - set(df.columns)
        if miss:
            raise ValueError(f"{p} missing columns: {sorted(miss)}")
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)

    all_df["video_frame"] = pd.to_numeric(all_df["video_frame"], errors="coerce")
    all_df["class_id"] = pd.to_numeric(all_df["class_id"], errors="coerce")
    for c in ("x1", "y1", "x2", "y2", "confidence"):
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
    for c in ("roi_x1", "roi_y1", "roi_x2", "roi_y2"):
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    all_df = all_df.dropna(subset=["video_path", "video_frame", "roi_x1", "roi_y1", "roi_x2", "roi_y2"])
    all_df["video_path"] = all_df["video_path"].astype(str).str.strip()
    all_df = all_df[all_df["video_path"] != ""].copy()

    samples: List[FrameSample] = []
    grouped = all_df.groupby(["video_path", "video_frame"], sort=False)
    for (video_path, video_frame), g in grouped:
        r0 = g.iloc[0]
        roi = (
            int(r0["roi_x1"]),
            int(r0["roi_y1"]),
            int(r0["roi_x2"]),
            int(r0["roi_y2"]),
        )
        dets: List[Det] = []
        for _, r in g.iterrows():
            cid = int(r["class_id"]) if pd.notna(r["class_id"]) else -1
            if cid not in (0, 1):
                continue
            if not all(pd.notna(r[c]) for c in ("x1", "y1", "x2", "y2")):
                continue
            dets.append(
                Det(
                    class_id=cid,
                    conf=float(r["confidence"]) if pd.notna(r["confidence"]) else 0.0,
                    x1=float(r["x1"]),
                    y1=float(r["y1"]),
                    x2=float(r["x2"]),
                    y2=float(r["y2"]),
                )
            )
        samples.append(
            FrameSample(
                video_path=str(video_path),
                video_frame=int(video_frame),
                roi=roi,
                dets=dets,
            )
        )
    return samples


def clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(x1), float(w - 1)))
    y1 = max(0.0, min(float(y1), float(h - 1)))
    x2 = max(0.0, min(float(x2), float(w - 1)))
    y2 = max(0.0, min(float(y2), float(h - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx = ((x1 + x2) * 0.5) / max(1.0, float(w))
    cy = ((y1 + y2) * 0.5) / max(1.0, float(h))
    bw = (x2 - x1) / max(1.0, float(w))
    bh = (y2 - y1) / max(1.0, float(h))
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    out_dir = Path(args.out_dir)
    csv_paths = expand_csv_patterns(args.det_csv)
    samples = collect_samples(csv_paths)
    if not samples:
        raise SystemExit("No frame samples found in det-csv inputs.")

    # Keep positives always, negatives by probability.
    kept: List[FrameSample] = []
    for s in samples:
        if s.dets:
            kept.append(s)
        elif args.include_negatives and (rng.random() <= float(args.neg_keep_prob)):
            kept.append(s)
    if not kept:
        raise SystemExit("No samples kept after negative filtering.")

    rng.shuffle(kept)
    n_total = len(kept)
    n_val = max(1, int(round(n_total * float(args.val_ratio))))
    n_val = min(n_val, n_total - 1) if n_total > 1 else n_total
    val_keys = set(id(x) for x in kept[:n_val])

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    cap_cache: Dict[str, cv2.VideoCapture] = {}
    manifest_rows: List[Dict[str, str]] = []
    n_img = 0
    n_pos = 0
    n_neg = 0
    n_box = 0

    for s in kept:
        split = "val" if id(s) in val_keys else "train"
        cap = cap_cache.get(s.video_path)
        if cap is None:
            cap = cv2.VideoCapture(s.video_path)
            if not cap.isOpened():
                print(f"[WARN] skip unreadable video: {s.video_path}")
                continue
            cap_cache[s.video_path] = cap

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s.video_frame))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] skip unreadable frame: {s.video_path} frame={s.video_frame}")
            continue
        fh, fw = frame.shape[:2]

        roi_x1, roi_y1, roi_x2, roi_y2 = s.roi
        roi_x1 = max(0, min(int(roi_x1), fw - 1))
        roi_y1 = max(0, min(int(roi_y1), fh - 1))
        roi_x2 = max(1, min(int(roi_x2), fw))
        roi_y2 = max(1, min(int(roi_y2), fh))
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            print(f"[WARN] invalid roi skip: {s.video_path} frame={s.video_frame} roi={s.roi}")
            continue

        if args.use_roi_crop:
            img = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            ox, oy = roi_x1, roi_y1
        else:
            img = frame
            ox, oy = 0, 0
        ih, iw = img.shape[:2]
        if ih <= 1 or iw <= 1:
            continue

        stem = safe_stem(s.video_path)
        name = f"{stem}__f{s.video_frame:07d}"
        img_path = out_dir / "images" / split / f"{name}.jpg"
        lbl_path = out_dir / "labels" / split / f"{name}.txt"

        lines: List[str] = []
        for d in s.dets:
            x1 = d.x1 - ox
            y1 = d.y1 - oy
            x2 = d.x2 - ox
            y2 = d.y2 - oy
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, iw, ih)
            bw = x2 - x1
            bh = y2 - y1
            if bw < float(args.min_box_size) or bh < float(args.min_box_size):
                continue
            lines.append(to_yolo_line(d.class_id, x1, y1, x2, y2, iw, ih))
            n_box += 1

        if lines:
            n_pos += 1
        else:
            n_neg += 1

        cv2.imwrite(str(img_path), img)
        with lbl_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        manifest_rows.append(
            {
                "split": split,
                "image_path": str(img_path),
                "label_path": str(lbl_path),
                "video_path": s.video_path,
                "video_frame": str(s.video_frame),
                "roi_x1": str(roi_x1),
                "roi_y1": str(roi_y1),
                "roi_x2": str(roi_x2),
                "roi_y2": str(roi_y2),
                "num_boxes": str(len(lines)),
            }
        )
        n_img += 1

    for cap in cap_cache.values():
        cap.release()

    if not manifest_rows:
        raise SystemExit("No samples written. Please check video paths and frame ids.")

    # data.yaml for Ultralytics
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: hand",
                "  1: steering_wheel",
                "",
            ]
        ),
        encoding="utf-8",
    )

    man_csv = out_dir / "manifest.csv"
    with man_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"Det CSV files: {len(csv_paths)}")
    print(f"Samples in: {len(samples)}  kept: {len(kept)}")
    print(f"Images written: {n_img}  positive: {n_pos}  negative: {n_neg}  boxes: {n_box}")
    print(f"Dataset root: {out_dir}")
    print(f"data.yaml: {data_yaml}")
    print(f"manifest:  {man_csv}")


if __name__ == "__main__":
    main()

