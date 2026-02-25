#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare a YOLOv8-cls dataset from labeled frames.

Workflow:
1. Read labels.csv (from web label tool)
2. Open the RAW video (not annotated)
3. Run SCRFD face detection on each labeled frame
4. Crop the face with padding
5. Save into  dataset_dir/{train,val}/{Forward,Non-Forward,In-Car}/

Usage:
  conda run -n adri python gaze_onnx/experiments/prepare_cls_dataset.py \
    --video "6月1日.mp4" \
    --labels gaze_onnx/experiments/samples_smooth4_full_500/labels.csv \
    --scrfd-model models/scrfd_person_2.5g.onnx \
    --out-dir gaze_onnx/experiments/cls_dataset \
    --val-ratio 0.2 \
    --face-pad 0.5 \
    --seed 42
"""

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Allow importing from parent dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_state_onnx import SCRFDDetector, Face


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Raw input video")
    p.add_argument("--labels", required=True, help="labels.csv from web label tool")
    p.add_argument("--scrfd-model", default="models/scrfd_person_2.5g.onnx")
    p.add_argument("--out-dir", required=True, help="Output dataset directory")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--face-pad", type=float, default=0.5,
                   help="Padding around face box as fraction of box size")
    p.add_argument("--crop-size", type=int, default=224,
                   help="Output face crop size (square)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment-minority", action="store_true", default=True,
                   help="Augment minority classes to balance dataset")
    p.add_argument("--target-per-class", type=int, default=0,
                   help="Target samples per class after augmentation (0=auto)")
    return p.parse_args()


def read_labels(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            label = (r.get("label") or r.get("Human_Label") or "").strip()
            if label and label not in ("Unknown", "Skip", ""):
                rows.append({
                    "frame_id": int(float(r["FrameID"])),
                    "label": label,
                })
    return rows


def crop_face_padded(img: np.ndarray, face: Face, pad_frac: float, crop_size: int) -> np.ndarray:
    """Crop face with padding and resize to square."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.xyxy
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * pad_frac
    pad_y = bh * pad_frac

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))

    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)


def augment_image(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simple augmentations: flip, brightness, rotation."""
    out = img.copy()

    # Random horizontal flip
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    # Random brightness
    factor = rng.uniform(0.7, 1.3)
    out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Random small rotation
    angle = rng.uniform(-15, 15)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    return out


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # Read labels
    label_rows = read_labels(args.labels)
    if not label_rows:
        raise SystemExit("No valid labels found")
    print(f"Loaded {len(label_rows)} labeled frames")

    # Count per class
    from collections import Counter
    class_counts = Counter(r["label"] for r in label_rows)
    print("Class distribution:", dict(class_counts.most_common()))

    # Init SCRFD
    scrfd = SCRFDDetector(args.scrfd_model, conf_thresh=0.3, min_face_size=30)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video} ({total_frames} frames)")

    # Create output dirs
    classes = sorted(class_counts.keys())
    out_dir = Path(args.out_dir)
    for split in ("train", "val"):
        for cls in classes:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Split into train/val per class (stratified)
    label_by_class: Dict[str, List[Dict]] = {}
    for r in label_rows:
        label_by_class.setdefault(r["label"], []).append(r)

    train_rows = []
    val_rows = []
    for cls, rows in label_by_class.items():
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_ratio))
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])

    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    # Extract face crops
    def extract_and_save(rows_list, split_name):
        saved = 0
        no_face = 0
        for r in rows_list:
            fid = r["frame_id"]
            label = r["label"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            faces = scrfd.detect(frame)
            if not faces:
                no_face += 1
                continue

            # Take the best (highest score) face
            face = faces[0]
            crop = crop_face_padded(frame, face, args.face_pad, args.crop_size)

            fname = f"frame{fid:06d}.jpg"
            save_path = out_dir / split_name / label / fname
            cv2.imwrite(str(save_path), crop)
            saved += 1

        print(f"  {split_name}: saved {saved}, no_face {no_face}")
        return saved

    print("Extracting face crops...")
    extract_and_save(val_rows, "val")
    n_train_real = extract_and_save(train_rows, "train")

    # Augmentation for minority classes
    if args.augment_minority:
        train_class_counts = Counter(r["label"] for r in train_rows)
        max_count = max(train_class_counts.values())
        target = args.target_per_class if args.target_per_class > 0 else max_count

        print(f"\nAugmenting minority classes to ~{target} samples each...")
        for cls in classes:
            cls_dir = out_dir / "train" / cls
            existing = list(cls_dir.glob("*.jpg"))
            n_existing = len(existing)
            if n_existing >= target:
                continue

            n_aug = target - n_existing
            print(f"  {cls}: {n_existing} existing, generating {n_aug} augmented")

            aug_idx = 0
            for _ in range(n_aug):
                src_path = rng.choice(existing)
                src_img = cv2.imread(str(src_path))
                if src_img is None:
                    continue
                aug_img = augment_image(src_img, rng)
                aug_name = f"aug_{aug_idx:05d}.jpg"
                cv2.imwrite(str(cls_dir / aug_name), aug_img)
                aug_idx += 1

    # Print final dataset summary
    print("\n=== Final Dataset ===")
    for split in ("train", "val"):
        print(f"{split}:")
        for cls in classes:
            cls_dir = out_dir / split / cls
            n = len(list(cls_dir.glob("*.jpg")))
            print(f"  {cls}: {n}")

    cap.release()
    print(f"\nDataset ready at: {out_dir}")


if __name__ == "__main__":
    main()
