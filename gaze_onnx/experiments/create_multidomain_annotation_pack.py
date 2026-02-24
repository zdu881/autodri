#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a multi-domain annotation pack from videos and ROI definitions.

Input CSV columns:
- domain_id (e.g., carA_personA)
- video
- roi_x1, roi_y1, roi_x2, roi_y2
- n_samples (optional, default from --samples-per-domain)

Outputs:
- out_dir/images/*.jpg (ROI crops)
- out_dir/manifest.csv (compatible with web_label_tool.py)

Example:
  python gaze_onnx/experiments/create_multidomain_annotation_pack.py \
    --domains-csv gaze_onnx/experiments/manifests/two_domain_videos.csv \
    --out-dir gaze_onnx/experiments/anno_two_domain_v1 \
    --samples-per-domain 600 --seed 42
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2


@dataclass
class DomainItem:
    domain_id: str
    video: str
    roi: Tuple[int, int, int, int]
    n_samples: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create multi-domain annotation pack")
    p.add_argument("--domains-csv", required=True, help="CSV with domain/video/roi definitions")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--samples-per-domain", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jpeg-quality", type=int, default=95)
    return p.parse_args()


def _to_int(x: str, d: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return d


def read_domains(path: str, default_n: int) -> List[DomainItem]:
    rows: List[DomainItem] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = ["domain_id", "video", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]
        for k in required:
            if k not in (r.fieldnames or []):
                raise ValueError(f"Missing column '{k}' in {path}")
        for row in r:
            domain_id = str(row["domain_id"]).strip()
            video = str(row["video"]).strip()
            x1 = _to_int(row["roi_x1"], -1)
            y1 = _to_int(row["roi_y1"], -1)
            x2 = _to_int(row["roi_x2"], -1)
            y2 = _to_int(row["roi_y2"], -1)
            n = _to_int(row.get("n_samples", ""), default_n)
            rows.append(DomainItem(domain_id=domain_id, video=video, roi=(x1, y1, x2, y2), n_samples=max(1, n)))
    return rows


def sample_indices(total_frames: int, n_samples: int, rng: random.Random) -> List[int]:
    if total_frames <= 1:
        return [0]
    n_samples = max(1, min(n_samples, total_frames))

    stride = max(1, total_frames // n_samples)
    picks = []
    for i in range(n_samples):
        base = i * stride
        lo = base
        hi = min(total_frames - 1, base + stride - 1)
        if lo > hi:
            lo = hi
        picks.append(rng.randint(lo, hi))

    picks = sorted(set(picks))
    while len(picks) < n_samples:
        picks.append(rng.randint(0, total_frames - 1))
        picks = sorted(set(picks))
        if len(picks) >= total_frames:
            break

    if len(picks) > n_samples:
        picks = sorted(rng.sample(picks, n_samples))
    return picks


def clamp_roi(roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    domains = read_domains(args.domains_csv, default_n=args.samples_per_domain)
    if not domains:
        raise SystemExit("No domain rows loaded.")

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    manifest_rows = []
    total_saved = 0

    for item in domains:
        cap = cv2.VideoCapture(item.video)
        if not cap.isOpened():
            print(f"[WARN] Skip: cannot open {item.video}")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        roi = clamp_roi(item.roi, w, h)

        picks = sample_indices(total, item.n_samples, rng)
        target_set = set(picks)
        saved = 0
        frame_idx = -1

        while True:
            ok = cap.grab()
            if not ok:
                break
            frame_idx += 1
            if frame_idx not in target_set:
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            x1, y1, x2, y2 = roi
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ts = (frame_idx / fps) if fps > 0 else 0.0
            img_name = f"{item.domain_id}_f{frame_idx:06d}_t{ts:08.3f}.jpg"
            img_path = img_dir / img_name
            cv2.imwrite(str(img_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])

            # Columns keep compatibility with existing web_label_tool.py.
            manifest_rows.append({
                "img": f"images/{img_name}",
                "FrameID": str(frame_idx),
                "Timestamp": f"{ts:.3f}",
                "Pred_Class": "",
                "Raw_Pitch": "",
                "Raw_Yaw": "",
                "Smooth_Pitch": "",
                "Smooth_Yaw": "",
                "Ref_Pitch": "",
                "Ref_Yaw": "",
                "Delta_Pitch": "",
                "Delta_Yaw": "",
                "Domain": item.domain_id,
                "Video": item.video,
                "ROI_X1": str(roi[0]),
                "ROI_Y1": str(roi[1]),
                "ROI_X2": str(roi[2]),
                "ROI_Y2": str(roi[3]),
            })
            saved += 1
            if saved >= len(target_set):
                break

        cap.release()
        total_saved += saved
        print(f"[OK] {item.domain_id}: saved={saved}  video={item.video}")

    manifest_rows.sort(key=lambda r: (r["Domain"], int(float(r["FrameID"]))))
    fieldnames = [
        "img", "FrameID", "Timestamp", "Pred_Class",
        "Raw_Pitch", "Raw_Yaw", "Smooth_Pitch", "Smooth_Yaw",
        "Ref_Pitch", "Ref_Yaw", "Delta_Pitch", "Delta_Yaw",
        "Domain", "Video", "ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    print(f"\nDone. total_saved={total_saved}")
    print(f"Manifest: {manifest_path}")
    print(f"Images:   {img_dir}")


if __name__ == "__main__":
    main()
