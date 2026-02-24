#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Suggest ROI coordinates from videos using SCRFD face detections.

For each domain row, sample frames across time, detect faces, and estimate a stable
ROI that covers the main driver's face region with margins.

Input CSV columns:
- domain_id, video, roi_x1, roi_y1, roi_x2, roi_y2

Output CSV appends:
- sug_x1, sug_y1, sug_x2, sug_y2, num_face_samples

Example:
  python gaze_onnx/experiments/suggest_roi_scrfd.py \
    --domains-csv gaze_onnx/experiments/manifests/two_domain_videos.csv \
    --scrfd-model models/scrfd_person_2.5g.onnx \
    --out-csv gaze_onnx/experiments/manifests/two_domain_videos.suggested.csv
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Reuse existing detector implementation
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_state_onnx import SCRFDDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Suggest ROI from SCRFD detections")
    p.add_argument("--domains-csv", required=True)
    p.add_argument("--scrfd-model", default="models/scrfd_person_2.5g.onnx")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--samples", type=int, default=80, help="Frames sampled per video")
    p.add_argument("--face-conf", type=float, default=0.5)
    p.add_argument("--expand", type=float, default=1.8, help="Expand suggested box around robust face box")
    return p.parse_args()


def to_int(x: str, d: int = -1) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return d


def sample_indices(total: int, n: int) -> List[int]:
    if total <= 1:
        return [0]
    n = max(1, min(n, total))
    out = []
    for i in range(n):
        out.append(int(round(i * (total - 1) / max(1, n - 1))))
    return sorted(set(out))


def robust_roi_from_boxes(boxes: np.ndarray, w: int, h: int, expand: float) -> Tuple[int, int, int, int]:
    # boxes Nx4 in xyxy
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Robust bounds by percentiles
    rx1 = float(np.percentile(x1, 10))
    ry1 = float(np.percentile(y1, 10))
    rx2 = float(np.percentile(x2, 90))
    ry2 = float(np.percentile(y2, 90))

    cx = 0.5 * (rx1 + rx2)
    cy = 0.5 * (ry1 + ry2)
    bw = max(20.0, (rx2 - rx1) * float(expand))
    bh = max(20.0, (ry2 - ry1) * float(expand))

    nx1 = int(max(0, min(w - 1, math.floor(cx - bw * 0.5))))
    ny1 = int(max(0, min(h - 1, math.floor(cy - bh * 0.5))))
    nx2 = int(max(1, min(w, math.ceil(cx + bw * 0.5))))
    ny2 = int(max(1, min(h, math.ceil(cy + bh * 0.5))))

    if nx2 <= nx1 or ny2 <= ny1:
        return 0, 0, w, h
    return nx1, ny1, nx2, ny2


def process_video(detector: SCRFDDetector, video_path: str, n_samples: int, expand: float) -> Dict[str, object]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "reason": "open_failed"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    idxs = sample_indices(total, n_samples)

    all_boxes = []
    for fid in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        faces = detector.detect(frame)
        if not faces:
            continue
        # take top face candidate
        f = faces[0]
        all_boxes.append(np.asarray(f.xyxy, dtype=np.float32))

    cap.release()

    if not all_boxes:
        return {
            "ok": True,
            "num_face_samples": 0,
            "sug": (0, 0, w, h),
            "size": (w, h),
        }

    boxes = np.stack(all_boxes, axis=0)
    sug = robust_roi_from_boxes(boxes, w, h, expand=float(expand))
    return {
        "ok": True,
        "num_face_samples": int(boxes.shape[0]),
        "sug": sug,
        "size": (w, h),
    }


def main() -> None:
    args = parse_args()

    detector = SCRFDDetector(
        args.scrfd_model,
        conf_thresh=float(args.face_conf),
        min_face_size=30,
        pre_nms_topk=800,
    )

    in_rows = []
    with open(args.domains_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            in_rows.append(dict(row))

    out_rows = []
    for row in in_rows:
        video = str(row.get("video", "")).strip()
        domain = str(row.get("domain_id", "")).strip()

        res = process_video(detector, video_path=video, n_samples=int(args.samples), expand=float(args.expand))
        if not res.get("ok"):
            print(f"[WARN] {domain}: failed ({res.get('reason')})")
            row.update({
                "sug_x1": "", "sug_y1": "", "sug_x2": "", "sug_y2": "",
                "num_face_samples": "0",
            })
            out_rows.append(row)
            continue

        sx1, sy1, sx2, sy2 = res["sug"]
        nfs = int(res["num_face_samples"])
        print(f"[OK] {domain}: suggest=({sx1},{sy1},{sx2},{sy2}) face_samples={nfs}")

        row.update({
            "sug_x1": str(sx1),
            "sug_y1": str(sy1),
            "sug_x2": str(sx2),
            "sug_y2": str(sy2),
            "num_face_samples": str(nfs),
        })
        out_rows.append(row)

    # write CSV with stable field order
    fields = list(in_rows[0].keys()) if in_rows else []
    for extra in ["sug_x1", "sug_y1", "sug_x2", "sug_y2", "num_face_samples"]:
        if extra not in fields:
            fields.append(extra)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
