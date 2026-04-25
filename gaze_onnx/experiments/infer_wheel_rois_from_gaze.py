#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Infer wheel ROI manifests from current gaze ROI manifests.

Rules:
1. Dual-panel family: wheel ROI is the opposite fixed slot.
2. Single-panel / driver-region family: wheel ROI = gaze ROI.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


DUAL_A = (0, 0, 1900, 1100)
DUAL_B = (1900, 660, 3300, 1400)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer wheel ROI csvs from gaze ROI csvs")
    p.add_argument("--participant", required=True)
    p.add_argument("--gaze-roi-csv", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def clamp_roi(roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), max(0, w - 1)))
    y1 = max(0, min(int(y1), max(0, h - 1)))
    x2 = max(1, min(int(x2), max(1, w)))
    y2 = max(1, min(int(y2), max(1, h)))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def video_size(video: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"bad video size: {video}")
    return w, h


def infer_wheel_roi(gaze_roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[Tuple[int, int, int, int], str]:
    gx1, gy1, gx2, gy2 = gaze_roi
    if w >= 3000:
        g_center_x = 0.5 * (gx1 + gx2)
        if g_center_x < w * 0.5:
            return clamp_roi(DUAL_B, w, h), "dual_panel_opposite_from_left"
        return clamp_roi(DUAL_A, w, h), "dual_panel_opposite_from_right"
    return clamp_roi(gaze_roi, w, h), "single_panel_same_as_gaze"


def main() -> None:
    args = parse_args()
    src = Path(args.gaze_roi_csv)
    rows = list(csv.DictReader(src.open("r", encoding="utf-8-sig", newline="")))
    out_rows: List[Dict[str, str]] = []
    for r in rows:
        video = str(r.get("video", "")).strip()
        if not video:
            continue
        w, h = video_size(video)
        gaze_roi = (
            int(float(r["roi_x1"])),
            int(float(r["roi_y1"])),
            int(float(r["roi_x2"])),
            int(float(r["roi_y2"])),
        )
        wheel_roi, note = infer_wheel_roi(gaze_roi, w, h)
        out_rows.append(
            {
                "domain_id": str(args.participant),
                "video": video,
                "roi_x1": str(wheel_roi[0]),
                "roi_y1": str(wheel_roi[1]),
                "roi_x2": str(wheel_roi[2]),
                "roi_y2": str(wheel_roi[3]),
                "n_samples": str(r.get("n_samples", "1")),
                "source_swapped": str(r.get("source_swapped", "")),
                "source_uncertain": str(r.get("source_uncertain", "")),
                "inferred_rule": note,
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"rows={len(out_rows)}")
    print(f"out_csv={out_csv}")


if __name__ == "__main__":
    main()
