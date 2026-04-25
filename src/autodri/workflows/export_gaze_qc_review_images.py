#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from autodri.common.paths import participant_analysis_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export human-readable review images for gaze QC windows")
    p.add_argument("--qc-csv", required=True, help="Window-level QC CSV to visualize")
    p.add_argument("--out-dir", required=True, help="Output directory for rendered images")
    p.add_argument(
        "--plan-csv",
        action="append",
        default=[],
        help="Optional infer-plan CSV(s) with gaze ROI columns. Repeatable. Auto-inferred when omitted.",
    )
    p.add_argument("--thumb-cols", type=int, default=4, help="Columns per participant contact sheet")
    p.add_argument("--jpg-quality", type=int, default=92, help="JPEG quality for exported images")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def infer_plan_paths(qc_rows: List[Dict[str, str]]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for row in qc_rows:
        participant = str(row.get("participant", "")).strip()
        if not participant or participant in seen:
            continue
        seen.add(participant)
        if participant == "p1":
            p = participant_analysis_dir("p1") / "p1_infer_plan.segment.csv"
        else:
            p = participant_analysis_dir(participant) / f"{participant}_infer_plan.current.csv"
        out.append(p)
    return out


def load_roi_by_segment(plan_paths: List[Path]) -> Dict[str, Tuple[int, int, int, int]]:
    out: Dict[str, Tuple[int, int, int, int]] = {}
    for path in plan_paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            seg = str(row.get("segment_uid", "")).strip()
            if not seg:
                continue
            try:
                roi = (
                    int(float(row.get("gaze_roi_x1", "0"))),
                    int(float(row.get("gaze_roi_y1", "0"))),
                    int(float(row.get("gaze_roi_x2", "0"))),
                    int(float(row.get("gaze_roi_y2", "0"))),
                )
            except Exception:
                continue
            if roi[2] > roi[0] and roi[3] > roi[1]:
                out[seg] = roi
    return out


def read_frame(video_path: Path, at_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    attempts = [
        float(at_sec),
        max(0.0, float(at_sec) - 0.2),
        max(0.0, float(at_sec) + 0.2),
        max(0.0, float(at_sec) - 0.5),
        max(0.0, float(at_sec) + 0.5),
        max(0.0, float(at_sec) - 1.0),
        max(0.0, float(at_sec) + 1.0),
    ]
    for sec in attempts:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            cap.release()
            return frame
        if total_frames > 0:
            frame_idx = max(0, min(total_frames - 1, int(round(sec * fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                cap.release()
                return frame
    cap.release()
    raise RuntimeError(f"failed to read frame near {at_sec:.3f}s: {video_path}")


def fit_with_padding(img: np.ndarray, width: int, height: int, pad_color: Tuple[int, int, int]) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = min(float(width) / max(1, iw), float(height) / max(1, ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), pad_color, dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def draw_text_lines(img: np.ndarray, lines: List[str], x: int, y: int, scale: float = 0.75) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    dy = int(34 * scale / 0.75)
    cy = y
    for line in lines:
        cv2.putText(img, line, (x, cy), font, scale, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, cy), font, scale, (250, 250, 250), 1, cv2.LINE_AA)
        cy += dy


def render_review_panel(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    meta: Dict[str, str],
) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))

    full = frame.copy()
    cv2.rectangle(full, (x1, y1), (x2, y2), (0, 220, 255), 4)
    cv2.putText(full, "Gaze ROI", (x1 + 10, max(36, y1 + 32)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 3, cv2.LINE_AA)
    full_view = fit_with_padding(full, 980, 680, (28, 32, 40))

    roi_crop = frame[y1:y2, x1:x2].copy()
    roi_view = fit_with_padding(roi_crop, 480, 360, (18, 22, 28))
    cv2.rectangle(roi_view, (0, 0), (roi_view.shape[1] - 1, roi_view.shape[0] - 1), (0, 220, 255), 3)

    canvas = np.full((900, 1600, 3), (243, 246, 248), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (1599, 94), (32, 41, 56), -1)
    cv2.putText(
        canvas,
        f"{meta['participant']}  {meta['window_uid']}  {meta['gaze_qc_reason']}",
        (36, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    canvas[130:810, 40:1020] = full_view
    canvas[150:510, 1080:1560] = roi_view

    info = [
        f"segment: {meta['segment_uid']}",
        f"time: {meta['window_start_hhmmss']} -> {meta['window_end_hhmmss']}",
        f"rows: {meta['gaze_rows']} / {meta['expected_gaze_rows']}  coverage={meta['gaze_coverage_ratio']}",
        f"fps: {meta['nominal_gaze_fps']}",
        f"video: {Path(meta['video_path']).name}",
    ]
    draw_text_lines(canvas, info, 1080, 580, scale=0.75)
    return canvas


def make_contact_sheet(image_paths: List[Path], out_path: Path, cols: int, title: str) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        thumb = fit_with_padding(img, 380, 214, (240, 240, 240))
        label = path.stem
        cv2.rectangle(thumb, (0, 180), (379, 213), (22, 28, 36), -1)
        cv2.putText(thumb, label[:38], (8, 204), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    if not thumbs:
        return
    cols = max(1, int(cols))
    rows = int(math.ceil(len(thumbs) / cols))
    canvas = np.full((120 + rows * 230, 40 + cols * 390, 3), (248, 249, 251), dtype=np.uint8)
    cv2.putText(canvas, title, (24, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (28, 34, 42), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"windows: {len(thumbs)}", (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 98, 110), 2, cv2.LINE_AA)
    for idx, thumb in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        y = 120 + r * 230
        x = 20 + c * 390
        canvas[y : y + 214, x : x + 380] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    qc_csv = Path(args.qc_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(qc_csv)
    plan_paths = [Path(p) for p in args.plan_csv] if args.plan_csv else infer_plan_paths(rows)
    roi_by_segment = load_roi_by_segment(plan_paths)

    manifest_rows: List[Dict[str, str]] = []
    by_participant: Dict[str, List[Path]] = {}

    for row in rows:
        participant = str(row.get("participant", "")).strip()
        segment_uid = str(row.get("segment_uid", "")).strip()
        roi = roi_by_segment.get(segment_uid)
        if roi is None:
            continue
        video_path = Path(str(row["video_path"]).strip())
        mid_sec = (float(row["window_start_sec"]) + float(row["window_end_sec"])) * 0.5
        frame = read_frame(video_path, mid_sec)
        panel = render_review_panel(frame, roi, row)
        participant_dir = out_dir / participant
        participant_dir.mkdir(parents=True, exist_ok=True)
        img_path = participant_dir / f"{row['window_uid']}.jpg"
        cv2.imwrite(str(img_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)])
        by_participant.setdefault(participant, []).append(img_path)
        manifest_rows.append(
            {
                "participant": participant,
                "segment_uid": segment_uid,
                "window_uid": str(row["window_uid"]),
                "gaze_qc_reason": str(row.get("gaze_qc_reason", "")),
                "image_path": str(img_path),
                "video_path": str(video_path),
                "window_start_sec": str(row.get("window_start_sec", "")),
                "window_end_sec": str(row.get("window_end_sec", "")),
            }
        )

    for participant, paths in by_participant.items():
        make_contact_sheet(
            sorted(paths),
            out_dir / f"{participant}.contact_sheet.jpg",
            cols=int(args.thumb_cols),
            title=f"{participant} gaze QC manual review",
        )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        if manifest_rows:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
        else:
            csv.writer(f).writerow(["empty"])

    print(f"rows={len(manifest_rows)}")
    print(f"out_dir={out_dir}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
