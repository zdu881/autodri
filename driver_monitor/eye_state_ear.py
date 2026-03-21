#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Face-landmark + EAR demo for driver eye open/closed state.

This script is intentionally demo-oriented:
- Supports optional fixed ROI crop for driver face area.
- Uses MediaPipe Tasks FaceLandmarker.
- Computes left/right/average EAR per frame.
- Applies light smoothing + hysteresis for OPEN/CLOSED labeling.
- Exports annotated video and per-frame CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


Point = Tuple[float, float]
ROI = Tuple[int, int, int, int]

# MediaPipe 478-face landmarks. These 6-point sets follow the common EAR layout:
# p1/p4 = eye corners, p2/p6 + p3/p5 = upper/lower eyelids.
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]


@dataclass
class EyeMetrics:
    face_detected: bool
    left_ear: Optional[float]
    right_ear: Optional[float]
    avg_ear: Optional[float]
    smooth_ear: Optional[float]
    open_ref: Optional[float]
    threshold_close: Optional[float]
    threshold_open: Optional[float]
    state: str
    face_box: Optional[Tuple[int, int, int, int]]
    eye_points: Optional[Dict[str, List[Point]]]


class EyeStateFilter:
    def __init__(
        self,
        fixed_threshold: float = 0.0,
        smooth_alpha: float = 0.35,
        adaptive_window: int = 120,
        min_closed_frames: int = 2,
        reopen_margin: float = 0.02,
    ) -> None:
        self.fixed_threshold = float(fixed_threshold)
        self.smooth_alpha = float(smooth_alpha)
        self.adaptive_window = int(max(5, adaptive_window))
        self.min_closed_frames = int(max(1, min_closed_frames))
        self.reopen_margin = float(max(0.0, reopen_margin))

        self.recent_ears: Deque[float] = deque(maxlen=self.adaptive_window)
        self.smooth_ear: Optional[float] = None
        self.state = "UNKNOWN"
        self.closed_streak = 0

    def update(
        self,
        avg_ear: Optional[float],
        left_ear: Optional[float],
        right_ear: Optional[float],
        face_detected: bool,
    ) -> Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]:
        if (not face_detected) or avg_ear is None or left_ear is None or right_ear is None:
            self.closed_streak = 0
            self.state = "NO_FACE"
            return self.state, self.smooth_ear, None, None, None

        avg_ear = float(avg_ear)
        left_ear = float(left_ear)
        right_ear = float(right_ear)
        self.recent_ears.append(avg_ear)
        if self.smooth_ear is None:
            self.smooth_ear = avg_ear
        else:
            alpha = self.smooth_alpha
            self.smooth_ear = alpha * avg_ear + (1.0 - alpha) * self.smooth_ear

        open_ref = float(np.quantile(np.array(self.recent_ears, dtype=np.float32), 0.85))
        if self.fixed_threshold > 0:
            threshold_close = float(self.fixed_threshold)
        else:
            # Clamp keeps demo behavior stable across clips with different scales.
            threshold_close = min(0.30, max(0.16, open_ref * 0.72))
        threshold_open = threshold_close + self.reopen_margin

        both_low = left_ear < (threshold_close + 0.010) and right_ear < (threshold_close + 0.010)
        strongly_low = avg_ear < (threshold_close - 0.035)
        closed_candidate = (self.smooth_ear < threshold_close) and (both_low or strongly_low)

        if closed_candidate:
            self.closed_streak += 1
        else:
            self.closed_streak = 0

        if self.state in ("UNKNOWN", "OPEN", "NO_FACE"):
            if self.closed_streak >= self.min_closed_frames:
                self.state = "CLOSED"
            elif self.smooth_ear >= threshold_open or max(left_ear, right_ear) >= threshold_open:
                self.state = "OPEN"
        elif self.state == "CLOSED":
            if self.smooth_ear >= threshold_open or max(left_ear, right_ear) >= threshold_open:
                self.state = "OPEN"

        return self.state, self.smooth_ear, open_ref, threshold_close, threshold_open


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Driver eye open/closed demo with FaceLandmarker + EAR")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output annotated video")
    p.add_argument("--csv", default="", help="Optional per-frame CSV output")
    p.add_argument("--model", default="models/face_landmarker.task", help="MediaPipe face landmarker task model")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="Optional fixed ROI crop")
    p.add_argument("--start-sec", type=float, default=0.0, help="Seek start time")
    p.add_argument("--duration-sec", type=float, default=0.0, help="Process duration from start-sec, 0=full")
    p.add_argument("--max-frames", type=int, default=0, help="Optional hard frame limit")
    p.add_argument("--fixed-threshold", type=float, default=0.0, help="EAR closed threshold, 0=adaptive")
    p.add_argument("--smooth-alpha", type=float, default=0.35)
    p.add_argument("--adaptive-window", type=int, default=120)
    p.add_argument("--min-closed-frames", type=int, default=2)
    p.add_argument("--reopen-margin", type=float, default=0.02)
    p.add_argument("--draw-inset", action="store_true", help="Add a zoomed face inset for easier review")
    p.add_argument(
        "--face-priority",
        choices=["largest", "rightmost_largest"],
        default="rightmost_largest",
        help="How to select target face if multiple faces exist in ROI",
    )
    p.add_argument("--min-face-det-conf", type=float, default=0.5)
    p.add_argument("--min-face-pres-conf", type=float, default=0.5)
    p.add_argument("--min-track-conf", type=float, default=0.5)
    return p.parse_args()


def clamp_roi(roi: Optional[Sequence[int]], width: int, height: int) -> ROI:
    if roi is None:
        return 0, 0, width, height
    x1, y1, x2, y2 = [int(v) for v in roi]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_ear(eye_points: Sequence[Point]) -> float:
    if len(eye_points) != 6:
        return 0.0
    return (dist(eye_points[1], eye_points[5]) + dist(eye_points[2], eye_points[4])) / max(
        1e-6, 2.0 * dist(eye_points[0], eye_points[3])
    )


def expand_box(box: Tuple[int, int, int, int], width: int, height: int, scale: float = 1.25) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = max(0, int(round(cx - 0.5 * bw)))
    ny1 = max(0, int(round(cy - 0.5 * bh)))
    nx2 = min(width, int(round(cx + 0.5 * bw)))
    ny2 = min(height, int(round(cy + 0.5 * bh)))
    return nx1, ny1, nx2, ny2


def make_landmarker(args: argparse.Namespace) -> vision.FaceLandmarker:
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"FaceLandmarker model not found: {model_path}")
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=float(args.min_face_det_conf),
        min_face_presence_confidence=float(args.min_face_pres_conf),
        min_tracking_confidence=float(args.min_track_conf),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def pick_face(
    result: vision.FaceLandmarkerResult,
    width: int,
    height: int,
    face_priority: str,
) -> Optional[List[Point]]:
    if not result.face_landmarks:
        return None
    # Driver is typically on the right half in our inward-facing ROI.
    # Keep a right-side bias to avoid selecting the passenger/back-seat face.
    best_pts: Optional[List[Point]] = None
    best_score = -1.0
    for lm_list in result.face_landmarks:
        pts = [(float(p.x) * width, float(p.y) * height) for p in lm_list]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        area = max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        if face_priority == "rightmost_largest":
            cx_norm = 0.5 * (min(xs) + max(xs)) / max(1.0, float(width))
            score = area * (0.75 + 0.50 * cx_norm)
        else:
            score = area
        if score > best_score:
            best_score = score
            best_pts = pts
    return best_pts


def analyze_frame(
    crop_bgr: np.ndarray,
    timestamp_ms: int,
    landmarker: vision.FaceLandmarker,
    state_filter: EyeStateFilter,
    face_priority: str,
) -> EyeMetrics:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

    h, w = crop_bgr.shape[:2]
    pts = pick_face(result, width=w, height=h, face_priority=face_priority)
    if pts is None:
        state, smooth, open_ref, t_close, t_open = state_filter.update(
            None, None, None, face_detected=False
        )
        return EyeMetrics(
            face_detected=False,
            left_ear=None,
            right_ear=None,
            avg_ear=None,
            smooth_ear=smooth,
            open_ref=open_ref,
            threshold_close=t_close,
            threshold_open=t_open,
            state=state,
            face_box=None,
            eye_points=None,
        )

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    face_box = (
        max(0, int(min(xs))),
        max(0, int(min(ys))),
        min(w, int(max(xs))),
        min(h, int(max(ys))),
    )
    left_pts = [pts[i] for i in LEFT_EYE_EAR_IDX]
    right_pts = [pts[i] for i in RIGHT_EYE_EAR_IDX]
    left_ear = compute_ear(left_pts)
    right_ear = compute_ear(right_pts)
    avg_ear = 0.5 * (left_ear + right_ear)
    state, smooth, open_ref, t_close, t_open = state_filter.update(
        avg_ear, left_ear, right_ear, face_detected=True
    )
    return EyeMetrics(
        face_detected=True,
        left_ear=left_ear,
        right_ear=right_ear,
        avg_ear=avg_ear,
        smooth_ear=smooth,
        open_ref=open_ref,
        threshold_close=t_close,
        threshold_open=t_open,
        state=state,
        face_box=face_box,
        eye_points={"left": left_pts, "right": right_pts},
    )


def draw_panel(frame: np.ndarray, lines: Iterable[str], color: Tuple[int, int, int]) -> None:
    lines = list(lines)
    if not lines:
        return
    x0, y0 = 16, 16
    row_h = 28
    box_h = 16 + row_h * len(lines)
    box_w = 520
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate(lines):
        y = y0 + 28 + i * row_h
        cv2.putText(frame, text, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)


def draw_inset(frame: np.ndarray, crop_bgr: np.ndarray, metrics: EyeMetrics) -> None:
    if (not metrics.face_detected) or metrics.face_box is None:
        return
    h, w = crop_bgr.shape[:2]
    x1, y1, x2, y2 = expand_box(metrics.face_box, width=w, height=h, scale=1.3)
    face_crop = crop_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return

    inset_w = min(360, frame.shape[1] // 3)
    scale = inset_w / float(face_crop.shape[1])
    inset_h = max(1, int(round(face_crop.shape[0] * scale)))
    inset = cv2.resize(face_crop, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)

    ox = frame.shape[1] - inset_w - 16
    oy = 16
    y2o = min(frame.shape[0], oy + inset_h)
    roi = frame[oy:y2o, ox : ox + inset_w]
    src = inset[: roi.shape[0], : roi.shape[1]]
    if roi.shape[:2] != src.shape[:2]:
        return
    roi[:] = src
    cv2.rectangle(frame, (ox, oy), (ox + roi.shape[1], oy + roi.shape[0]), (255, 255, 255), 2)


def draw_eye_points(frame: np.ndarray, metrics: EyeMetrics) -> None:
    if not metrics.face_detected or not metrics.eye_points:
        return
    for pts, color in ((metrics.eye_points["left"], (0, 255, 0)), (metrics.eye_points["right"], (0, 255, 255))):
        for x, y in pts:
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
        for a, b in zip(pts, pts[1:] + pts[:1]):
            cv2.line(
                frame,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                color,
                1,
                cv2.LINE_AA,
            )
    if metrics.face_box is not None:
        x1, y1, x2, y2 = metrics.face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2, cv2.LINE_AA)


def format_num(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"


def main() -> None:
    args = parse_args()

    landmarker = make_landmarker(args)
    state_filter = EyeStateFilter(
        fixed_threshold=float(args.fixed_threshold),
        smooth_alpha=float(args.smooth_alpha),
        adaptive_window=int(args.adaptive_window),
        min_closed_frames=int(args.min_closed_frames),
        reopen_margin=float(args.reopen_margin),
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    roi = clamp_roi(args.roi, src_w, src_h)

    start_sec = max(0.0, float(args.start_sec))
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    start_frame = int(round(start_sec * fps))

    out_dir = Path(args.output).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv).resolve() if args.csv else None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    out_w = roi[2] - roi[0]
    out_h = roi[3] - roi[1]
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot create output video: {args.output}")

    csv_rows: List[Dict[str, str]] = []
    frame_idx = 0
    processed = 0
    state_counter: Counter[str] = Counter()
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        src_frame_idx = start_frame + frame_idx
        timestamp_sec = src_frame_idx / max(1e-6, fps)
        if args.duration_sec > 0 and (timestamp_sec - start_sec) > float(args.duration_sec):
            break
        if args.max_frames > 0 and processed >= int(args.max_frames):
            break

        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2].copy()
        metrics = analyze_frame(
            crop_bgr=crop,
            timestamp_ms=int(round(timestamp_sec * 1000.0)),
            landmarker=landmarker,
            state_filter=state_filter,
            face_priority=str(args.face_priority),
        )
        state_counter[metrics.state] += 1

        draw_eye_points(crop, metrics)
        if args.draw_inset:
            draw_inset(crop, crop, metrics)
        state_color = {
            "OPEN": (0, 220, 0),
            "CLOSED": (0, 0, 255),
            "NO_FACE": (0, 215, 255),
            "UNKNOWN": (255, 255, 255),
        }.get(metrics.state, (255, 255, 255))
        draw_panel(
            crop,
            [
                f"Eye State: {metrics.state}",
                f"EAR avg={format_num(metrics.avg_ear)}  smooth={format_num(metrics.smooth_ear)}",
                f"EAR left={format_num(metrics.left_ear)}  right={format_num(metrics.right_ear)}",
                f"thr_close={format_num(metrics.threshold_close)}  thr_open={format_num(metrics.threshold_open)}",
                f"open_ref={format_num(metrics.open_ref)}  t={timestamp_sec:.2f}s",
            ],
            state_color,
        )

        writer.write(crop)
        if csv_path is not None:
            csv_rows.append(
                {
                    "frame_idx": str(src_frame_idx),
                    "timestamp_sec": f"{timestamp_sec:.3f}",
                    "face_detected": "1" if metrics.face_detected else "0",
                    "left_ear": "" if metrics.left_ear is None else f"{metrics.left_ear:.6f}",
                    "right_ear": "" if metrics.right_ear is None else f"{metrics.right_ear:.6f}",
                    "avg_ear": "" if metrics.avg_ear is None else f"{metrics.avg_ear:.6f}",
                    "smooth_ear": "" if metrics.smooth_ear is None else f"{metrics.smooth_ear:.6f}",
                    "open_ref": "" if metrics.open_ref is None else f"{metrics.open_ref:.6f}",
                    "threshold_close": "" if metrics.threshold_close is None else f"{metrics.threshold_close:.6f}",
                    "threshold_open": "" if metrics.threshold_open is None else f"{metrics.threshold_open:.6f}",
                    "state": metrics.state,
                    "roi_x1": str(x1),
                    "roi_y1": str(y1),
                    "roi_x2": str(x2),
                    "roi_y2": str(y2),
                }
            )

        processed += 1
        frame_idx += 1
        if processed % 100 == 0:
            elapsed = time.time() - t0
            print(f"Processed {processed} frames ({processed / max(1e-6, elapsed):.2f} fps)")

    cap.release()
    writer.release()
    if csv_path is not None:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "frame_idx",
                "timestamp_sec",
                "face_detected",
                "left_ear",
                "right_ear",
                "avg_ear",
                "smooth_ear",
                "open_ref",
                "threshold_close",
                "threshold_open",
                "state",
                "roi_x1",
                "roi_y1",
                "roi_x2",
                "roi_y2",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)

    elapsed = time.time() - t0
    print(f"Done. frames={processed} elapsed={elapsed:.2f}s mean_fps={processed / max(1e-6, elapsed):.2f}")
    print(f"State counts: {dict(state_counter)}")
    print(f"Video: {args.output}")
    if csv_path is not None:
        print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()
