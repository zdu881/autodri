#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver gaze state classification using SCRFD (face detect) + YOLOv8-cls (ONNX).

Pipeline:
Video -> ROI crop -> SCRFD face -> expand bbox -> face crop -> YOLOv8-cls -> class

This version adds a robust `Other` class for driver-absent / out-of-seat cases using
presence gating with temporal hysteresis.
"""

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def apply_clahe_bgr(frame_bgr: np.ndarray, clip_limit: float = 2.0,
                    tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


class EMAFilter:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self._has = False
        self._v = 0.0

    def reset(self) -> None:
        self._has = False
        self._v = 0.0

    def update(self, x: float) -> float:
        x = float(x)
        if not self._has:
            self._v = x
            self._has = True
            return x
        self._v = self.alpha * x + (1.0 - self.alpha) * self._v
        return float(self._v)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


@dataclass
class Face:
    xyxy: np.ndarray
    score: float
    kps: Optional[np.ndarray] = None


def _face_center(face: Face) -> Tuple[float, float]:
    return (
        float((face.xyxy[0] + face.xyxy[2]) * 0.5),
        float((face.xyxy[1] + face.xyxy[3]) * 0.5),
    )


def _center_dist(face: Face, center: Tuple[float, float]) -> float:
    cx, cy = _face_center(face)
    px, py = center
    return math.hypot(cx - px, cy - py)


def choose_face(
    faces: List[Face],
    prev_center: Optional[Tuple[float, float]],
    mode: str = "right_to_left_track",
    max_center_dist: float = 160.0,
    top_k: int = 50,
) -> Tuple[Face, Tuple[float, float]]:
    """
    Select one face among detections.

    Modes:
    - score_track: legacy behavior, track nearest among top score faces.
    - right_to_left: pick rightmost face every frame.
    - right_to_left_track: prioritize rightmost face while preserving temporal stability.
    """
    mode = str(mode or "right_to_left_track").strip().lower()

    if mode == "right_to_left":
        best = max(faces, key=lambda f: (_face_center(f)[0], f.score))
    elif mode == "score_track":
        if prev_center is None or len(faces) == 1:
            best = max(faces, key=lambda f: f.score)
        else:
            scored = sorted(faces, key=lambda f: f.score, reverse=True)[:top_k]
            near = [f for f in scored if _center_dist(f, prev_center) <= max_center_dist]
            if near:
                best = min(near, key=lambda f: _center_dist(f, prev_center))
            else:
                best = max(scored, key=lambda f: f.score)
    else:
        # Default: right-to-left priority with tracking (recommended for this setup).
        ranked = sorted(faces, key=lambda f: (_face_center(f)[0], f.score), reverse=True)[:top_k]
        if prev_center is None or len(ranked) == 1:
            best = ranked[0]
        else:
            near = [f for f in ranked if _center_dist(f, prev_center) <= max_center_dist]
            if near:
                # Keep right-side priority even within local neighborhood.
                best = max(near, key=lambda f: (_face_center(f)[0], f.score))
            else:
                best = ranked[0]
    cx, cy = _face_center(best)
    return best, (cx, cy)


def expand_bbox(xyxy: np.ndarray, w: int, h: int, scale: float = 1.2) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = np.clip(cx - bw * 0.5, 0, w - 1)
    ny1 = np.clip(cy - bh * 0.5, 0, h - 1)
    nx2 = np.clip(cx + bw * 0.5, 0, w - 1)
    ny2 = np.clip(cy + bh * 0.5, 0, h - 1)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def safe_crop_xyxy(img: np.ndarray, xyxy: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    xi1 = max(0, min(int(math.floor(x1)), w - 1))
    yi1 = max(0, min(int(math.floor(y1)), h - 1))
    xi2 = max(0, min(int(math.ceil(x2)), w))
    yi2 = max(0, min(int(math.ceil(y2)), h))
    if xi2 <= xi1 + 1 or yi2 <= yi1 + 1:
        return None, (xi1, yi1, xi2, yi2)
    crop = img[yi1:yi2, xi1:xi2]
    if crop.size == 0:
        return None, (xi1, yi1, xi2, yi2)
    return crop, (xi1, yi1, xi2, yi2)


def draw_text_panel(frame, lines, origin=(20, 20), font_scale=0.85,
                    text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 2
    line_h = int(30 * font_scale)
    pad = 8
    max_w = max(cv2.getTextSize(l, font, font_scale, thick)[0][0] for l in lines)
    panel_w = max_w + 2 * pad
    panel_h = line_h * len(lines) + 2 * pad
    x0, y0 = origin
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, line in enumerate(lines):
        ty = y0 + pad + (i + 1) * line_h - 4
        cv2.putText(frame, line, (x0 + pad, ty), font, font_scale, text_color, thick, cv2.LINE_AA)


def face_area_ratio(xyxy: np.ndarray, w: int, h: int) -> float:
    if w <= 0 or h <= 0:
        return 0.0
    bw = max(0.0, float(xyxy[2] - xyxy[0]))
    bh = max(0.0, float(xyxy[3] - xyxy[1]))
    return float((bw * bh) / max(1.0, float(w * h)))


def fmt_float(x: Optional[float], ndigits: int = 4) -> str:
    if x is None:
        return ""
    if not np.isfinite(float(x)):
        return ""
    return f"{float(x):.{ndigits}f}"


class PresenceGate:
    """Temporal hysteresis for driver-present state."""

    def __init__(self, other_enter_frames: int = 8, other_exit_frames: int = 3):
        self.other_enter_frames = max(1, int(other_enter_frames))
        self.other_exit_frames = max(1, int(other_exit_frames))
        self.present_state = False
        self.present_count = 0
        self.absent_count = 0

    def update(self, present_candidate: bool) -> bool:
        if present_candidate:
            self.present_count += 1
            self.absent_count = 0
            if (not self.present_state) and self.present_count >= self.other_exit_frames:
                self.present_state = True
        else:
            self.absent_count += 1
            self.present_count = 0
            if self.present_state and self.absent_count >= self.other_enter_frames:
                self.present_state = False
        return self.present_state


# ---------------------------------------------------------------------------
# SCRFD Face Detector (from gaze_state_onnx.py)
# ---------------------------------------------------------------------------

class SCRFDDetector:
    """Minimal SCRFD (InsightFace-style) ONNX wrapper."""

    def __init__(self, onnx_path: str, input_size: Tuple[int, int] = (640, 640),
                 conf_thresh: float = 0.5, nms_thresh: float = 0.4,
                 pre_nms_topk: int = 500, min_face_size: int = 40,
                 providers: Optional[List[str]] = None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [p for p in providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = int(pre_nms_topk)
        self.min_face_size = int(min_face_size)
        self.strides: List[int] = []
        self.num_anchors: Optional[int] = None

    def _infer_pyramid(self, cls_outputs: List[np.ndarray]) -> None:
        if self.strides and self.num_anchors:
            return
        target_w, target_h = self.input_size
        flat_ns = []
        for o in cls_outputs:
            if o.shape[-1] != 1:
                continue
            flat_n = int(np.prod(o.shape[:-1]))
            if flat_n > 0:
                flat_ns.append(flat_n)
        if not flat_ns:
            raise RuntimeError("SCRFD: no cls outputs to infer pyramid")
        best = None
        for anchors in (1, 2):
            strides = []
            matches = 0
            for n in flat_ns:
                if n % anchors != 0:
                    continue
                hw = n // anchors
                side = int(round(math.sqrt(hw)))
                if side * side != hw:
                    continue
                if target_h % side != 0 or target_w % side != 0:
                    continue
                sh = target_h // side
                sw = target_w // side
                if sh != sw:
                    continue
                stride = int(sh)
                if stride <= 0:
                    continue
                strides.append(stride)
                matches += 1
            strides = sorted(set(strides))
            cand = (matches, anchors, strides)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None or best[0] == 0 or not best[2]:
            raise RuntimeError("SCRFD: failed to infer strides/anchors")
        _, anchors, strides = best
        self.num_anchors = anchors
        self.strides = strides

    def _resize_letterbox(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = img_bgr.shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / w, target_h / h)
        nw = int(round(w * scale))
        nh = int(round(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        dx = (target_w - nw) // 2
        dy = (target_h - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, scale, dx, dy

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        padded, scale, dx, dy = self._resize_letterbox(img_bgr)
        img = padded.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[None, :, :, :]
        return img, scale, dx, dy

    def _generate_anchors(self, feat_h: int, feat_w: int, stride: int) -> np.ndarray:
        shifts_x = (np.arange(0, feat_w) + 0.5) * stride
        shifts_y = (np.arange(0, feat_h) + 0.5) * stride
        shift_y, shift_x = np.meshgrid(shifts_y, shifts_x, indexing="ij")
        centers = np.stack([shift_x, shift_y], axis=-1).reshape(-1, 2)
        if self.num_anchors > 1:
            centers = np.repeat(centers, self.num_anchors, axis=0)
        return centers

    def _decode(self, centers: np.ndarray, deltas: np.ndarray, stride: int) -> np.ndarray:
        deltas = deltas.astype(np.float32) * float(stride)
        x1 = centers[:, 0] - deltas[:, 0]
        y1 = centers[:, 1] - deltas[:, 1]
        x2 = centers[:, 0] + deltas[:, 2]
        y2 = centers[:, 1] + deltas[:, 3]
        return np.stack([x1, y1, x2, y2], axis=1)

    def detect(self, img_bgr: np.ndarray) -> List[Face]:
        blob, scale, dx, dy = self._preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        cls_outputs = [o for o in outputs if o.ndim == 3 and o.shape[-1] == 1]
        bbox_outputs = [o for o in outputs if o.ndim == 3 and o.shape[-1] == 4]
        if len(cls_outputs) == 0 or len(bbox_outputs) == 0:
            cls_outputs = [o for o in outputs if o.ndim in (2, 3) and o.shape[-1] == 1]
            bbox_outputs = [o for o in outputs if o.ndim in (2, 3) and o.shape[-1] == 4]
        if len(cls_outputs) == 0 or len(bbox_outputs) == 0:
            raise RuntimeError("Unable to parse SCRFD outputs")
        self._infer_pyramid(cls_outputs)
        assert self.num_anchors is not None
        target_w, target_h = self.input_size
        all_boxes = []
        all_scores = []
        for stride in self.strides:
            feat_h = target_h // stride
            feat_w = target_w // stride
            num_points = feat_h * feat_w * self.num_anchors
            cls_t = None
            bbox_t = None
            for o in cls_outputs:
                flat_n = int(np.prod(o.shape[:-1]))
                if flat_n == num_points:
                    cls_t = o.reshape(-1)
                    break
            for o in bbox_outputs:
                flat_n = int(np.prod(o.shape[:-1]))
                if flat_n == num_points:
                    bbox_t = o.reshape(-1, 4)
                    break
            if cls_t is None or bbox_t is None:
                continue
            centers = self._generate_anchors(feat_h, feat_w, stride)
            scores = 1 / (1 + np.exp(-cls_t))
            keep = scores >= self.conf_thresh
            if not np.any(keep):
                continue
            boxes = self._decode(centers, bbox_t, stride)
            boxes = boxes[keep]
            scores = scores[keep]
            all_boxes.append(boxes)
            all_scores.append(scores)
        if len(all_boxes) == 0:
            return []
        boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
        scores = np.concatenate(all_scores, axis=0).astype(np.float32)
        k = self.pre_nms_topk
        if k > 0 and scores.shape[0] > k:
            idx = np.argpartition(scores, -k)[-k:]
            boxes = boxes[idx]
            scores = scores[idx]
        keep_idx = nms_xyxy(boxes, scores, self.nms_thresh)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        boxes[:, [0, 2]] -= dx
        boxes[:, [1, 3]] -= dy
        boxes /= (scale + 1e-9)
        h, w = img_bgr.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        if self.min_face_size > 0 and boxes.shape[0] > 0:
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            keep = (bw >= self.min_face_size) & (bh >= self.min_face_size)
            boxes = boxes[keep]
            scores = scores[keep]
        faces = [Face(xyxy=boxes[i], score=float(scores[i])) for i in range(boxes.shape[0])]
        faces.sort(key=lambda f: f.score, reverse=True)
        return faces


# ---------------------------------------------------------------------------
# YOLOv8-cls ONNX Classifier
# ---------------------------------------------------------------------------

class GazeClassifier:
    """YOLOv8-cls ONNX model for face-crop gaze classification."""

    CLASS_NAMES_3 = ("Forward", "In-Car", "Non-Forward")
    CLASS_NAMES_4 = ("Forward", "In-Car", "Non-Forward", "Other")

    def __init__(self, onnx_path: str, input_size: int = 224):
        available = set(ort.get_available_providers())
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = input_size
        self.num_classes = self._infer_num_classes()
        if self.num_classes == 3:
            self.class_names = self.CLASS_NAMES_3
        elif self.num_classes == 4:
            self.class_names = self.CLASS_NAMES_4
        else:
            self.class_names = tuple(f"Class_{i}" for i in range(self.num_classes))
        print(f"[INFO] Classifier loaded: num_classes={self.num_classes}, names={self.class_names}")

    def _infer_num_classes(self) -> int:
        # Run one dry forward to make class count robust for dynamic output shapes.
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        out = self.sess.run(None, {self.input_name: dummy})[0]
        logits = np.asarray(out).reshape(-1)
        n = int(logits.shape[0])
        if n <= 0:
            raise RuntimeError("Failed to infer classifier output classes from ONNX model.")
        return n

    def _preprocess(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        h, w = face_crop_bgr.shape[:2]
        sz = self.input_size
        scale = min(sz / h, sz / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(face_crop_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        top = (sz - nh) // 2
        left = (sz - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        img = canvas[:, :, ::-1].astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        return img

    def infer(self, face_crop_bgr: np.ndarray, class_bias: Optional[np.ndarray] = None) -> Tuple[str, float, np.ndarray]:
        inp = self._preprocess(face_crop_bgr)
        logits = self.sess.run(None, {self.input_name: inp})[0].flatten()
        if class_bias is not None:
            if class_bias.shape[0] != logits.shape[0]:
                raise ValueError(
                    f"class_bias length mismatch: bias={class_bias.shape[0]} logits={logits.shape[0]}"
                )
            logits = logits + class_bias
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        idx = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx]), probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Driver gaze classification using SCRFD + YOLOv8-cls (ONNX)"
    )
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--scrfd", default="models/scrfd_person_2.5g.onnx", help="SCRFD ONNX path")
    p.add_argument("--cls-model", default="models/gaze_cls_yolov8n.onnx",
                   help="YOLOv8-cls ONNX model for gaze classification")
    p.add_argument("--cls-imgsz", type=int, default=224, help="Classifier input size")

    p.add_argument("--roi", nargs=4, type=int, default=[950, 300, 1650, 690],
                   metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--out-video", default="gaze_output_cls.mp4", help="Output video path")
    p.add_argument("--no-video", action="store_true", help="Disable mp4 writing and only save CSV/summary")
    p.add_argument("--csv", default="gaze_output_cls.csv", help="Output CSV path")
    p.add_argument("--start-sec", type=float, default=0.0,
                   help="Start time in seconds for segment inference (default: 0, from video start)")
    p.add_argument("--duration-sec", type=float, default=0.0,
                   help="Segment duration in seconds (default: 0, process to video end)")

    p.add_argument("--scrfd-input", nargs=2, type=int, default=[640, 640], metavar=("W", "H"))
    p.add_argument("--face-conf", type=float, default=0.55, help="SCRFD confidence threshold")
    p.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold")
    p.add_argument("--pre-nms-topk", type=int, default=800)
    p.add_argument("--min-face-size", type=int, default=40)

    p.add_argument("--face-expand", type=float, default=1.25,
                   help="Expand face bbox by this factor before classifier crop")
    p.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing")

    p.add_argument("--smooth-center-alpha", type=float, default=0.3,
                   help="EMA smoothing alpha for face center (0 disables)")
    p.add_argument(
        "--face-priority",
        choices=["score_track", "right_to_left", "right_to_left_track"],
        default="right_to_left_track",
        help=(
            "Face selection priority when multiple faces appear in ROI. "
            "'right_to_left_track' is recommended for driver-on-right camera views."
        ),
    )
    p.add_argument("--track-max-dist", type=float, default=160.0)
    p.add_argument("--track-topk", type=int, default=50)
    p.add_argument("--class-debounce", type=int, default=3,
                   help="Require K consecutive frames before switching gaze class")
    p.add_argument("--cls-threshold", type=float, default=0.0,
                   help="Min confidence to accept 3-way class; else keep stable class")
    p.add_argument(
        "--class-bias",
        nargs="*",
        type=float,
        default=[],
        help=(
            "Additive logit bias. For 3-class model pass 3 values (F/IC/NF). "
            "For 4-class model you can pass 4 values (F/IC/NF/Other), "
            "or pass 3 values and Other bias defaults to 0."
        ),
    )

    p.add_argument("--presence-min-face-score", type=float, default=0.45,
                   help="Face score threshold for presence candidate")
    p.add_argument("--presence-min-face-ratio", type=float, default=0.0,
                   help="Min face area ratio in ROI for presence candidate")
    p.add_argument("--other-enter-frames", type=int, default=8,
                   help="Absent-candidate frames to switch to Other")
    p.add_argument("--other-exit-frames", type=int, default=3,
                   help="Present-candidate frames to switch from Other to present")

    p.add_argument("--write-when-other", choices=["skip", "other", "noface"], default="other",
                   help="Write Other rows or skip; 'noface' kept as legacy alias")
    p.add_argument("--write-when-noface", dest="write_when_other",
                   choices=["skip", "other", "noface"], help=argparse.SUPPRESS)

    p.add_argument("--max-frames", type=int, default=0, help="Debug: stop after N frames (0=all)")
    p.add_argument("--no-overlay", action="store_true", help="Do not draw annotations")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_when_other == "noface":
        args.write_when_other = "other"

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_sec = max(0.0, float(args.start_sec))
    duration_sec = max(0.0, float(args.duration_sec))
    start_frame = int(round(start_sec * fps))
    if total_frames > 0 and start_frame >= total_frames:
        raise ValueError(
            f"--start-sec ({start_sec:.3f}s) exceeds video duration "
            f"({total_frames / max(1e-6, fps):.3f}s)"
        )
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    target_frames_by_duration = 0
    if duration_sec > 0:
        target_frames_by_duration = max(1, int(math.ceil(duration_sec * fps)))

    max_proc_frames = 0
    if total_frames > 0:
        remain_from_start = max(0, total_frames - start_frame)
        max_proc_frames = remain_from_start
        if target_frames_by_duration > 0:
            max_proc_frames = min(max_proc_frames, target_frames_by_duration)
    elif target_frames_by_duration > 0:
        max_proc_frames = target_frames_by_duration
    if args.max_frames and int(args.max_frames) > 0:
        if max_proc_frames > 0:
            max_proc_frames = min(max_proc_frames, int(args.max_frames))
        else:
            max_proc_frames = int(args.max_frames)

    x1, y1, x2, y2 = map(int, args.roi)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI")

    scrfd = SCRFDDetector(
        onnx_path=args.scrfd,
        input_size=(int(args.scrfd_input[0]), int(args.scrfd_input[1])),
        conf_thresh=float(args.face_conf),
        nms_thresh=float(args.nms),
        pre_nms_topk=int(args.pre_nms_topk),
        min_face_size=int(args.min_face_size),
    )
    classifier = GazeClassifier(onnx_path=args.cls_model, input_size=args.cls_imgsz)

    raw_bias = np.asarray(args.class_bias, dtype=np.float32).reshape(-1)
    if raw_bias.size == 0:
        class_bias = np.zeros((classifier.num_classes,), dtype=np.float32)
    elif raw_bias.size == classifier.num_classes:
        class_bias = raw_bias
    elif raw_bias.size == 3 and classifier.num_classes == 4:
        class_bias = np.concatenate([raw_bias, np.zeros((1,), dtype=np.float32)], axis=0)
        print("[WARN] 4-class model with 3-element class-bias: using Other bias = 0.")
    elif raw_bias.size == 4 and classifier.num_classes == 3:
        class_bias = raw_bias[:3]
        print("[WARN] 3-class model with 4-element class-bias: dropping 4th element.")
    else:
        raise ValueError(
            f"Invalid --class-bias length {raw_bias.size} for model with {classifier.num_classes} classes."
        )
    print(f"[INFO] Face priority mode: {args.face_priority}")

    writer = None
    if not args.no_video:
        os.makedirs(os.path.dirname(args.out_video) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {args.out_video}")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    csv_f = open(args.csv, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "Timestamp", "FrameID", "Face_Score", "Face_Area_Ratio",
        "Presence_Candidate", "Presence_State",
        "Cls_Forward", "Cls_InCar", "Cls_NonForward", "Cls_Other",
        "Raw_Class", "Confidence", "Base_Class", "Gaze_Class",
        "Video_Timestamp", "Video_FrameID",
    ])

    frame_id = 0
    start_t = time.time()

    class_counter: Counter = Counter()
    prev_face_center: Optional[Tuple[float, float]] = None
    cx_ema = EMAFilter(alpha=float(args.smooth_center_alpha))
    cy_ema = EMAFilter(alpha=float(args.smooth_center_alpha))

    stable_gaze_class: Optional[str] = None
    pending_gaze_class: Optional[str] = None
    pending_count = 0
    debounce_k = max(1, int(args.class_debounce))
    presence_gate = PresenceGate(
        other_enter_frames=int(args.other_enter_frames),
        other_exit_frames=int(args.other_exit_frames),
    )

    while True:
        if max_proc_frames and frame_id >= max_proc_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            break

        segment_frame_id = frame_id
        video_frame_id = start_frame + frame_id
        proc = apply_clahe_bgr(roi) if args.clahe else roi
        ts = segment_frame_id / fps
        video_ts = video_frame_id / fps

        face: Optional[Face] = None
        face_crop: Optional[np.ndarray] = None
        face_score: Optional[float] = None
        face_ratio: Optional[float] = None
        fx1 = fy1 = fx2 = fy2 = 0.0

        raw_class = ""
        confidence: Optional[float] = None
        probs = np.full((classifier.num_classes,), np.nan, dtype=np.float32)
        presence_reason = ""
        has_valid_face = False

        faces = scrfd.detect(proc)
        if len(faces) == 0:
            prev_face_center = None
            presence_reason = "No face"
        else:
            face, prev_face_center = choose_face(
                faces,
                prev_face_center,
                mode=str(args.face_priority),
                max_center_dist=float(args.track_max_dist),
                top_k=int(args.track_topk),
            )
            fx1, fy1, fx2, fy2 = [float(v) for v in face.xyxy]
            face_score = float(face.score)
            rw = proc.shape[1]
            rh = proc.shape[0]
            face_ratio = face_area_ratio(face.xyxy, rw, rh)
            expanded = expand_bbox(face.xyxy, rw, rh, scale=float(args.face_expand))
            face_crop, _ = safe_crop_xyxy(proc, expanded)
            if face_crop is not None:
                has_valid_face = True
            else:
                presence_reason = "Face crop empty"

        presence_candidate = (
            has_valid_face
            and face_score is not None
            and face_ratio is not None
            and face_score >= float(args.presence_min_face_score)
            and face_ratio >= float(args.presence_min_face_ratio)
        )
        presence_state = presence_gate.update(bool(presence_candidate))

        if not presence_candidate and has_valid_face:
            if face_score is not None and face_score < float(args.presence_min_face_score):
                presence_reason = f"Low face score ({face_score:.2f})"
            elif face_ratio is not None and face_ratio < float(args.presence_min_face_ratio):
                presence_reason = f"Face too small ({face_ratio * 100:.2f}%)"

        if not presence_state:
            final_class = "Other"
            class_counter[final_class] += 1
            pending_gaze_class = None
            pending_count = 0

            if args.write_when_other != "skip":
                csv_w.writerow([
                    f"{ts:.3f}",
                    segment_frame_id,
                    fmt_float(face_score),
                    fmt_float(face_ratio, ndigits=6),
                    int(bool(presence_candidate)),
                    0,
                    "", "", "", "",
                    "", "",
                    "",
                    final_class,
                    f"{video_ts:.3f}",
                    video_frame_id,
                ])

            if not args.no_overlay:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 6)
                if face is not None:
                    cv2.rectangle(
                        frame,
                        (x1 + int(fx1), y1 + int(fy1)),
                        (x1 + int(fx2), y1 + int(fy2)),
                        (0, 255, 0),
                        4,
                    )
                draw_text_panel(
                    frame,
                    [
                        f"Frame={segment_frame_id}  t={ts:.2f}s",
                        f"VideoFrame={video_frame_id}  video_t={video_ts:.2f}s",
                        "Presence: OUT (Other)",
                        presence_reason or "Absent candidate",
                    ],
                    origin=(20, 20), font_scale=0.85,
                    text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7,
                )
                cv2.putText(
                    frame,
                    final_class,
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (255, 255, 255),
                    5,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(frame)
            if not has_valid_face:
                cx_ema.reset()
                cy_ema.reset()

            frame_id += 1
            continue

        if has_valid_face and face_crop is not None:
            raw_class, confidence, probs = classifier.infer(face_crop, class_bias=class_bias)
            if args.cls_threshold > 0 and confidence < args.cls_threshold:
                cand_class = stable_gaze_class if stable_gaze_class is not None else raw_class
            else:
                cand_class = raw_class
        else:
            cand_class = stable_gaze_class if stable_gaze_class is not None else "Forward"

        if stable_gaze_class is None:
            stable_gaze_class = cand_class
            pending_gaze_class = None
            pending_count = 0
        else:
            if cand_class == stable_gaze_class:
                pending_gaze_class = None
                pending_count = 0
            else:
                if pending_gaze_class != cand_class:
                    pending_gaze_class = cand_class
                    pending_count = 1
                else:
                    pending_count += 1
                if pending_count >= debounce_k:
                    stable_gaze_class = cand_class
                    pending_gaze_class = None
                    pending_count = 0

        base_class = stable_gaze_class if stable_gaze_class is not None else cand_class
        final_class = base_class
        class_counter[final_class] += 1

        csv_w.writerow([
            f"{ts:.3f}",
            segment_frame_id,
            fmt_float(face_score),
            fmt_float(face_ratio, ndigits=6),
            int(bool(presence_candidate)),
            1,
            fmt_float(float(probs[0])) if probs.shape[0] > 0 and np.isfinite(float(probs[0])) else "",
            fmt_float(float(probs[1])) if probs.shape[0] > 1 and np.isfinite(float(probs[1])) else "",
            fmt_float(float(probs[2])) if probs.shape[0] > 2 and np.isfinite(float(probs[2])) else "",
            fmt_float(float(probs[3])) if probs.shape[0] > 3 and np.isfinite(float(probs[3])) else "",
            raw_class,
            fmt_float(confidence),
            base_class,
            final_class,
            f"{video_ts:.3f}",
            video_frame_id,
        ])

        if not args.no_overlay:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 6)
            if face is not None:
                cv2.rectangle(
                    frame,
                    (x1 + int(fx1), y1 + int(fy1)),
                    (x1 + int(fx2), y1 + int(fy2)),
                    (0, 255, 0),
                    4,
                )
                cx_raw = x1 + float((fx1 + fx2) * 0.5)
                cy_raw = y1 + float((fy1 + fy2) * 0.5)
                if float(args.smooth_center_alpha) > 0.0:
                    _ = int(round(cx_ema.update(cx_raw)))
                    _ = int(round(cy_ema.update(cy_raw)))
            else:
                cx_ema.reset()
                cy_ema.reset()

            class_color = {
                "Forward": (0, 255, 0),
                "In-Car": (0, 165, 255),
                "Non-Forward": (0, 0, 255),
                "Other": (255, 255, 255),
            }.get(final_class, (255, 255, 255))

            score_line = "F=NA  IC=NA  NF=NA"
            if probs.shape[0] >= 3 and all(np.isfinite(float(v)) for v in probs[:3]):
                score_line = f"F={probs[0]:.2f}  IC={probs[1]:.2f}  NF={probs[2]:.2f}"
                if probs.shape[0] >= 4 and np.isfinite(float(probs[3])):
                    score_line += f"  O={probs[3]:.2f}"
            raw_line = f"Raw: {raw_class or 'NA'} ({confidence:.2f})" if confidence is not None else "Raw: NA"

            draw_text_panel(
                frame,
                [
                    f"Frame={segment_frame_id}  t={ts:.2f}s",
                    f"VideoFrame={video_frame_id}  video_t={video_ts:.2f}s",
                    "Presence: IN",
                    score_line,
                    raw_line,
                    f"Class: {final_class}",
                ],
                origin=(20, 20), font_scale=0.85,
                text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7,
            )
            cv2.putText(
                frame,
                final_class,
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                class_color,
                5,
                cv2.LINE_AA,
            )

        if writer is not None:
            writer.write(frame)
        frame_id += 1

        if frame_id % 300 == 0:
            now = time.time()
            dt = max(1e-6, now - start_t)
            proc_fps = frame_id / dt
            if max_proc_frames > 0:
                remain = max(0, max_proc_frames - frame_id)
                eta_s = remain / max(1e-6, proc_fps)
                print(
                    f"[{frame_id}/{max_proc_frames}] proc_fps={proc_fps:.2f} "
                    f"elapsed={dt/60:.1f}m ETA={eta_s/60:.1f}m"
                )
            elif total_frames > 0:
                done_video_frame = start_frame + frame_id
                remain = max(0, total_frames - done_video_frame)
                eta_s = remain / max(1e-6, proc_fps)
                print(
                    f"[seg={frame_id}, video={done_video_frame}/{total_frames}] "
                    f"proc_fps={proc_fps:.2f} elapsed={dt/60:.1f}m ETA={eta_s/60:.1f}m"
                )
            else:
                print(f"[{frame_id}] proc_fps={proc_fps:.2f} elapsed={dt/60:.1f}m")

    csv_f.close()
    cap.release()
    if writer is not None:
        writer.release()

    total = sum(class_counter.values())
    summary = {
        "video": args.video,
        "out_video": "" if args.no_video else args.out_video,
        "csv": args.csv,
        "model": args.cls_model,
        "start_sec": float(start_sec),
        "duration_sec": float(duration_sec),
        "start_frame": int(start_frame),
        "max_proc_frames": int(max_proc_frames),
        "total_frames_written": int(frame_id),
        "end_frame_exclusive": int(start_frame + frame_id),
        "class_counts": dict(class_counter),
        "class_percent": {k: (v / total * 100.0 if total else 0.0) for k, v in class_counter.items()},
    }
    try:
        with open(args.csv + ".summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print("\n=== Summary ===")
    print("Counts:", dict(class_counter))
    if total:
        pct = {k: round(v / total * 100.0, 2) for k, v in class_counter.items()}
        print("Percent:", pct)
    if writer is not None:
        print(f"Video: {args.out_video}")
    else:
        print("Video: <disabled by --no-video>")
    print(f"CSV:   {args.csv}")
    print(f"JSON:  {args.csv}.summary.json")
    print("Done")


if __name__ == "__main__":
    main()
