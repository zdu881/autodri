#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from ort_runtime import ensure_onnxruntime_cuda_runtime

ensure_onnxruntime_cuda_runtime()

import onnxruntime as ort


def apply_clahe_bgr(frame_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to L channel in LAB, then convert back to BGR."""
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
        a = self.alpha
        self._v = a * x + (1.0 - a) * self._v
        return float(self._v)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Pure numpy NMS. boxes: (N,4) xyxy, scores: (N,)"""
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


@dataclass
class Face:
    xyxy: np.ndarray  # (4,) float32 in original image coordinates
    score: float


def choose_face_by_tracking(
    faces: List[Face],
    prev_center: Optional[Tuple[float, float]],
    max_center_dist: float = 160.0,
    top_k: int = 50,
) -> Tuple[Face, Tuple[float, float]]:
    """Pick a stable face across frames.

    If prev_center is provided, pick the face whose center is closest to it among top_k.
    Otherwise pick the highest-score face.
    """
    if not faces:
        raise ValueError("faces is empty")

    def center_of(f: Face) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(v) for v in f.xyxy]
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    if prev_center is None:
        c = center_of(faces[0])
        return faces[0], c

    best = faces[0]
    best_c = center_of(best)
    best_d2 = (best_c[0] - prev_center[0]) ** 2 + (best_c[1] - prev_center[1]) ** 2

    for f in faces[: max(1, int(top_k))]:
        c = center_of(f)
        d2 = (c[0] - prev_center[0]) ** 2 + (c[1] - prev_center[1]) ** 2
        if d2 < best_d2:
            best = f
            best_c = c
            best_d2 = d2

    if math.sqrt(best_d2) > float(max_center_dist):
        # Too far: likely a different false positive; fall back to best score.
        c = center_of(faces[0])
        return faces[0], c
    return best, best_c


class SCRFDDetector:
    """Minimal SCRFD (InsightFace-style) ONNX wrapper.

    This implementation targets common SCRFD ONNX exports (e.g. scrfd_2.5g_bnkps.onnx).
    Different exports can have different output orders/names; we try to infer outputs
    robustly by shapes.
    """

    def __init__(
        self,
        onnx_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        pre_nms_topk: int = 500,
        min_face_size: int = 40,
        providers: Optional[List[str]] = None,
    ):
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

        # Will be inferred dynamically from model outputs on first call to detect().
        self.strides: List[int] = []
        self.num_anchors: Optional[int] = None

    def _infer_pyramid(self, cls_outputs: List[np.ndarray]) -> None:
        """Infer (num_anchors, strides) from SCRFD output sizes.

        Many SCRFD exports differ:
        - 3 levels (8/16/32) with 2 anchors
        - 5 levels (8/16/32/64/128) with 1 anchor
        This function chooses the best match against the configured input size.
        """
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

        best = None  # (matches, num_anchors, strides)
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
            raise RuntimeError("SCRFD: failed to infer strides/anchors from outputs")

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
        # SCRFD commonly uses BGR with mean 127.5 and scale 1/128; but many exports accept 0-255.
        # We'll use the common normalization:
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[None, :, :, :]
        return img, scale, dx, dy

    def _generate_anchors(self, feat_h: int, feat_w: int, stride: int) -> np.ndarray:
        """Generate anchor centers (x,y) for each location and each anchor."""
        # centers at (j + 0.5) * stride
        shifts_x = (np.arange(0, feat_w) + 0.5) * stride
        shifts_y = (np.arange(0, feat_h) + 0.5) * stride
        shift_y, shift_x = np.meshgrid(shifts_y, shifts_x, indexing="ij")
        centers = np.stack([shift_x, shift_y], axis=-1).reshape(-1, 2)
        # duplicate for num_anchors
        if self.num_anchors > 1:
            centers = np.repeat(centers, self.num_anchors, axis=0)
        return centers

    def _decode(self, centers: np.ndarray, deltas: np.ndarray, stride: int) -> np.ndarray:
        """Decode bbox deltas to xyxy in resized/letterboxed space.

        deltas: (N,4) in ltrb distances.
        """
        # Many SCRFD ONNX exports output ltrb distances in *stride units*.
        # If we don't scale by stride, boxes collapse to a few pixels.
        deltas = deltas.astype(np.float32) * float(stride)
        # left, top, right, bottom distances
        x1 = centers[:, 0] - deltas[:, 0]
        y1 = centers[:, 1] - deltas[:, 1]
        x2 = centers[:, 0] + deltas[:, 2]
        y2 = centers[:, 1] + deltas[:, 3]
        return np.stack([x1, y1, x2, y2], axis=1)

    def detect(self, img_bgr: np.ndarray) -> List[Face]:
        blob, scale, dx, dy = self._preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: blob})

        # Heuristic parse: SCRFD usually outputs for each pyramid level:
        # - cls (..,1)
        # - bbox (..,4)
        # - (optional) keypoints (..,10)
        # We group by last-dimension size.
        cls_outputs = [o for o in outputs if o.ndim == 3 and o.shape[-1] == 1]
        bbox_outputs = [o for o in outputs if o.ndim == 3 and o.shape[-1] == 4]

        # Some exports flatten to (N, C) or (1, N, C); fall back to a looser match.
        if len(cls_outputs) == 0 or len(bbox_outputs) == 0:
            cls_outputs = [o for o in outputs if o.ndim in (2, 3) and o.shape[-1] == 1]
            bbox_outputs = [o for o in outputs if o.ndim in (2, 3) and o.shape[-1] == 4]

        if len(cls_outputs) == 0 or len(bbox_outputs) == 0:
            raise RuntimeError(
                "Unable to parse SCRFD outputs. Please check the ONNX export format and adjust the parser."
            )

        self._infer_pyramid(cls_outputs)
        assert self.num_anchors is not None

        target_w, target_h = self.input_size
        all_boxes = []
        all_scores = []

        for stride in self.strides:
            feat_h = target_h // stride
            feat_w = target_w // stride
            num_points = feat_h * feat_w * self.num_anchors

            # Find matching tensors by length
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
                # Skip if we can't match this stride
                continue

            centers = self._generate_anchors(feat_h, feat_w, stride)
            scores = 1 / (1 + np.exp(-cls_t))  # sigmoid
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

        # Speed guard: keep only top-k scores before NMS.
        # SCRFD can emit thousands of candidates; full NMS on them is too slow for long videos.
        k = self.pre_nms_topk
        if k > 0 and scores.shape[0] > k:
            idx = np.argpartition(scores, -k)[-k:]
            boxes = boxes[idx]
            scores = scores[idx]

        keep_idx = nms_xyxy(boxes, scores, self.nms_thresh)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

        # Map from letterboxed space back to original image
        # boxes are in padded (target) coordinates; remove padding then unscale.
        boxes[:, [0, 2]] -= dx
        boxes[:, [1, 3]] -= dy
        boxes /= (scale + 1e-9)

        # Clip
        h, w = img_bgr.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

        # Filter tiny boxes (likely false positives)
        if self.min_face_size > 0 and boxes.shape[0] > 0:
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            keep = (bw >= self.min_face_size) & (bh >= self.min_face_size)
            boxes = boxes[keep]
            scores = scores[keep]

        faces = [Face(xyxy=boxes[i], score=float(scores[i])) for i in range(boxes.shape[0])]
        faces.sort(key=lambda f: f.score, reverse=True)
        return faces


class L2CSGazeEstimator:
    """L2CS gaze estimator ONNX wrapper.

    Supports common output formats:
    - Direct regression: (1,2) => [pitch_deg, yaw_deg]
    - Two outputs: pitch and yaw scalars
    - Logits classification over bins: (1, n_bins) each => expected angle
    """

    def __init__(
        self,
        onnx_path: str,
        input_size: Tuple[int, int] = (448, 448),
        providers: Optional[List[str]] = None,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        available = set(ort.get_available_providers())
        providers = [p for p in providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

        # For logits-based L2CS (commonly 90 bins from -90..90)
        self.default_bins = np.arange(-90, 90, dtype=np.float32)

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        w, h = self.input_size
        resized = cv2.resize(face_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) / 255.0
        # L2CS commonly expects RGB normalization; but exports vary.
        # We'll convert to RGB and apply ImageNet mean/std which is the common training setup.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[None, :, :, :]
        return img

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)

    def infer(self, face_bgr: np.ndarray) -> Tuple[float, float]:
        blob = self._preprocess(face_bgr)
        outs = self.session.run(None, {self.input_name: blob})

        # Normalize outputs to numpy arrays
        outs = [np.asarray(o) for o in outs]

        # Case A: single output (1,2)
        if len(outs) == 1 and outs[0].ndim == 2 and outs[0].shape[-1] == 2:
            pitch, yaw = float(outs[0][0, 0]), float(outs[0][0, 1])
            return pitch, yaw

        # Case B: two scalar outputs (1,1) or (1,)
        if len(outs) >= 2:
            o0, o1 = outs[0], outs[1]
            if o0.size == 1 and o1.size == 1:
                pitch, yaw = float(o0.reshape(-1)[0]), float(o1.reshape(-1)[0])
                return pitch, yaw

        # Case C: logits over bins (most common L2CS)
        # We pick two arrays with shape (1, n_bins)
        candidates = [o for o in outs if o.ndim == 2 and o.shape[0] == 1 and o.shape[1] >= 10]
        if len(candidates) >= 2:
            pitch_logits = candidates[0]
            yaw_logits = candidates[1]
            n_bins = pitch_logits.shape[1]
            if n_bins != self.default_bins.shape[0]:
                bins = np.linspace(-90, 90, n_bins, endpoint=False, dtype=np.float32)
            else:
                bins = self.default_bins

            pitch_prob = self._softmax(pitch_logits, axis=1)
            yaw_prob = self._softmax(yaw_logits, axis=1)

            pitch = float(np.sum(pitch_prob[0] * bins))
            yaw = float(np.sum(yaw_prob[0] * bins))
            return pitch, yaw

        raise RuntimeError("Unsupported L2CS ONNX output format. Please inspect model outputs and adjust parsing.")


class GlobalCalibrator:
    """Global 2D histogram accumulator for (pitch, yaw) mode."""

    def __init__(
        self,
        pitch_range: Tuple[float, float] = (-90.0, 90.0),
        yaw_range: Tuple[float, float] = (-90.0, 90.0),
        bin_size_deg: float = 2.0,
    ):
        self.pitch_min, self.pitch_max = pitch_range
        self.yaw_min, self.yaw_max = yaw_range
        self.bin_size = float(bin_size_deg)

        self.pitch_bins = int(math.ceil((self.pitch_max - self.pitch_min) / self.bin_size))
        self.yaw_bins = int(math.ceil((self.yaw_max - self.yaw_min) / self.bin_size))
        self.hist = np.zeros((self.pitch_bins, self.yaw_bins), dtype=np.int64)

    def _to_bin(self, pitch: float, yaw: float) -> Tuple[int, int]:
        p = np.clip(pitch, self.pitch_min, self.pitch_max - 1e-6)
        y = np.clip(yaw, self.yaw_min, self.yaw_max - 1e-6)
        pi = int((p - self.pitch_min) / self.bin_size)
        yi = int((y - self.yaw_min) / self.bin_size)
        pi = int(np.clip(pi, 0, self.pitch_bins - 1))
        yi = int(np.clip(yi, 0, self.yaw_bins - 1))
        return pi, yi

    def update(self, pitch: float, yaw: float) -> None:
        pi, yi = self._to_bin(pitch, yaw)
        self.hist[pi, yi] += 1

    def get_reference(self) -> Tuple[float, float]:
        return self.get_reference_prefer_pitch(prefer_pitch="mode", topk=20)

    def get_reference_prefer_pitch(self, prefer_pitch: str = "mode", topk: int = 20) -> Tuple[float, float]:
        """Get reference from the global histogram.

        prefer_pitch:
          - "mode": strict argmax bin
          - "high": among top-k most frequent bins, choose the one with highest pitch
          - "low":  among top-k most frequent bins, choose the one with lowest pitch
        """
        prefer_pitch = str(prefer_pitch)
        topk = max(1, int(topk))

        flat = self.hist.reshape(-1)
        if flat.size == 0:
            return 0.0, 0.0

        if prefer_pitch == "mode":
            idx = int(np.argmax(flat))
            pi, yi = np.unravel_index(idx, self.hist.shape)
            ref_pitch = self.pitch_min + (int(pi) + 0.5) * self.bin_size
            ref_yaw = self.yaw_min + (int(yi) + 0.5) * self.bin_size
            return float(ref_pitch), float(ref_yaw)

        # Top-k bins by count
        k = min(topk, flat.size)
        idxs = np.argpartition(flat, -k)[-k:]
        best = None
        best_pitch = None
        best_count = None
        for idx in idxs:
            count = int(flat[idx])
            if count <= 0:
                continue
            pi, yi = np.unravel_index(int(idx), self.hist.shape)
            pitch_c = self.pitch_min + (int(pi) + 0.5) * self.bin_size
            if best is None:
                best = (int(pi), int(yi))
                best_pitch = float(pitch_c)
                best_count = count
                continue

            if count > best_count:
                best = (int(pi), int(yi))
                best_pitch = float(pitch_c)
                best_count = count
            elif count == best_count:
                if prefer_pitch == "high" and float(pitch_c) > float(best_pitch):
                    best = (int(pi), int(yi))
                    best_pitch = float(pitch_c)
                elif prefer_pitch == "low" and float(pitch_c) < float(best_pitch):
                    best = (int(pi), int(yi))
                    best_pitch = float(pitch_c)

        if best is None:
            return 0.0, 0.0

        pi, yi = best
        ref_pitch = self.pitch_min + (pi + 0.5) * self.bin_size
        ref_yaw = self.yaw_min + (yi + 0.5) * self.bin_size
        return float(ref_pitch), float(ref_yaw)


class RefCalibrator:
    """Estimate 'forward reference' robustly even if forward is not the mode."""

    def __init__(
        self,
        strategy: str = "top_pitch",
        bin_size_deg: float = 2.0,
        topk_pitch: int = 60,
        pitch_range: Tuple[float, float] = (-90.0, 90.0),
        yaw_range: Tuple[float, float] = (-90.0, 90.0),
    ):
        self.strategy = str(strategy)
        self.topk_pitch = int(topk_pitch)
        self.pitch_min, self.pitch_max = pitch_range
        self.yaw_min, self.yaw_max = yaw_range
        self._p: List[float] = []
        self._y: List[float] = []
        self._hist = GlobalCalibrator(
            pitch_range=pitch_range,
            yaw_range=yaw_range,
            bin_size_deg=float(bin_size_deg),
        )

    def reset(self) -> None:
        self._p.clear()
        self._y.clear()
        self._hist = GlobalCalibrator(
            pitch_range=(self.pitch_min, self.pitch_max),
            yaw_range=(self.yaw_min, self.yaw_max),
            bin_size_deg=float(self._hist.bin_size),
        )

    def update(self, pitch: float, yaw: float) -> None:
        p = float(np.clip(float(pitch), self.pitch_min, self.pitch_max))
        y = float(np.clip(float(yaw), self.yaw_min, self.yaw_max))
        self._p.append(p)
        self._y.append(y)
        self._hist.update(p, y)

    def get_reference(self, prefer_pitch: str = "mode", topk_bins: int = 20) -> Tuple[float, float]:
        if not self._p:
            return 0.0, 0.0

        if self.strategy == "hist_mode":
            return self._hist.get_reference_prefer_pitch(prefer_pitch=prefer_pitch, topk=topk_bins)

        # top_pitch: take top-K pitch samples (most up/forward) and compute ref from them.
        p = np.asarray(self._p, dtype=np.float32)
        y = np.asarray(self._y, dtype=np.float32)
        k = max(1, min(int(self.topk_pitch), int(p.size)))
        idx = np.argpartition(p, -k)[-k:]
        p_sel = p[idx]
        y_sel = y[idx]
        ref_pitch = float(np.median(p_sel))
        ref_yaw = float(np.median(y_sel))
        return ref_pitch, ref_yaw


def classify_gaze(delta_pitch: float, delta_yaw: float) -> str:
    # Backward compatible default rule.
    if delta_pitch < -15.0:
        return "In-Car"
    if abs(delta_yaw) > 20.0 or delta_pitch > 15.0:
        return "Non-Forward"
    return "Forward"


def classify_gaze_v2(
    delta_pitch: float,
    delta_yaw: float,
    *,
    incar_pitch_neg: float,
    incar_pitch_pos: float,
    incar_pitch_pos_max: float,
    incar_yaw_max: float,
    nonforward_yaw_enter: float,
    nonforward_pitch_up_enter: float,
) -> str:
    """3-class rule aligned with user definition.

    - Forward: looking ahead
    - Non-Forward: left/right/back/up
    - In-Car: looking down inside the cabin

    We define in-car as "down enough" AND "yaw close to forward".
    """
    dy = abs(float(delta_yaw))
    dp = float(delta_pitch)

    # Obvious left/right => Non-Forward
    if dy >= float(nonforward_yaw_enter):
        return "Non-Forward"

    # Up => Non-Forward (keep this separate from In-Car band)
    # Note: depending on model/export, pitch sign can vary; treat "up" as sufficiently large positive.
    if dp >= float(nonforward_pitch_up_enter):
        return "Non-Forward"

    # In-Car: looking down inside cabin.
    # Practical rule: require yaw close to forward, and pitch indicates down either:
    # - strongly negative (dp <= incar_pitch_neg), OR
    # - mild positive band (incar_pitch_pos <= dp <= incar_pitch_pos_max)
    if dy <= float(incar_yaw_max):
        if dp <= float(incar_pitch_neg):
            return "In-Car"
        if float(incar_pitch_pos) <= dp <= float(incar_pitch_pos_max):
            return "In-Car"

    return "Forward"


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
    """Safely crop an image with float xyxy, returning (crop or None, (x1,y1,x2,y2)).

    - Uses floor/ceil so thin boxes don't collapse to empty after int().
    - Returns x2/y2 as exclusive indices (python slicing convention).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]

    xi1 = int(math.floor(x1))
    yi1 = int(math.floor(y1))
    xi2 = int(math.ceil(x2))
    yi2 = int(math.ceil(y2))

    xi1 = max(0, min(xi1, w - 1))
    yi1 = max(0, min(yi1, h - 1))
    xi2 = max(0, min(xi2, w))
    yi2 = max(0, min(yi2, h))

    if xi2 <= xi1 + 1 or yi2 <= yi1 + 1:
        return None, (xi1, yi1, xi2, yi2)

    crop = img[yi1:yi2, xi1:xi2]
    if crop.size == 0:
        return None, (xi1, yi1, xi2, yi2)
    return crop, (xi1, yi1, xi2, yi2)


def draw_gaze_arrow(frame: np.ndarray, origin: Tuple[int, int], pitch: float, yaw: float, length: int = 120) -> None:
    """Draw a simple 2D gaze arrow.

    This is a visualization; not a full 3D projection.
    """
    ox, oy = origin
    # Map angles to screen deltas: yaw -> x, pitch -> y
    # Looking right (positive yaw) => arrow to the right.
    # Looking down (negative pitch) => arrow down.
    dx = int(length * math.sin(math.radians(yaw)))
    dy = int(-length * math.sin(math.radians(pitch)))
    end = (ox + dx, oy + dy)
    cv2.arrowedLine(frame, (ox, oy), end, (0, 255, 255), 4, tipLength=0.25)


def draw_text_panel(
    frame: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int] = (20, 20),
    font_scale: float = 0.8,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.65,
    line_gap: int = 10,
) -> None:
    """Draw multi-line text with a semi-transparent background panel."""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    if not sizes:
        return
    w = max(s[0] for s in sizes)
    h = sum(s[1] for s in sizes) + (len(lines) - 1) * line_gap

    pad = 10
    x1 = x - pad
    y1 = y - pad
    x2 = x + w + pad
    y2 = y + h + pad

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cy = y
    for t, (tw, th) in zip(lines, sizes):
        cv2.putText(frame, t, (x, cy + th), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cy += th + line_gap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Driver gaze state classification using SCRFD + L2CS (ONNX)")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--scrfd", default="scrfd_person_2.5g.onnx", help="SCRFD ONNX path")
    p.add_argument("--l2cs", default="L2CSNet_gaze360.onnx", help="L2CS ONNX path")
    p.add_argument("--roi", nargs=4, type=int, default=[950, 300, 1650, 690], metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--out-video", default="gaze_output.mp4", help="Output video path")
    p.add_argument("--csv", default="gaze_output.csv", help="Output CSV path")

    p.add_argument("--scrfd-input", nargs=2, type=int, default=[640, 640], metavar=("W", "H"))
    p.add_argument("--l2cs-input", nargs=2, type=int, default=[448, 448], metavar=("W", "H"))

    p.add_argument("--face-conf", type=float, default=0.55, help="SCRFD confidence threshold")
    p.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold")
    p.add_argument("--pre-nms-topk", type=int, default=800, help="Keep top-k face candidates before NMS (0 disables)")
    p.add_argument("--min-face-size", type=int, default=40, help="Filter detections with width/height < this value (ROI pixels)")

    p.add_argument("--bin-size", type=float, default=2.0, help="Histogram bin size (degrees)")
    p.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing")

    p.add_argument("--smooth-alpha", type=float, default=0.2, help="EMA smoothing alpha for pitch/yaw (0 disables)")
    p.add_argument("--smooth-center-alpha", type=float, default=0.3, help="EMA smoothing alpha for face center (0 disables)")
    p.add_argument("--ref-freeze-after", type=int, default=90, help="Freeze reference pitch/yaw after N frames (0 disables)")
    p.add_argument("--ref-strategy", choices=["top_pitch", "hist_mode"], default="top_pitch", help="How to estimate reference gaze")
    p.add_argument("--ref-topk-pitch", type=int, default=60, help="For ref-strategy=top_pitch: use top-K pitch samples")
    p.add_argument("--ref-prefer-pitch", choices=["mode", "high", "low"], default="mode", help="For ref-strategy=hist_mode")
    p.add_argument("--ref-topk", type=int, default=20, help="For ref-strategy=hist_mode: top-k bins considered when prefer-pitch is high/low")
    p.add_argument("--track-max-dist", type=float, default=160.0, help="Max face center distance for tracking (ROI pixels)")
    p.add_argument("--track-topk", type=int, default=50, help="How many top-score faces to consider for tracking")

    # Classification thresholds (delta space)
    # NOTE: pitch sign can vary by model/export; we use a practical "down band" rule.
    p.add_argument("--incar-pitch-neg", type=float, default=-7.0, help="Delta pitch <= this => In-Car (down)")
    p.add_argument("--incar-pitch-pos", type=float, default=2.0, help="Delta pitch >= this (and <= incar-pitch-pos-max) => In-Car (mild down)")
    p.add_argument("--incar-pitch-pos-max", type=float, default=12.0, help="Upper bound for mild-down In-Car band")
    # Backward-compatible alias: maps to incar-pitch-pos
    p.add_argument("--incar-pitch-enter", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--incar-yaw-max", type=float, default=12.0, help="|Delta yaw| <= this required for In-Car")
    p.add_argument("--nonforward-yaw-enter", type=float, default=18.0, help="|Delta yaw| >= this => Non-Forward")
    p.add_argument("--nonforward-pitch-up-enter", type=float, default=15.0, help="Delta pitch >= this => Non-Forward (looking up)")
    p.add_argument("--class-debounce", type=int, default=3, help="Require K consecutive frames before switching class")

    p.add_argument("--write-when-noface", choices=["skip", "noface"], default="noface")
    p.add_argument("--max-frames", type=int, default=0, help="Debug: stop after N frames (0=all)")
    p.add_argument("--no-overlay", dest="no_overlay", action="store_true", help="Do not draw annotations onto output video")
    # Backward-compatible alias
    p.add_argument("--no-display", dest="no_overlay", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    l2cs = L2CSGazeEstimator(
        onnx_path=args.l2cs,
        input_size=(int(args.l2cs_input[0]), int(args.l2cs_input[1])),
    )
    ref_cal = RefCalibrator(
        strategy=str(args.ref_strategy),
        bin_size_deg=float(args.bin_size),
        topk_pitch=int(args.ref_topk_pitch),
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {args.out_video}")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    csv_f = open(args.csv, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "Timestamp",
        "FrameID",
        "Face_Score",
        "Raw_Pitch",
        "Raw_Yaw",
        "Smooth_Pitch",
        "Smooth_Yaw",
        "Ref_Pitch",
        "Ref_Yaw",
        "Delta_Pitch",
        "Delta_Yaw",
        "Gaze_Class",
    ])

    frame_id = 0
    start_t = time.time()
    last_log_t = start_t
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    class_counter: Counter[str] = Counter()
    prev_face_center: Optional[Tuple[float, float]] = None
    pitch_ema = EMAFilter(alpha=float(args.smooth_alpha))
    yaw_ema = EMAFilter(alpha=float(args.smooth_alpha))
    cx_ema = EMAFilter(alpha=float(args.smooth_center_alpha))
    cy_ema = EMAFilter(alpha=float(args.smooth_center_alpha))
    locked_ref: Optional[Tuple[float, float]] = None
    ref_freeze_after = int(args.ref_freeze_after)

    stable_class: str = "No Face"
    pending_class: Optional[str] = None
    pending_count = 0
    debounce_k = max(1, int(args.class_debounce))
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            break

        proc = apply_clahe_bgr(roi) if args.clahe else roi

        faces = scrfd.detect(proc)
        if len(faces) == 0:
            ts = frame_id / fps
            if args.write_when_noface == "noface":
                csv_w.writerow([f"{ts:.3f}", frame_id, "", "", "", "", "", "", "", "", "", "No Face"])
            class_counter["No Face"] += 1
            stable_class = "No Face"
            pending_class = None
            pending_count = 0
            if not args.no_overlay:
                # Always draw ROI + an obvious message
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 6)
                draw_text_panel(
                    frame,
                    [f"Frame={frame_id}", f"t={ts:.2f}s", "No Face"],
                    origin=(20, 20),
                    font_scale=0.9,
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    alpha=0.7,
                )
            writer.write(frame)
            prev_face_center = None
            pitch_ema.reset()
            yaw_ema.reset()
            cx_ema.reset()
            cy_ema.reset()
            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break
            continue

        face, prev_face_center = choose_face_by_tracking(
            faces,
            prev_face_center,
            max_center_dist=float(args.track_max_dist),
            top_k=int(args.track_topk),
        )
        fx1, fy1, fx2, fy2 = face.xyxy
        rw = proc.shape[1]
        rh = proc.shape[0]
        eb = expand_bbox(face.xyxy, rw, rh, scale=1.25)
        face_crop, (c1, r1, c2, r2) = safe_crop_xyxy(proc, eb)
        if face_crop is None:
            ts = frame_id / fps
            if args.write_when_noface == "noface":
                csv_w.writerow([f"{ts:.3f}", frame_id, "", "", "", "", "", "", "", "", "", "No Face"])
            class_counter["No Face"] += 1
            stable_class = "No Face"
            pending_class = None
            pending_count = 0
            if not args.no_overlay:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 6)
                # Still show the detected face bbox even if crop collapsed
                cv2.rectangle(
                    frame,
                    (x1 + int(fx1), y1 + int(fy1)),
                    (x1 + int(fx2), y1 + int(fy2)),
                    (0, 255, 0),
                    4,
                )
                draw_text_panel(
                    frame,
                    [f"Frame={frame_id}", f"t={ts:.2f}s", "Face crop empty"],
                    origin=(20, 20),
                    font_scale=0.9,
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    alpha=0.7,
                )
            writer.write(frame)
            prev_face_center = None
            pitch_ema.reset()
            yaw_ema.reset()
            cx_ema.reset()
            cy_ema.reset()
            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break
            continue

        raw_pitch, raw_yaw = l2cs.infer(face_crop)
        # Smooth raw pitch/yaw to reduce jitter
        if float(args.smooth_alpha) > 0.0:
            smooth_pitch = pitch_ema.update(raw_pitch)
            smooth_yaw = yaw_ema.update(raw_yaw)
        else:
            smooth_pitch, smooth_yaw = float(raw_pitch), float(raw_yaw)

        # Always clip angles before using them for calibration/delta
        smooth_pitch_c = float(np.clip(smooth_pitch, -90.0, 90.0))
        smooth_yaw_c = float(np.clip(smooth_yaw, -90.0, 90.0))

        # Update reference estimator and optionally freeze reference after warm-up
        if locked_ref is None:
            ref_cal.update(smooth_pitch_c, smooth_yaw_c)
            ref_pitch, ref_yaw = ref_cal.get_reference(
                prefer_pitch=str(args.ref_prefer_pitch),
                topk_bins=int(args.ref_topk),
            )
            if ref_freeze_after > 0 and frame_id + 1 >= ref_freeze_after:
                locked_ref = (ref_pitch, ref_yaw)
        else:
            ref_pitch, ref_yaw = locked_ref

        delta_pitch = smooth_pitch_c - ref_pitch
        delta_yaw = smooth_yaw_c - ref_yaw

        incar_pitch_pos = float(args.incar_pitch_pos)
        if getattr(args, "incar_pitch_enter", None) is not None:
            # Backward-compatible alias
            incar_pitch_pos = float(args.incar_pitch_enter)

        cand_class = classify_gaze_v2(
            delta_pitch,
            delta_yaw,
            incar_pitch_neg=float(args.incar_pitch_neg),
            incar_pitch_pos=incar_pitch_pos,
            incar_pitch_pos_max=float(args.incar_pitch_pos_max),
            incar_yaw_max=float(args.incar_yaw_max),
            nonforward_yaw_enter=float(args.nonforward_yaw_enter),
            nonforward_pitch_up_enter=float(args.nonforward_pitch_up_enter),
        )

        # Debounce class switching
        if cand_class == stable_class:
            pending_class = None
            pending_count = 0
        else:
            if pending_class != cand_class:
                pending_class = cand_class
                pending_count = 1
            else:
                pending_count += 1
            if pending_count >= debounce_k:
                stable_class = cand_class
                pending_class = None
                pending_count = 0

        gaze_class = stable_class
        class_counter[gaze_class] += 1

        ts = frame_id / fps
        csv_w.writerow([
            f"{ts:.3f}",
            frame_id,
            f"{face.score:.4f}",
            f"{raw_pitch:.3f}",
            f"{raw_yaw:.3f}",
            f"{smooth_pitch:.3f}",
            f"{smooth_yaw:.3f}",
            f"{ref_pitch:.3f}",
            f"{ref_yaw:.3f}",
            f"{delta_pitch:.3f}",
            f"{delta_yaw:.3f}",
            gaze_class,
        ])

        if not args.no_overlay:
            # Draw ROI and face bbox (map back to full frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 6)
            cv2.rectangle(
                frame,
                (x1 + int(fx1), y1 + int(fy1)),
                (x1 + int(fx2), y1 + int(fy2)),
                (0, 255, 0),
                4,
            )

            # Gaze arrow origin at face center (in full frame coords)
            cx_raw = x1 + float((fx1 + fx2) * 0.5)
            cy_raw = y1 + float((fy1 + fy2) * 0.5)
            if float(args.smooth_center_alpha) > 0.0:
                cx = int(round(cx_ema.update(cx_raw)))
                cy = int(round(cy_ema.update(cy_raw)))
            else:
                cx, cy = int(round(cx_raw)), int(round(cy_raw))

            draw_gaze_arrow(frame, (cx, cy), delta_pitch, delta_yaw, length=140)

            ts2 = frame_id / fps
            class_color = (0, 255, 0) if gaze_class == "Forward" else ((0, 165, 255) if gaze_class == "In-Car" else (0, 0, 255))
            draw_text_panel(
                frame,
                [
                    f"Frame={frame_id}  t={ts2:.2f}s",
                    f"Raw  pitch={raw_pitch:.1f}  yaw={raw_yaw:.1f}",
                    f"Smth pitch={smooth_pitch:.1f}  yaw={smooth_yaw:.1f}",
                    f"Ref  pitch={ref_pitch:.1f}  yaw={ref_yaw:.1f}",
                    f"Delta pitch={delta_pitch:.1f}  yaw={delta_yaw:.1f}",
                    f"Class: {gaze_class}",
                ],
                origin=(20, 20),
                font_scale=0.85,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                alpha=0.7,
            )
            # Extra visible class tag
            cv2.putText(frame, gaze_class, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, class_color, 5, cv2.LINE_AA)

        writer.write(frame)
        frame_id += 1

        if frame_id % 300 == 0:
            now = time.time()
            dt = max(1e-6, now - start_t)
            inst = max(1e-6, now - last_log_t)
            proc_fps = frame_id / dt
            last_log_t = now
            if total_frames > 0:
                remain = max(0, total_frames - frame_id)
                eta_s = remain / max(1e-6, proc_fps)
                print(f"[{frame_id}/{total_frames}] proc_fps={proc_fps:.2f} elapsed={dt/60:.1f}m ETA={eta_s/60:.1f}m")
            else:
                print(f"[{frame_id}] proc_fps={proc_fps:.2f} elapsed={dt/60:.1f}m")

        if args.max_frames and frame_id >= args.max_frames:
            break

    csv_f.close()
    cap.release()
    writer.release()

    # Summary
    total = sum(class_counter.values())
    summary = {
        "video": args.video,
        "out_video": args.out_video,
        "csv": args.csv,
        "total_frames_written": int(frame_id),
        "class_counts": dict(class_counter),
        "class_percent": {k: (v / total * 100.0 if total else 0.0) for k, v in class_counter.items()},
    }
    try:
        with open(args.csv + ".summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print("Summary (counts):", dict(class_counter))
    if total:
        pct = {k: round(v / total * 100.0, 2) for k, v in class_counter.items()}
        print("Summary (percent):", pct)
    print("Summary JSON:", args.csv + ".summary.json")

    print("Done")
    print(f"Video: {args.video}")
    print(f"Out video: {args.out_video}")
    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
