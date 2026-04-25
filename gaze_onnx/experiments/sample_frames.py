#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sample frames for manual labeling.

Goal: create a *small* but informative labeled set to separate:
- model output issues (pitch/yaw themselves wrong)
- strategy issues (thresholds/ref/debounce mapping wrong)

Typical usage:
  python gaze_onnx/experiments/sample_frames.py \
    --video gaze_onnx/output/output_gaze_smooth4_full.mp4 \
    --pred-csv gaze_onnx/output/output_gaze_smooth4_full.csv \
    --out-dir gaze_onnx/experiments/samples_smooth4_full \
    --n-total 360 --time-bins 6 --include-noface 0
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2


PRED_FIELDS = [
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
]


@dataclass(frozen=True)
class PredRow:
    timestamp: float
    frame_id: int
    face_score: Optional[float]
    raw_pitch: Optional[float]
    raw_yaw: Optional[float]
    smooth_pitch: Optional[float]
    smooth_yaw: Optional[float]
    ref_pitch: Optional[float]
    ref_yaw: Optional[float]
    delta_pitch: Optional[float]
    delta_yaw: Optional[float]
    gaze_class: str


def _to_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _to_int(x: str) -> int:
    return int(float(x))


def read_pred_csv(path: str) -> List[PredRow]:
    rows: List[PredRow] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for k in PRED_FIELDS:
            if k not in (r.fieldnames or []):
                raise ValueError(f"Missing column '{k}' in {path}. Found: {r.fieldnames}")
        for row in r:
            rows.append(
                PredRow(
                    timestamp=float(row["Timestamp"]),
                    frame_id=_to_int(row["FrameID"]),
                    face_score=_to_float(row["Face_Score"]),
                    raw_pitch=_to_float(row["Raw_Pitch"]),
                    raw_yaw=_to_float(row["Raw_Yaw"]),
                    smooth_pitch=_to_float(row["Smooth_Pitch"]),
                    smooth_yaw=_to_float(row["Smooth_Yaw"]),
                    ref_pitch=_to_float(row["Ref_Pitch"]),
                    ref_yaw=_to_float(row["Ref_Yaw"]),
                    delta_pitch=_to_float(row["Delta_Pitch"]),
                    delta_yaw=_to_float(row["Delta_Yaw"]),
                    gaze_class=str(row["Gaze_Class"]),
                )
            )
    return rows


def margin_to_thresholds(
    row: PredRow,
    *,
    incar_pitch_neg: float,
    incar_pitch_pos: float,
    incar_pitch_pos_max: float,
    incar_yaw_max: float,
    nonforward_yaw_enter: float,
    nonforward_pitch_up_enter: float,
) -> Optional[float]:
    """Heuristic: smaller margin => closer to decision boundary => likely strategy-sensitive."""
    if row.delta_pitch is None or row.delta_yaw is None:
        return None
    dp = float(row.delta_pitch)
    dy = abs(float(row.delta_yaw))

    # Distances to the key boundaries (0 means on boundary, negative means inside the region).
    d_yaw_nf = abs(dy - float(nonforward_yaw_enter))
    d_pitch_up = abs(dp - float(nonforward_pitch_up_enter))

    # In-car band boundaries
    d_incar_yaw = abs(dy - float(incar_yaw_max))
    d_incar_neg = abs(dp - float(incar_pitch_neg))
    d_incar_pos_lo = abs(dp - float(incar_pitch_pos))
    d_incar_pos_hi = abs(dp - float(incar_pitch_pos_max))

    return float(min(d_yaw_nf, d_pitch_up, d_incar_yaw, d_incar_neg, d_incar_pos_lo, d_incar_pos_hi))


def put_panel(img, lines: Sequence[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    x, y = 12, 18
    line_gap = 6

    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    w = max((s[0] for s in sizes), default=0)
    h = (sum((s[1] for s in sizes)) if sizes else 0) + max(0, len(lines) - 1) * line_gap
    pad = 8

    x1, y1 = x - pad, y - pad
    x2, y2 = x + w + pad, y + h + pad
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1] - 1, x2)
    y2 = min(img.shape[0] - 1, y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cy = y
    for t, (tw, th) in zip(lines, sizes):
        cv2.putText(img, t, (x, cy + th), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cy += th + line_gap


def read_frame(cap: cv2.VideoCapture, frame_id: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def stratified_sample(
    rows: List[PredRow],
    *,
    n_total: int,
    time_bins: int,
    seed: int,
    include_noface: bool,
    boundary_frac: float,
    thresholds: Dict[str, float],
) -> List[PredRow]:
    rng = random.Random(int(seed))

    candidates = rows if include_noface else [r for r in rows if r.gaze_class != "No Face"]
    if not candidates:
        return []

    t0 = candidates[0].timestamp
    t1 = candidates[-1].timestamp
    dur = max(1e-6, t1 - t0)

    # Assign time bin
    def tb(r: PredRow) -> int:
        x = (r.timestamp - t0) / dur
        b = int(x * int(time_bins))
        return max(0, min(int(time_bins) - 1, b))

    groups: Dict[Tuple[int, str], List[PredRow]] = {}
    for r in candidates:
        key = (tb(r), r.gaze_class)
        groups.setdefault(key, []).append(r)

    classes = sorted({r.gaze_class for r in candidates})

    # Base stratified sample
    per_stratum = max(1, int(n_total * (1.0 - float(boundary_frac)) / max(1, len(groups))))
    chosen: List[PredRow] = []
    for key, g in groups.items():
        if not g:
            continue
        k = min(per_stratum, len(g))
        chosen.extend(rng.sample(g, k=k))

    # Boundary-focused sample (near thresholds)
    n_boundary = max(0, int(n_total * float(boundary_frac)))
    if n_boundary > 0:
        scored: List[Tuple[float, PredRow]] = []
        for r in candidates:
            m = margin_to_thresholds(r, **thresholds)
            if m is None:
                continue
            scored.append((m, r))
        scored.sort(key=lambda x: x[0])  # smallest margin first
        boundary_rows = [r for _, r in scored[: min(n_boundary * 3, len(scored))]]
        if boundary_rows:
            chosen.extend(rng.sample(boundary_rows, k=min(n_boundary, len(boundary_rows))))

    # Deduplicate by frame_id
    uniq: Dict[int, PredRow] = {}
    for r in chosen:
        uniq[r.frame_id] = r

    out = list(uniq.values())
    out.sort(key=lambda r: r.frame_id)

    # If we're short (due to dedupe / empty strata), top up from remaining candidates.
    if len(out) < n_total:
        remaining = [r for r in candidates if r.frame_id not in uniq]
        need = min(int(n_total) - len(out), len(remaining))
        if need > 0:
            out.extend(rng.sample(remaining, k=need))
            out.sort(key=lambda r: r.frame_id)

    # Cap to n_total
    if len(out) > n_total:
        out = rng.sample(out, k=n_total)
        out.sort(key=lambda r: r.frame_id)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample frames from a gaze CSV for manual labeling")
    p.add_argument("--video", required=True, help="Video to sample frames from (raw or annotated mp4)")
    p.add_argument("--pred-csv", required=True, help="Prediction CSV produced by gaze_state_onnx.py")
    p.add_argument("--out-dir", required=True, help="Output dir for PNGs + manifest.csv")
    p.add_argument("--n-total", type=int, default=360, help="Total number of frames to sample")
    p.add_argument("--time-bins", type=int, default=6, help="Stratify by time bins")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--include-noface", type=int, default=0, help="Include No Face samples (1/0)")
    p.add_argument("--boundary-frac", type=float, default=0.35, help="Fraction reserved for boundary-near samples")

    # Same defaults as gaze_state_onnx.py, so boundary sampling matches your current strategy
    p.add_argument("--incar-pitch-neg", type=float, default=-7.0)
    p.add_argument("--incar-pitch-pos", type=float, default=2.0)
    p.add_argument("--incar-pitch-pos-max", type=float, default=12.0)
    p.add_argument("--incar-yaw-max", type=float, default=12.0)
    p.add_argument("--nonforward-yaw-enter", type=float, default=18.0)
    p.add_argument("--nonforward-pitch-up-enter", type=float, default=15.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_pred_csv(args.pred_csv)

    thresholds = {
        "incar_pitch_neg": float(args.incar_pitch_neg),
        "incar_pitch_pos": float(args.incar_pitch_pos),
        "incar_pitch_pos_max": float(args.incar_pitch_pos_max),
        "incar_yaw_max": float(args.incar_yaw_max),
        "nonforward_yaw_enter": float(args.nonforward_yaw_enter),
        "nonforward_pitch_up_enter": float(args.nonforward_pitch_up_enter),
    }

    sample_rows = stratified_sample(
        rows,
        n_total=int(args.n_total),
        time_bins=int(args.time_bins),
        seed=int(args.seed),
        include_noface=bool(int(args.include_noface)),
        boundary_frac=float(args.boundary_frac),
        thresholds=thresholds,
    )
    if not sample_rows:
        raise SystemExit("No rows sampled. Check inputs.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    wrote = 0
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img", "FrameID", "Timestamp", "Pred_Class", "Raw_Pitch", "Raw_Yaw", "Smooth_Pitch", "Smooth_Yaw", "Ref_Pitch", "Ref_Yaw", "Delta_Pitch", "Delta_Yaw"])

        for r in sample_rows:
            frame = read_frame(cap, r.frame_id)
            if frame is None:
                continue

            lines = [
                f"Frame={r.frame_id}  t={r.timestamp:.2f}s",
                f"Pred={r.gaze_class}",
            ]
            if r.smooth_pitch is not None and r.smooth_yaw is not None:
                lines.append(f"Smth pitch={r.smooth_pitch:.1f} yaw={r.smooth_yaw:.1f}")
            if r.ref_pitch is not None and r.ref_yaw is not None:
                lines.append(f"Ref  pitch={r.ref_pitch:.1f} yaw={r.ref_yaw:.1f}")
            if r.delta_pitch is not None and r.delta_yaw is not None:
                lines.append(f"Delta pitch={r.delta_pitch:.1f} yaw={r.delta_yaw:.1f}")
            put_panel(frame, lines)

            img_name = f"f{r.frame_id:06d}_t{r.timestamp:08.3f}_{r.gaze_class}.png"
            img_path = os.path.join(args.out_dir, img_name)
            cv2.imwrite(img_path, frame)

            wrote += 1

            w.writerow([
                img_name,
                r.frame_id,
                f"{r.timestamp:.3f}",
                r.gaze_class,
                "" if r.raw_pitch is None else f"{r.raw_pitch:.3f}",
                "" if r.raw_yaw is None else f"{r.raw_yaw:.3f}",
                "" if r.smooth_pitch is None else f"{r.smooth_pitch:.3f}",
                "" if r.smooth_yaw is None else f"{r.smooth_yaw:.3f}",
                "" if r.ref_pitch is None else f"{r.ref_pitch:.3f}",
                "" if r.ref_yaw is None else f"{r.ref_yaw:.3f}",
                "" if r.delta_pitch is None else f"{r.delta_pitch:.3f}",
                "" if r.delta_yaw is None else f"{r.delta_yaw:.3f}",
            ])

    cap.release()
    print(f"Requested {len(sample_rows)} rows, wrote {wrote} samples")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
