#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assign gaze/wheel ROIs from two fixed candidates with swap correction.

The script scores two candidate ROIs by face evidence and assigns:
- higher face-evidence ROI -> gaze
- the other ROI -> wheel

It can process one or many videos and saves a per-video assignment CSV.
Optionally, it can run downstream gaze/wheel inference scripts with assigned ROIs.
"""

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Reuse existing SCRFD wrapper and face-area helper.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_state_cls import SCRFDDetector, face_area_ratio  # noqa: E402


ROI = Tuple[int, int, int, int]
DEFAULT_ROI_A: ROI = (0, 0, 1900, 1100)
DEFAULT_ROI_B: ROI = (1900, 660, 3300, 1400)


@dataclass
class ROIStats:
    sampled_frames: int = 0
    valid_frames: int = 0
    face_hits: int = 0
    score_sum: float = 0.0
    area_sum: float = 0.0

    @property
    def hit_ratio(self) -> float:
        if self.valid_frames <= 0:
            return 0.0
        return float(self.face_hits) / float(self.valid_frames)

    @property
    def mean_face_score(self) -> float:
        if self.face_hits <= 0:
            return 0.0
        return float(self.score_sum) / float(self.face_hits)

    @property
    def mean_face_ratio(self) -> float:
        if self.face_hits <= 0:
            return 0.0
        return float(self.area_sum) / float(self.face_hits)

    def evidence(self, area_ref: float) -> float:
        area_ref = max(1e-6, float(area_ref))
        area_term = min(1.0, self.mean_face_ratio / area_ref)
        # Heavier weight on "how often face appears" for swap robustness.
        return 0.72 * self.hit_ratio + 0.20 * self.mean_face_score + 0.08 * area_term


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assign gaze/wheel ROI from two fixed candidate ROIs")
    p.add_argument("--video", action="append", default=[], help="Video path (repeatable)")
    p.add_argument("--videos-txt", default="", help="Optional txt file, one video path per line")
    p.add_argument(
        "--videos-csv",
        default="",
        help="Optional CSV file with at least a 'video' column",
    )
    p.add_argument("--max-videos", type=int, default=0, help="Debug: process only first N videos (0=all)")

    p.add_argument("--roi-a", nargs=4, type=int, default=list(DEFAULT_ROI_A), metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--roi-b", nargs=4, type=int, default=list(DEFAULT_ROI_B), metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument(
        "--base-gaze-roi",
        choices=["A", "B"],
        default="A",
        help="Canonical gaze ROI before swap correction. swapped=True means final gaze_roi != base_gaze_roi",
    )

    p.add_argument("--scrfd-model", default="models/scrfd_person_2.5g.onnx")
    p.add_argument("--scrfd-input", nargs=2, type=int, default=[640, 640], metavar=("W", "H"))
    p.add_argument("--face-conf", type=float, default=0.5)
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--pre-nms-topk", type=int, default=800)
    p.add_argument("--min-face-size", type=int, default=30)

    p.add_argument("--samples", type=int, default=64, help="Sampled frames per video for ROI scoring")
    p.add_argument("--uncertain-margin", type=float, default=0.06, help="Low confidence if |evidence_a-evidence_b| < this")
    p.add_argument("--min-hit-ratio", type=float, default=0.05, help="Low confidence if both ROI face hit-ratios are below this")
    p.add_argument(
        "--face-area-ref",
        type=float,
        default=0.02,
        help="Used in absolute evidence for diagnostics only",
    )

    p.add_argument(
        "--assignment-csv",
        default="gaze_onnx/experiments/output/dual_roi_assignment.csv",
        help="Output CSV for per-video assignment",
    )
    p.add_argument("--preview-dir", default="", help="Optional directory to save 1 preview image per video")
    p.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any video fails")

    p.add_argument("--run-infer", action="store_true", help="Run gaze + wheel inference after assignment")
    p.add_argument("--python-bin", default=sys.executable, help="Python executable for downstream scripts")
    p.add_argument("--gaze-script", default="gaze_onnx/gaze_state_cls.py")
    p.add_argument("--wheel-script", default="driver_monitor/hand_on_wheel.py")
    p.add_argument("--gaze-cls-model", default="models/gaze_cls_yolov8n.onnx")
    p.add_argument("--run-out-dir", default="gaze_onnx/experiments/output/dual_roi_runs")
    p.add_argument("--wheel-config", default="", help="Optional override for hand_on_wheel.py --config")
    p.add_argument("--wheel-weights", default="", help="Optional override for hand_on_wheel.py --weights")
    p.add_argument("--wheel-device", default="", help="Optional override for hand_on_wheel.py --device")
    p.add_argument("--wheel-sample-fps", type=float, default=0.0, help="Optional override for hand_on_wheel.py --sample-fps")
    return p.parse_args()


def sample_indices(total_frames: int, n_samples: int) -> List[int]:
    if total_frames <= 1:
        return [0]
    n = max(1, min(int(n_samples), int(total_frames)))
    out: List[int] = []
    for i in range(n):
        out.append(int(round(i * (total_frames - 1) / max(1, n - 1))))
    return sorted(set(out))


def roi_to_str(roi: Optional[ROI]) -> str:
    if roi is None:
        return ""
    return ",".join(str(int(v)) for v in roi)


def clamp_roi(roi: ROI, width: int, height: int) -> Optional[ROI]:
    if width <= 1 or height <= 1:
        return None
    x1, y1, x2, y2 = [int(v) for v in roi]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def collect_videos(args: argparse.Namespace) -> List[str]:
    videos: List[str] = []
    videos.extend([str(v).strip() for v in args.video if str(v).strip()])

    if args.videos_txt:
        with open(args.videos_txt, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t or t.startswith("#"):
                    continue
                videos.append(t)

    if args.videos_csv:
        with open(args.videos_csv, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if "video" not in (r.fieldnames or []):
                raise ValueError(f"CSV {args.videos_csv} missing required column: video")
            for row in r:
                p = str(row.get("video", "")).strip()
                if p:
                    videos.append(p)

    # stable de-dup
    out: List[str] = []
    seen = set()
    for v in videos:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)

    if args.max_videos > 0:
        out = out[: int(args.max_videos)]
    return out


def annotate_preview(
    frame: np.ndarray,
    roi_a: Optional[ROI],
    roi_b: Optional[ROI],
    gaze_key: str,
    uncertain: bool,
    out_path: str,
) -> None:
    vis = frame.copy()
    if roi_a is not None:
        x1, y1, x2, y2 = roi_a
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(vis, "ROI-A", (x1 + 8, max(24, y1 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    if roi_b is not None:
        x1, y1, x2, y2 = roi_b
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(vis, "ROI-B", (x1 + 8, max(24, y1 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    label = f"Assigned GAZE={gaze_key} | WHEEL={'B' if gaze_key == 'A' else 'A'}"
    if uncertain:
        label += " | LOW_MARGIN"
    cv2.putText(vis, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)


def run_cmd(cmd: Sequence[str]) -> Tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    text = proc.stdout if proc.stdout is not None else ""
    return int(proc.returncode), text


def build_output_prefix(video_path: str, out_dir: str) -> str:
    p = Path(video_path)
    stem = p.stem.replace(" ", "_")
    # Keep names deterministic while reducing collision risk.
    parent_tail = p.parent.name.replace(" ", "_")
    base = f"{parent_tail}__{stem}" if parent_tail else stem
    return str(Path(out_dir) / base)


def relative_pair(a: float, b: float) -> Tuple[float, float]:
    a = float(max(0.0, a))
    b = float(max(0.0, b))
    s = a + b
    if s <= 1e-9:
        return 0.5, 0.5
    return a / s, b / s


def main() -> None:
    args = parse_args()
    videos = collect_videos(args)
    if not videos:
        raise ValueError("No videos provided. Use --video / --videos-txt / --videos-csv")

    detector = SCRFDDetector(
        onnx_path=args.scrfd_model,
        input_size=(int(args.scrfd_input[0]), int(args.scrfd_input[1])),
        conf_thresh=float(args.face_conf),
        nms_thresh=float(args.nms),
        pre_nms_topk=int(args.pre_nms_topk),
        min_face_size=int(args.min_face_size),
    )

    roi_a_raw: ROI = tuple(int(v) for v in args.roi_a)  # type: ignore[assignment]
    roi_b_raw: ROI = tuple(int(v) for v in args.roi_b)  # type: ignore[assignment]
    rows: List[Dict[str, str]] = []

    for i, video in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] {video}")
        row: Dict[str, str] = {
            "video": video,
            "status": "ok",
            "reason": "",
            "roi_a": roi_to_str(roi_a_raw),
            "roi_b": roi_to_str(roi_b_raw),
            "gaze_csv": "",
            "gaze_video": "",
            "wheel_csv": "",
            "wheel_video": "",
            "preview_path": "",
        }

        if not os.path.exists(video):
            row["status"] = "error"
            row["reason"] = "video_not_found"
            rows.append(row)
            print("  [WARN] video_not_found")
            continue

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            row["status"] = "error"
            row["reason"] = "open_failed"
            rows.append(row)
            print("  [WARN] open_failed")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        roi_a = clamp_roi(roi_a_raw, width, height)
        roi_b = clamp_roi(roi_b_raw, width, height)

        row["width"] = str(width)
        row["height"] = str(height)
        row["total_frames"] = str(total_frames)
        row["roi_a_clamped"] = roi_to_str(roi_a)
        row["roi_b_clamped"] = roi_to_str(roi_b)

        if roi_a is None and roi_b is None:
            cap.release()
            row["status"] = "error"
            row["reason"] = "both_rois_invalid_after_clamp"
            rows.append(row)
            print("  [WARN] both_rois_invalid_after_clamp")
            continue

        idxs = sample_indices(total_frames, int(args.samples))
        stats = {"A": ROIStats(), "B": ROIStats()}
        frames_read = 0
        preview_frame: Optional[np.ndarray] = None

        for fid in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames_read += 1
            if preview_frame is None:
                preview_frame = frame.copy()

            for key, roi in (("A", roi_a), ("B", roi_b)):
                stats[key].sampled_frames += 1
                if roi is None:
                    continue
                x1, y1, x2, y2 = roi
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                stats[key].valid_frames += 1
                faces = detector.detect(crop)
                if not faces:
                    continue
                top = faces[0]
                stats[key].face_hits += 1
                stats[key].score_sum += float(top.score)
                stats[key].area_sum += float(face_area_ratio(top.xyxy, crop.shape[1], crop.shape[0]))

        cap.release()

        abs_ev_a = stats["A"].evidence(float(args.face_area_ref))
        abs_ev_b = stats["B"].evidence(float(args.face_area_ref))

        rel_hit_a, rel_hit_b = relative_pair(stats["A"].hit_ratio, stats["B"].hit_ratio)
        rel_score_a, rel_score_b = relative_pair(stats["A"].mean_face_score, stats["B"].mean_face_score)
        rel_area_a, rel_area_b = relative_pair(stats["A"].mean_face_ratio, stats["B"].mean_face_ratio)

        # Pairwise evidence (A+B ~= 1.0): robust against both ROIs having faces.
        ev_a = 0.50 * rel_hit_a + 0.20 * rel_score_a + 0.30 * rel_area_a
        ev_b = 0.50 * rel_hit_b + 0.20 * rel_score_b + 0.30 * rel_area_b
        margin = abs(ev_a - ev_b)

        gaze_key = "A" if ev_a >= ev_b else "B"
        wheel_key = "B" if gaze_key == "A" else "A"
        gaze_roi = roi_a if gaze_key == "A" else roi_b
        wheel_roi = roi_b if gaze_key == "A" else roi_a

        low_hit = max(stats["A"].hit_ratio, stats["B"].hit_ratio) < float(args.min_hit_ratio)
        uncertain = (margin < float(args.uncertain_margin)) or low_hit
        swapped = gaze_key != str(args.base_gaze_roi)

        row.update(
            {
                "sampled_frames": str(frames_read),
                "a_hit_ratio": f"{stats['A'].hit_ratio:.6f}",
                "a_mean_face_score": f"{stats['A'].mean_face_score:.6f}",
                "a_mean_face_ratio": f"{stats['A'].mean_face_ratio:.6f}",
                "a_abs_evidence": f"{abs_ev_a:.6f}",
                "a_evidence": f"{ev_a:.6f}",
                "a_rel_hit": f"{rel_hit_a:.6f}",
                "a_rel_score": f"{rel_score_a:.6f}",
                "a_rel_area": f"{rel_area_a:.6f}",
                "b_hit_ratio": f"{stats['B'].hit_ratio:.6f}",
                "b_mean_face_score": f"{stats['B'].mean_face_score:.6f}",
                "b_mean_face_ratio": f"{stats['B'].mean_face_ratio:.6f}",
                "b_abs_evidence": f"{abs_ev_b:.6f}",
                "b_evidence": f"{ev_b:.6f}",
                "b_rel_hit": f"{rel_hit_b:.6f}",
                "b_rel_score": f"{rel_score_b:.6f}",
                "b_rel_area": f"{rel_area_b:.6f}",
                "evidence_margin": f"{margin:.6f}",
                "assignment_uncertain": "1" if uncertain else "0",
                "gaze_roi_key": gaze_key,
                "wheel_roi_key": wheel_key,
                "gaze_roi": roi_to_str(gaze_roi),
                "wheel_roi": roi_to_str(wheel_roi),
                "swapped": "1" if swapped else "0",
            }
        )

        if args.preview_dir and preview_frame is not None:
            preview_name = build_output_prefix(video, args.preview_dir) + ".jpg"
            annotate_preview(preview_frame, roi_a, roi_b, gaze_key, uncertain, preview_name)
            row["preview_path"] = preview_name

        print(
            "  "
            + f"A(ev={ev_a:.3f}, hit={stats['A'].hit_ratio:.3f})  "
            + f"B(ev={ev_b:.3f}, hit={stats['B'].hit_ratio:.3f})  "
            + f"-> gaze={gaze_key}, swapped={int(swapped)}, uncertain={int(uncertain)}"
        )

        if args.run_infer:
            if gaze_roi is None or wheel_roi is None:
                row["status"] = "error"
                row["reason"] = "assigned_roi_invalid"
            else:
                os.makedirs(args.run_out_dir, exist_ok=True)
                out_prefix = build_output_prefix(video, args.run_out_dir)
                gaze_out_video = out_prefix + ".gaze.mp4"
                gaze_out_csv = out_prefix + ".gaze.csv"
                wheel_out_video = out_prefix + ".wheel.mp4"
                wheel_out_csv = out_prefix + ".wheel.csv"

                gaze_cmd = [
                    args.python_bin,
                    args.gaze_script,
                    "--video",
                    video,
                    "--roi",
                    str(gaze_roi[0]),
                    str(gaze_roi[1]),
                    str(gaze_roi[2]),
                    str(gaze_roi[3]),
                    "--scrfd",
                    args.scrfd_model,
                    "--cls-model",
                    args.gaze_cls_model,
                    "--out-video",
                    gaze_out_video,
                    "--csv",
                    gaze_out_csv,
                ]
                rc_g, log_g = run_cmd(gaze_cmd)
                if rc_g != 0:
                    row["status"] = "error"
                    row["reason"] = f"gaze_infer_failed({rc_g})"
                    print("  [ERR] gaze inference failed")
                    print(log_g[-800:])
                else:
                    row["gaze_video"] = gaze_out_video
                    row["gaze_csv"] = gaze_out_csv

                wheel_cmd = [
                    args.python_bin,
                    args.wheel_script,
                    "--video",
                    video,
                    "--roi",
                    str(wheel_roi[0]),
                    str(wheel_roi[1]),
                    str(wheel_roi[2]),
                    str(wheel_roi[3]),
                    "--output",
                    wheel_out_video,
                    "--state-csv",
                    wheel_out_csv,
                ]
                if args.wheel_config:
                    wheel_cmd.extend(["--config", args.wheel_config])
                if args.wheel_weights:
                    wheel_cmd.extend(["--weights", args.wheel_weights])
                if args.wheel_device:
                    wheel_cmd.extend(["--device", args.wheel_device])
                if args.wheel_sample_fps > 0:
                    wheel_cmd.extend(["--sample-fps", str(float(args.wheel_sample_fps))])

                rc_w, log_w = run_cmd(wheel_cmd)
                if rc_w != 0:
                    row["status"] = "error"
                    if row["reason"]:
                        row["reason"] += f"+wheel_infer_failed({rc_w})"
                    else:
                        row["reason"] = f"wheel_infer_failed({rc_w})"
                    print("  [ERR] wheel inference failed")
                    print(log_w[-800:])
                else:
                    row["wheel_video"] = wheel_out_video
                    row["wheel_csv"] = wheel_out_csv

        rows.append(row)

    fields = [
        "video",
        "status",
        "reason",
        "width",
        "height",
        "total_frames",
        "sampled_frames",
        "roi_a",
        "roi_b",
        "roi_a_clamped",
        "roi_b_clamped",
        "a_hit_ratio",
        "a_mean_face_score",
        "a_mean_face_ratio",
        "a_abs_evidence",
        "a_evidence",
        "a_rel_hit",
        "a_rel_score",
        "a_rel_area",
        "b_hit_ratio",
        "b_mean_face_score",
        "b_mean_face_ratio",
        "b_abs_evidence",
        "b_evidence",
        "b_rel_hit",
        "b_rel_score",
        "b_rel_area",
        "evidence_margin",
        "assignment_uncertain",
        "gaze_roi_key",
        "wheel_roi_key",
        "gaze_roi",
        "wheel_roi",
        "swapped",
        "preview_path",
        "gaze_csv",
        "gaze_video",
        "wheel_csv",
        "wheel_video",
    ]

    out_csv = Path(args.assignment_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    n_err = sum(1 for r in rows if r.get("status", "") != "ok")
    n_swap = sum(1 for r in rows if r.get("swapped", "0") == "1")
    n_uncertain = sum(1 for r in rows if r.get("assignment_uncertain", "0") == "1")

    print("")
    print(f"Saved assignment CSV: {out_csv}")
    print(f"Videos: total={len(rows)} ok={len(rows) - n_err} error={n_err}")
    print(f"Swap-detected: {n_swap}, low-margin assignments: {n_uncertain}")
    if n_err > 0 and args.fail_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
