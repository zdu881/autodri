#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a manual audit annotation pack from segment-level gaze inference outputs.

Use this script when you want humans to spot-check model outputs on current
inference results rather than on the original training annotation pack.

Input:
- plan CSV from build_p1_infer_plan.py / participant current infer plan
- each row must contain:
  video_path, segment_uid, gaze_csv, gaze_roi_x1..gaze_roi_y2

Output:
- out_dir/images/*.jpg
- out_dir/manifest.csv  (compatible with web_label_tool.py)
- out_dir/labels.csv    (blank labels in manifest order)
- out_dir/README.txt

Sampling strategy:
- uniform frames across each segment
- transition-adjacent frames where predicted class changes
- low-confidence frames for manual hard-case review
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2


@dataclass(frozen=True)
class PlanRow:
    participant: str
    segment_uid: str
    video_path: str
    gaze_csv: str
    roi: Tuple[int, int, int, int]


@dataclass(frozen=True)
class AuditPick:
    participant: str
    segment_uid: str
    video_path: str
    video_frame: int
    video_timestamp: float
    pred_class: str
    confidence: str
    audit_type: str
    roi: Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a gaze inference audit pack for manual review")
    p.add_argument("--plan-csv", required=True, help="Infer plan CSV with gaze_csv and gaze_roi columns")
    p.add_argument("--out-dir", required=True, help="Output annotation pack directory")
    p.add_argument("--per-segment-uniform", type=int, default=2, help="Uniform samples per segment")
    p.add_argument("--per-segment-transition", type=int, default=1, help="Transition samples per segment")
    p.add_argument("--per-segment-lowconf", type=int, default=1, help="Low-confidence samples per segment")
    p.add_argument("--min-transition-gap", type=int, default=15, help="Min frame gap between transition picks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--max-total", type=int, default=0, help="Optional cap on total sampled images")
    return p.parse_args()


def _to_int(x: str, d: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return d


def _to_float(x: str, d: float = 0.0) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return d


def _safe_token(text: str, limit: int = 48) -> str:
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    t = t.strip("._-")
    if not t:
        t = "v"
    return t[:limit]


def read_plan(path: Path) -> List[PlanRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out: List[PlanRow] = []
    for r in rows:
        status = str(r.get("status", "")).strip().lower()
        if status and status != "ok":
            continue
        video_path = str(r.get("video_path", "")).strip()
        gaze_csv = str(r.get("gaze_csv", "")).strip()
        segment_uid = str(r.get("segment_uid", "")).strip()
        if not video_path or not gaze_csv or not segment_uid:
            continue
        participant = segment_uid.split("_seg_")[0] if "_seg_" in segment_uid else "audit"
        out.append(
            PlanRow(
                participant=participant,
                segment_uid=segment_uid,
                video_path=video_path,
                gaze_csv=gaze_csv,
                roi=(
                    _to_int(r.get("gaze_roi_x1", "0")),
                    _to_int(r.get("gaze_roi_y1", "0")),
                    _to_int(r.get("gaze_roi_x2", "0")),
                    _to_int(r.get("gaze_roi_y2", "0")),
                ),
            )
        )
    return out


def sample_uniform_indices(n_rows: int, n_samples: int) -> List[int]:
    if n_rows <= 0 or n_samples <= 0:
        return []
    if n_samples >= n_rows:
        return list(range(n_rows))
    out = []
    for i in range(n_samples):
        x = (i + 0.5) * n_rows / n_samples
        idx = min(n_rows - 1, max(0, int(round(x - 0.5))))
        out.append(idx)
    return sorted(set(out))


def sample_transition_indices(rows: Sequence[dict], n_samples: int, min_gap: int) -> List[int]:
    if n_samples <= 0 or not rows:
        return []
    cand: List[int] = []
    prev = str(rows[0].get("Gaze_Class", "")).strip()
    for i in range(1, len(rows)):
        cur = str(rows[i].get("Gaze_Class", "")).strip()
        if cur and prev and cur != prev:
            cand.append(i)
        prev = cur or prev
    out: List[int] = []
    for idx in cand:
        if not out or idx - out[-1] >= max(1, min_gap):
            out.append(idx)
        if len(out) >= n_samples:
            break
    return out


def sample_lowconf_indices(rows: Sequence[dict], n_samples: int) -> List[int]:
    if n_samples <= 0 or not rows:
        return []
    scored = []
    for i, r in enumerate(rows):
        pred = str(r.get("Gaze_Class", "")).strip()
        if not pred:
            continue
        conf = str(r.get("Confidence", "")).strip()
        if not conf:
            continue
        try:
            c = float(conf)
        except Exception:
            continue
        scored.append((c, i))
    scored.sort(key=lambda x: x[0])
    return [i for _, i in scored[:n_samples]]


def choose_picks(rows: List[dict], plan: PlanRow, args: argparse.Namespace) -> List[AuditPick]:
    picks: Dict[Tuple[int, str], AuditPick] = {}

    def add_pick(idx: int, audit_type: str) -> None:
        if idx < 0 or idx >= len(rows):
            return
        r = rows[idx]
        pred_class = str(r.get("Gaze_Class", "")).strip()
        if not pred_class:
            return
        video_frame = _to_int(r.get("Video_FrameID", r.get("FrameID", "0")))
        video_timestamp = _to_float(r.get("Video_Timestamp", r.get("Timestamp", "0")))
        key = (video_frame, video_timestamp.__hash__())
        if key in picks:
            return
        picks[key] = AuditPick(
            participant=plan.participant,
            segment_uid=plan.segment_uid,
            video_path=plan.video_path,
            video_frame=video_frame,
            video_timestamp=video_timestamp,
            pred_class=pred_class,
            confidence=str(r.get("Confidence", "")).strip(),
            audit_type=audit_type,
            roi=plan.roi,
        )

    for idx in sample_uniform_indices(len(rows), int(args.per_segment_uniform)):
        add_pick(idx, "uniform")
    for idx in sample_transition_indices(rows, int(args.per_segment_transition), int(args.min_transition_gap)):
        add_pick(idx, "transition")
    for idx in sample_lowconf_indices(rows, int(args.per_segment_lowconf)):
        add_pick(idx, "lowconf")

    return list(picks.values())


def clamp_roi(roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), max(0, w - 1)))
    y1 = max(0, min(int(y1), max(0, h - 1)))
    x2 = max(1, min(int(x2), max(1, w)))
    y2 = max(1, min(int(y2), max(1, h)))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def read_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, (0, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if not ok or frame is None:
        return None, (w, h)
    return frame, (w, h)


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    plan_rows = read_plan(Path(args.plan_csv))
    if not plan_rows:
        raise SystemExit("No usable rows found in plan csv")

    all_picks: List[AuditPick] = []
    for plan in plan_rows:
        gpath = Path(plan.gaze_csv)
        if not gpath.is_absolute():
            gpath = Path.cwd() / gpath
        if not gpath.exists():
            continue
        with gpath.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        all_picks.extend(choose_picks(rows, plan, args))

    if args.max_total > 0 and len(all_picks) > int(args.max_total):
        all_picks = rng.sample(all_picks, int(args.max_total))

    all_picks.sort(key=lambda x: (x.participant, x.video_path, x.video_frame, x.audit_type))

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []
    labels_rows: List[dict] = []
    saved = 0

    for idx, pick in enumerate(all_picks):
        frame, (w, h) = read_frame(pick.video_path, pick.video_frame)
        if frame is None:
            continue
        roi = clamp_roi(pick.roi, w, h)
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        folder_tag = _safe_token(Path(pick.video_path).parent.name, limit=24)
        img_name = f"{pick.participant}_{folder_tag}_{pick.segment_uid}_f{pick.video_frame:06d}_{pick.audit_type}_a{idx:06d}.jpg"
        img_path = img_dir / img_name
        ok_write = cv2.imwrite(str(img_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
        if not ok_write:
            continue

        row = {
            "img": f"images/{img_name}",
            "FrameID": str(pick.video_frame),
            "Timestamp": f"{pick.video_timestamp:.3f}",
            "Pred_Class": pick.pred_class,
            "Raw_Pitch": "",
            "Raw_Yaw": "",
            "Smooth_Pitch": "",
            "Smooth_Yaw": "",
            "Ref_Pitch": "",
            "Ref_Yaw": "",
            "Delta_Pitch": "",
            "Delta_Yaw": "",
            "Domain": pick.participant,
            "Video": pick.video_path,
            "ROI_X1": str(roi[0]),
            "ROI_Y1": str(roi[1]),
            "ROI_X2": str(roi[2]),
            "ROI_Y2": str(roi[3]),
            "Audit_Type": pick.audit_type,
            "Segment_UID": pick.segment_uid,
            "Confidence": pick.confidence,
        }
        manifest_rows.append(row)
        labels_rows.append(
            {
                "img": row["img"],
                "label": "",
                "FrameID": row["FrameID"],
                "Timestamp": row["Timestamp"],
                "Pred_Class": row["Pred_Class"],
                "Domain": row["Domain"],
                "Video": row["Video"],
            }
        )
        saved += 1

    manifest_path = out_dir / "manifest.csv"
    labels_path = out_dir / "labels.csv"
    readme_path = out_dir / "README.txt"

    manifest_fields = [
        "img", "FrameID", "Timestamp", "Pred_Class",
        "Raw_Pitch", "Raw_Yaw", "Smooth_Pitch", "Smooth_Yaw",
        "Ref_Pitch", "Ref_Yaw", "Delta_Pitch", "Delta_Yaw",
        "Domain", "Video", "ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2",
        "Audit_Type", "Segment_UID", "Confidence",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(manifest_rows)

    labels_fields = ["img", "label", "FrameID", "Timestamp", "Pred_Class", "Domain", "Video"]
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=labels_fields)
        w.writeheader()
        w.writerows(labels_rows)

    with readme_path.open("w", encoding="utf-8") as f:
        f.write("Gaze audit pack ready.\n")
        f.write("Purpose: manually spot-check current inference outputs.\n")
        f.write("Sampling types:\n")
        f.write("- uniform: temporal coverage over each segment\n")
        f.write("- transition: class switch locations\n")
        f.write("- lowconf: low-confidence predictions\n")
        f.write("Open with web_label_tool.py and write corrected labels into labels.csv.\n")

    print(f"plan_rows={len(plan_rows)}")
    print(f"sampled_images={saved}")
    print(f"out_dir={out_dir}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
