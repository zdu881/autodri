#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a full manual review pack from segment-level gaze inference outputs.

Unlike build_gaze_audit_pack.py, this script keeps every frame from every
current gaze CSV. It is intended for exhaustive manual review when needed.

The output pack is lightweight on disk because web_label_tool.py can render
the ROI crop dynamically from the original video and frame index.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a full gaze review pack from current inference")
    p.add_argument("--plan-csv", required=True, help="Infer plan CSV")
    p.add_argument("--out-dir", required=True, help="Output review pack directory")
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


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan_csv)
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    with plan_path.open("r", encoding="utf-8-sig", newline="") as f:
        plan_rows = list(csv.DictReader(f))

    manifest_rows: List[dict] = []
    labels_rows: List[dict] = []
    row_idx = 0

    for pr in plan_rows:
        status = str(pr.get("status", "")).strip().lower()
        if status and status != "ok":
            continue
        segment_uid = str(pr.get("segment_uid", "")).strip()
        video_path = str(pr.get("video_path", "")).strip()
        gaze_csv = str(pr.get("gaze_csv", "")).strip()
        if not segment_uid or not video_path or not gaze_csv:
            continue

        gpath = Path(gaze_csv)
        if not gpath.is_absolute():
            gpath = Path.cwd() / gpath
        if not gpath.exists():
            continue

        participant = segment_uid.split("_seg_")[0] if "_seg_" in segment_uid else "review"
        roi_x1 = _to_int(pr.get("gaze_roi_x1", "0"))
        roi_y1 = _to_int(pr.get("gaze_roi_y1", "0"))
        roi_x2 = _to_int(pr.get("gaze_roi_x2", "0"))
        roi_y2 = _to_int(pr.get("gaze_roi_y2", "0"))

        with gpath.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        for r in rows:
            frame_id = _to_int(r.get("Video_FrameID", r.get("FrameID", "0")))
            ts = _to_float(r.get("Video_Timestamp", r.get("Timestamp", "0")))
            pred = str(r.get("Gaze_Class", "")).strip()
            conf = str(r.get("Confidence", "")).strip()
            img_name = f"dynamic/{participant}/{segment_uid}/f{frame_id:06d}_r{row_idx:07d}.jpg"
            manifest_rows.append(
                {
                    "img": img_name,
                    "FrameID": str(frame_id),
                    "Timestamp": f"{ts:.3f}",
                    "Pred_Class": pred,
                    "Raw_Pitch": "",
                    "Raw_Yaw": "",
                    "Smooth_Pitch": "",
                    "Smooth_Yaw": "",
                    "Ref_Pitch": "",
                    "Ref_Yaw": "",
                    "Delta_Pitch": "",
                    "Delta_Yaw": "",
                    "Domain": participant,
                    "Video": video_path,
                    "ROI_X1": str(roi_x1),
                    "ROI_Y1": str(roi_y1),
                    "ROI_X2": str(roi_x2),
                    "ROI_Y2": str(roi_y2),
                    "Segment_UID": segment_uid,
                    "Confidence": conf,
                    "SortKey": str(row_idx),
                }
            )
            labels_rows.append(
                {
                    "img": img_name,
                    "label": "",
                    "FrameID": str(frame_id),
                    "Timestamp": f"{ts:.3f}",
                    "Pred_Class": pred,
                    "Domain": participant,
                    "Video": video_path,
                }
            )
            row_idx += 1

    manifest_fields = [
        "img", "FrameID", "Timestamp", "Pred_Class",
        "Raw_Pitch", "Raw_Yaw", "Smooth_Pitch", "Smooth_Yaw",
        "Ref_Pitch", "Ref_Yaw", "Delta_Pitch", "Delta_Yaw",
        "Domain", "Video", "ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2",
        "Segment_UID", "Confidence", "SortKey",
    ]
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(manifest_rows)

    label_fields = ["img", "label", "FrameID", "Timestamp", "Pred_Class", "Domain", "Video"]
    with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=label_fields)
        w.writeheader()
        w.writerows(labels_rows)

    with (out_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write("Full gaze review pack ready.\n")
        f.write("This pack contains every frame from the current gaze inference outputs.\n")
        f.write("Images are rendered dynamically from the original videos by web_label_tool.py.\n")
        f.write("Pred_Class and Confidence are shown in the review UI.\n")

    print(f"plan_rows={len(plan_rows)}")
    print(f"manifest_rows={len(manifest_rows)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
