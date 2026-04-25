#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Align old labeled pack, current full review pack, and current audit pack.

This utility is mainly for debugging domain shifts and ROI changes between:
- old annotation packs (human labels),
- current full inference review packs,
- current sampled audit packs.
"""

from __future__ import annotations

import argparse
import csv
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Align old labels with current prediction and review entries")
    p.add_argument("--old-pack", required=True, help="Old labeled annotation pack directory")
    p.add_argument("--full-pack", required=True, help="Current full review pack directory")
    p.add_argument("--audit-pack", default="", help="Optional current audit pack directory")
    p.add_argument("--plan-csv", required=True, help="Current infer plan CSV")
    p.add_argument("--old-base-url", default="http://127.0.0.1:8014")
    p.add_argument("--full-base-url", default="http://127.0.0.1:8115")
    p.add_argument("--audit-base-url", default="http://127.0.0.1:8114")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--mismatch-out-csv", default="")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sort_manifest_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if rows and "SortKey" in rows[0]:
        return sorted(rows, key=lambda r: int(float(str(r.get("SortKey", "0")) or "0")))
    return sorted(rows, key=lambda r: (str(r.get("Domain", "")), int(float(str(r.get("FrameID", "0")) or "0"))))


def index_manifest(rows: List[Dict[str, str]]) -> Dict[str, int]:
    return {str(r["img"]): i for i, r in enumerate(rows)}


def build_video_frame_index(rows: List[Dict[str, str]]) -> Dict[str, List[Tuple[int, int, Dict[str, str]]]]:
    out: Dict[str, List[Tuple[int, int, Dict[str, str]]]] = {}
    for idx, r in enumerate(rows):
        video = str(r.get("Video", "")).strip()
        frame_id = int(float(str(r.get("FrameID", "0")) or "0"))
        if not video:
            continue
        out.setdefault(video, []).append((frame_id, idx, r))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def nearest_by_frame(items: List[Tuple[int, int, Dict[str, str]]], frame_id: int) -> Tuple[Optional[int], Optional[int], Optional[Dict[str, str]]]:
    if not items:
        return None, None, None
    frames = [x[0] for x in items]
    pos = bisect_left(frames, frame_id)
    cand = []
    if pos < len(items):
        cand.append(items[pos])
    if pos > 0:
        cand.append(items[pos - 1])
    best = min(cand, key=lambda x: abs(x[0] - frame_id))
    return best[1], abs(best[0] - frame_id), best[2]


def load_ranges(plan_rows: List[Dict[str, str]]) -> Dict[str, List[Tuple[float, float, str]]]:
    out: Dict[str, List[Tuple[float, float, str]]] = {}
    for r in plan_rows:
        video = str(r.get("video_path", "")).strip()
        seg = str(r.get("segment_uid", "")).strip()
        if not video or not seg:
            continue
        try:
            s = float(r["start_sec"])
            e = float(r["end_sec"])
        except Exception:
            continue
        out.setdefault(video, []).append((s, e, seg))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def locate_segment(ranges: List[Tuple[float, float, str]], ts: float) -> Tuple[str, str]:
    for s, e, seg in ranges:
        if s <= ts < e:
            return "1", seg
    return "0", ""


def main() -> None:
    args = parse_args()
    old_pack = Path(args.old_pack)
    full_pack = Path(args.full_pack)
    audit_pack = Path(args.audit_pack) if args.audit_pack else None
    plan_rows = read_csv(Path(args.plan_csv))

    old_manifest = sort_manifest_rows(read_csv(old_pack / "manifest.csv"))
    old_labels = {str(r["img"]): r for r in read_csv(old_pack / "labels.csv")}
    full_manifest = sort_manifest_rows(read_csv(full_pack / "manifest.csv"))
    audit_manifest = sort_manifest_rows(read_csv(audit_pack / "manifest.csv")) if audit_pack else []

    old_idx = index_manifest(old_manifest)
    full_video_idx = build_video_frame_index(full_manifest)
    audit_video_idx = build_video_frame_index(audit_manifest) if audit_manifest else {}
    ranges_by_video = load_ranges(plan_rows)

    out_rows: List[Dict[str, str]] = []
    mismatch_rows: List[Dict[str, str]] = []

    for row in old_manifest:
        img = str(row["img"])
        lab_row = old_labels.get(img, {})
        old_label = str(lab_row.get("label", "")).strip()
        video = str(row.get("Video", "")).strip()
        frame_id = int(float(str(row.get("FrameID", "0")) or "0"))
        ts = float(str(row.get("Timestamp", "0")) or "0")

        old_item = old_idx.get(img, -1)
        old_url = f"{args.old_base_url}/item/{old_item}" if old_item >= 0 else ""

        in_range, seg_uid = locate_segment(ranges_by_video.get(video, []), ts)

        full_item_idx, full_frame_diff, full_row = nearest_by_frame(full_video_idx.get(video, []), frame_id)
        audit_item_idx, audit_frame_diff, audit_row = nearest_by_frame(audit_video_idx.get(video, []), frame_id) if audit_pack else (None, None, None)

        current_pred = str(full_row.get("Pred_Class", "")).strip() if full_row else ""
        current_conf = str(full_row.get("Confidence", "")).strip() if full_row else ""
        full_url = f"{args.full_base_url}/item/{full_item_idx}" if full_item_idx is not None else ""
        audit_url = f"{args.audit_base_url}/item/{audit_item_idx}" if audit_item_idx is not None else ""

        out = {
            "video": video,
            "video_folder": Path(video).parent.name if video else "",
            "frame_id": str(frame_id),
            "timestamp_sec": f"{ts:.3f}",
            "old_label": old_label,
            "old_pack_img": img,
            "old_review_url": old_url,
            "in_current_infer_range": in_range,
            "current_segment_uid": seg_uid,
            "current_pred_class": current_pred,
            "current_pred_confidence": current_conf,
            "current_fullreview_url": full_url,
            "current_fullreview_frame_diff": "" if full_frame_diff is None else str(full_frame_diff),
            "current_audit_url": audit_url,
            "current_audit_frame_diff": "" if audit_frame_diff is None else str(audit_frame_diff),
        }
        out_rows.append(out)

        if old_label and current_pred and old_label != current_pred:
            mismatch_rows.append(out)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    mismatch_out = Path(args.mismatch_out_csv) if args.mismatch_out_csv else None
    if mismatch_out:
        with mismatch_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(mismatch_rows)

    print(f"rows={len(out_rows)}")
    print(f"mismatches={len(mismatch_rows)}")
    print(f"out_csv={out_csv}")
    if mismatch_out:
        print(f"mismatch_out_csv={mismatch_out}")


if __name__ == "__main__":
    main()
