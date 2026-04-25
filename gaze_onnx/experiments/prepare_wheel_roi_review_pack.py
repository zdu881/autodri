#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare one combined wheel ROI review pack across participants."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare combined wheel ROI review pack")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--grid-step", type=int, default=220)
    p.add_argument("--sample-position", choices=("first", "middle"), default="middle")
    return p.parse_args()


def pick_frame_index(total_frames: int, mode: str) -> int:
    if total_frames <= 1:
        return 0
    if mode == "middle":
        return max(0, (total_frames - 1) // 2)
    return 0


def draw_grid(img, step: int) -> None:
    h, w = img.shape[:2]
    step = max(40, int(step))
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), (255, 255, 255), 1)
        cv2.putText(img, str(x), (x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), (255, 255, 255), 1)
        cv2.putText(img, str(y), (4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 255, 255), 2)


def safe_id(text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = Path(text).stem
    stem = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    stem = stem[:48] if stem else "video"
    return f"{stem}__{h}"


def gather_rows(root: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    # p1 from infer plan
    p1_plan = root / "data/natural_driving_p1/analysis/p1_infer_plan.segment.csv"
    if p1_plan.exists():
        seen = set()
        for r in csv.DictReader(p1_plan.open("r", encoding="utf-8-sig", newline="")):
            video = str(r.get("video_path", "")).strip()
            if not video or video in seen:
                continue
            seen.add(video)
            out.append(
                {
                    "participant": "p1",
                    "video": video,
                    "roi_x1": r["wheel_roi_x1"],
                    "roi_y1": r["wheel_roi_y1"],
                    "roi_x2": r["wheel_roi_x2"],
                    "roi_y2": r["wheel_roi_y2"],
                    "roi_note": "from_p1_infer_plan_segment",
                }
            )

    for p in ["p2","p4","p6","p7","p8","p9","p10","p11","p13","p14","p15","p16","p17","p18"]:
        path = root / f"gaze_onnx/experiments/manifests/current/{p}_wheel_rois.current.csv"
        if not path.exists():
            continue
        for r in csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")):
            out.append(
                {
                    "participant": p,
                    "video": str(r["video"]).strip(),
                    "roi_x1": str(r["roi_x1"]).strip(),
                    "roi_y1": str(r["roi_y1"]).strip(),
                    "roi_x2": str(r["roi_x2"]).strip(),
                    "roi_y2": str(r["roi_y2"]).strip(),
                    "roi_note": str(r.get("inferred_rule", "")).strip(),
                }
            )
    return out


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    out_dir = Path(args.out_dir).resolve()
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    rows_in = gather_rows(root)
    rows_out: List[Dict[str, str]] = []

    for r in rows_in:
        video = Path(r["video"])
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[WARN] skip unreadable video: {video}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        idx = pick_frame_index(total, args.sample_position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"[WARN] skip unreadable frame: {video}")
            continue

        rel = f"{r['participant']}/{video.parent.name}/{video.name}"
        sid = safe_id(rel)
        raw_name = f"{sid}__raw.jpg"
        grid_name = f"{sid}__grid.jpg"
        raw_path = refs_dir / raw_name
        grid_path = refs_dir / grid_name

        raw = frame.copy()
        grid = frame.copy()
        draw_grid(grid, int(args.grid_step))
        cv2.imwrite(str(raw_path), raw)
        cv2.imwrite(str(grid_path), grid)

        rows_out.append(
            {
                "video_rel": rel,
                "video_abs": str(video),
                "ref_raw": str(raw_path.relative_to(out_dir)),
                "ref_grid": str(grid_path.relative_to(out_dir)),
                "frame_idx": idx,
                "timestamp_sec": f"{(idx / fps) if fps > 0 else 0.0:.3f}",
                "width": w,
                "height": h,
                "roi_x1": r["roi_x1"],
                "roi_y1": r["roi_y1"],
                "roi_x2": r["roi_x2"],
                "roi_y2": r["roi_y2"],
                "roi_note": r["roi_note"],
            }
        )

    manifest = out_dir / "roi_label_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows_out[0].keys()) if rows_out else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    readme = out_dir / "README.txt"
    with readme.open("w", encoding="utf-8") as f:
        f.write("Combined wheel ROI review pack.\n")
        f.write("The current wheel ROI is prefilled in roi_label_manifest.csv.\n")

    print(f"rows={len(rows_out)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
