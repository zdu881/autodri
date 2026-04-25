#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run batch gaze inference from a domains CSV.

Expected input CSV columns:
- domain_id
- video
- roi_x1, roi_y1, roi_x2, roi_y2

This is the full-video counterpart of the older p1-specific segment runner.
It is intended for participant-level batch inference after ROI assignment has
already been normalized.
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from autodri.common.paths import resolve_existing_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run batch gaze inference from domains CSV")
    p.add_argument("--domains-csv", required=True, help="CSV with video + ROI columns")
    p.add_argument("--out-dir", required=True, help="Output directory for per-video CSV/event files")
    p.add_argument("--python-bin", default=sys.executable, help="Python executable to run subcommands")
    p.add_argument("--gaze-script", default="gaze_onnx/gaze_state_cls.py")
    p.add_argument("--aggregate-script", default="gaze_onnx/experiments/aggregate_gaze_windows.py")
    p.add_argument(
        "--cls-model",
        default="",
        help="Optional single classifier model to force for all rows. If empty, choose by participant.",
    )
    p.add_argument(
        "--cls-model-base",
        default="models/gaze_cls_yolov8n.onnx",
        help="Base classifier model for participants without few-shot fine-tuned weights.",
    )
    p.add_argument(
        "--cls-model-ft",
        default="models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx",
        help="Few-shot fine-tuned classifier model used for selected participants.",
    )
    p.add_argument(
        "--ft-participants",
        default="p1,p2,p3,p6,p7",
        help="Comma-separated participants that should use --cls-model-ft when --cls-model is not forced.",
    )
    p.add_argument("--scrfd-model", default="models/scrfd_person_2.5g.onnx")
    p.add_argument("--window-sec", type=float, default=20.0, help="Window size for event aggregation")
    p.add_argument("--skip-existing", action="store_true", help="Skip video if frame CSV already exists")
    p.add_argument("--no-video", action="store_true", help="Disable mp4 output to save time and storage")
    p.add_argument("--limit", type=int, default=0, help="Only run the first N rows")
    return p.parse_args()


def safe_slug(text: str) -> str:
    t = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))
    t = t.strip("._-")
    return t or "v"


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"video", "roi_x1", "roi_y1", "roi_x2", "roi_y2"}
    miss = required - set(rows[0].keys() if rows else [])
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")
    return rows


def infer_participant(row: Dict[str, str]) -> str:
    domain_id = str(row.get("domain_id", "")).strip().lower()
    m = re.search(r"\bp\d+\b", domain_id)
    if m:
        return m.group(0)

    video = str(row.get("video", "")).strip()
    for part in Path(video).parts:
        t = str(part).strip().lower()
        if re.fullmatch(r"p\d+", t):
            return t
    m = re.search(r"(^|/)(p\d+)(/|$)", Path(video).as_posix().lower())
    if m:
        return m.group(2)
    return ""


def resolve_cls_model(row: Dict[str, str], args: argparse.Namespace, ft_participants: set[str]) -> str:
    forced = str(args.cls_model or "").strip()
    if forced:
        return forced

    participant = infer_participant(row)
    if participant and participant in ft_participants:
        return str(args.cls_model_ft)
    return str(args.cls_model_base)


def run_cmd(cmd: List[str]) -> int:
    print("$ " + " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def main() -> None:
    args = parse_args()
    args.scrfd_model = str(
        resolve_existing_path(
            "" if args.scrfd_model == "models/scrfd_person_2.5g.onnx" else args.scrfd_model,
            workspace_rel="models/scrfd_person_2.5g.onnx",
            legacy_rels=("models/scrfd_person_2.5g.onnx",),
            description="SCRFD model",
        )
    )
    if args.cls_model_base == "models/gaze_cls_yolov8n.onnx":
        args.cls_model_base = str(
            resolve_existing_path(
                "",
                workspace_rel="models/gaze_cls_yolov8n.onnx",
                legacy_rels=("models/gaze_cls_yolov8n.onnx",),
                description="base gaze classifier model",
            )
        )
    if args.cls_model_ft == "models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx":
        args.cls_model_ft = str(
            resolve_existing_path(
                "",
                workspace_rel="models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx",
                legacy_rels=("models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx",),
                description="few-shot gaze classifier model",
            )
        )
    rows = load_rows(Path(args.domains_csv))
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    ft_participants = {
        x.strip().lower()
        for x in str(args.ft_participants or "").split(",")
        if x.strip()
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0
    skip = 0

    for i, row in enumerate(rows, start=1):
        video = str(row.get("video", "")).strip()
        if not video:
            print(f"[{i}/{len(rows)}] skip missing video")
            skip += 1
            continue
        if not Path(video).exists():
            print(f"[{i}/{len(rows)}] skip missing file: {video}")
            skip += 1
            continue

        domain_id = str(row.get("domain_id", "")).strip() or "domain"
        folder = Path(video).parent.name
        stem = Path(video).stem
        slug = f"{safe_slug(domain_id)}__{safe_slug(folder)}__{safe_slug(stem)}"
        frame_csv = out_dir / f"{slug}.gaze.csv"
        event_csv = out_dir / f"{slug}.event{int(round(float(args.window_sec)))}s.csv"
        out_video = out_dir / f"{slug}.gaze.mp4"

        if args.skip_existing and frame_csv.exists():
            print(f"[{i}/{len(rows)}] skip existing: {frame_csv}")
            skip += 1
            continue

        print(f"\n[{i}/{len(rows)}] {video}")
        cls_model = resolve_cls_model(row, args, ft_participants)
        print(f"  [model] participant={infer_participant(row) or 'unknown'} cls_model={cls_model}")

        gaze_cmd = [
            args.python_bin,
            args.gaze_script,
            "--video",
            video,
            "--roi",
            str(int(float(row["roi_x1"]))),
            str(int(float(row["roi_y1"]))),
            str(int(float(row["roi_x2"]))),
            str(int(float(row["roi_y2"]))),
            "--scrfd",
            args.scrfd_model,
            "--cls-model",
            cls_model,
            "--out-video",
            str(out_video),
            "--csv",
            str(frame_csv),
        ]
        if args.no_video:
            gaze_cmd.append("--no-video")

        rc = run_cmd(gaze_cmd)
        if rc != 0:
            fail += 1
            print(f"  [gaze] failed rc={rc}")
            continue

        agg_cmd = [
            args.python_bin,
            args.aggregate_script,
            "--csv",
            str(frame_csv),
            "--out-csv",
            str(event_csv),
            "--window-sec",
            f"{float(args.window_sec):.3f}",
        ]
        rc = run_cmd(agg_cmd)
        if rc != 0:
            fail += 1
            print(f"  [aggregate] failed rc={rc}")
            continue

        ok += 1

    print("\n=== Done ===")
    print(f"ok={ok} fail={fail} skip={skip} total={len(rows)}")


if __name__ == "__main__":
    main()
