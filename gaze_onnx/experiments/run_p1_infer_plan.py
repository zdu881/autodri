#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run gaze + wheel inference from p1 segment plan CSV."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run p1 segment inference from plan CSV")
    p.add_argument("--plan-csv", required=True)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--run-gaze", action="store_true")
    p.add_argument("--run-wheel", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Only run first N rows (0=all)")
    p.add_argument("--skip-existing", action="store_true", help="Skip task if csv already exists")
    p.add_argument("--no-video", action="store_true", help="Disable mp4 outputs in gaze/wheel scripts")

    p.add_argument("--gaze-script", default="gaze_onnx/gaze_state_cls.py")
    p.add_argument("--gaze-cls-model", default="models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx")
    p.add_argument("--gaze-scrfd", default="models/scrfd_person_2.5g.onnx")

    p.add_argument("--wheel-script", default="driver_monitor/hand_on_wheel.py")
    p.add_argument("--wheel-config", default="")
    p.add_argument("--wheel-weights", default="")
    p.add_argument("--wheel-device", default="")
    p.add_argument("--wheel-sample-fps", type=float, default=5.0)
    p.add_argument("--wheel-window-sec", type=float, default=0.0)
    p.add_argument(
        "--wheel-det-csv-dir",
        default="",
        help="Optional output dir for wheel detection CSVs (for YOLO dataset building).",
    )
    return p.parse_args()


def run_cmd(cmd: List[str], dry_run: bool) -> int:
    print("$ " + " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def as_int(r: Dict[str, str], k: str) -> int:
    return int(float(str(r.get(k, "0")).strip() or "0"))


def as_float(r: Dict[str, str], k: str) -> float:
    return float(str(r.get(k, "0")).strip() or "0")


def main() -> None:
    args = parse_args()
    if not args.run_gaze and not args.run_wheel:
        raise SystemExit("At least one of --run-gaze / --run-wheel is required")

    plan_path = Path(args.plan_csv)
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)

    with plan_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[: int(args.limit)]

    ok = 0
    fail = 0
    skip = 0

    wheel_weights = str(args.wheel_weights or "").strip()
    if not wheel_weights:
        for cand in [
            "models/groundingdino_swint_ogc.pth",
            "GroundingDINO/weights/groundingdino_swint_ogc.pth",
        ]:
            if Path(cand).exists():
                wheel_weights = cand
                break
    if args.run_wheel:
        if wheel_weights:
            print(f"[wheel] weights: {wheel_weights}")
        else:
            print("[wheel] WARN: no wheel weights path resolved; will rely on script defaults.")

    for i, r in enumerate(rows, start=1):
        status = str(r.get("status", "")).strip().lower()
        if status and status != "ok":
            print(f"[{i}/{len(rows)}] skip row status={status}")
            skip += 1
            continue

        video = str(r.get("video_path", "")).strip()
        if not video:
            print(f"[{i}/{len(rows)}] skip missing video_path")
            skip += 1
            continue
        if not Path(video).exists():
            print(f"[{i}/{len(rows)}] skip video not found: {video}")
            skip += 1
            continue

        start_sec = as_float(r, "start_sec")
        duration_sec = as_float(r, "duration_sec")
        print(f"\n[{i}/{len(rows)}] {r.get('segment_uid','')}  start={start_sec:.3f}s dur={duration_sec:.3f}s")

        row_failed = False
        if args.run_gaze:
            gaze_csv = str(r.get("gaze_csv", "")).strip()
            gaze_video = str(r.get("gaze_video", "")).strip()
            if args.skip_existing and gaze_csv and Path(gaze_csv).exists():
                print("  [gaze] skip existing csv")
            else:
                if gaze_csv:
                    os.makedirs(str(Path(gaze_csv).parent), exist_ok=True)
                if gaze_video and (not args.no_video):
                    os.makedirs(str(Path(gaze_video).parent), exist_ok=True)
                cmd = [
                    args.python_bin,
                    args.gaze_script,
                    "--video",
                    video,
                    "--start-sec",
                    f"{start_sec:.3f}",
                    "--duration-sec",
                    f"{duration_sec:.3f}",
                    "--roi",
                    str(as_int(r, "gaze_roi_x1")),
                    str(as_int(r, "gaze_roi_y1")),
                    str(as_int(r, "gaze_roi_x2")),
                    str(as_int(r, "gaze_roi_y2")),
                    "--scrfd",
                    args.gaze_scrfd,
                    "--cls-model",
                    args.gaze_cls_model,
                    "--out-video",
                    gaze_video,
                    "--csv",
                    gaze_csv,
                ]
                if args.no_video:
                    cmd.append("--no-video")
                rc = run_cmd(cmd, dry_run=bool(args.dry_run))
                if rc != 0:
                    row_failed = True
                    print(f"  [gaze] failed rc={rc}")

        if (not row_failed) and args.run_wheel:
            wheel_csv = str(r.get("wheel_csv", "")).strip()
            wheel_video = str(r.get("wheel_video", "")).strip()
            segment_uid = str(r.get("segment_uid", "")).strip() or f"row_{i:05d}"
            wheel_det_csv = ""
            if args.wheel_det_csv_dir:
                det_dir = Path(args.wheel_det_csv_dir)
                wheel_det_csv = str((det_dir / f"{segment_uid}.wheel.det.csv").as_posix())
            wheel_state_exists = bool(wheel_csv) and Path(wheel_csv).exists()
            wheel_det_exists = (not wheel_det_csv) or Path(wheel_det_csv).exists()
            if args.skip_existing and wheel_state_exists and wheel_det_exists:
                print("  [wheel] skip existing csv/det-csv")
            else:
                if wheel_csv:
                    os.makedirs(str(Path(wheel_csv).parent), exist_ok=True)
                if wheel_video and (not args.no_video):
                    os.makedirs(str(Path(wheel_video).parent), exist_ok=True)
                if wheel_det_csv:
                    os.makedirs(str(Path(wheel_det_csv).parent), exist_ok=True)
                cmd = [
                    args.python_bin,
                    args.wheel_script,
                    "--video",
                    video,
                    "--start-sec",
                    f"{start_sec:.3f}",
                    "--duration-sec",
                    f"{duration_sec:.3f}",
                    "--roi",
                    str(as_int(r, "wheel_roi_x1")),
                    str(as_int(r, "wheel_roi_y1")),
                    str(as_int(r, "wheel_roi_x2")),
                    str(as_int(r, "wheel_roi_y2")),
                    "--output",
                    wheel_video,
                    "--state-csv",
                    wheel_csv,
                    "--sample-fps",
                    f"{float(args.wheel_sample_fps):.3f}",
                    "--decision-window-sec",
                    f"{float(args.wheel_window_sec):.3f}",
                ]
                if wheel_det_csv:
                    cmd.extend(["--det-csv", wheel_det_csv])
                if args.no_video:
                    cmd.append("--no-video")
                if args.wheel_config:
                    cmd.extend(["--config", args.wheel_config])
                if wheel_weights:
                    cmd.extend(["--weights", wheel_weights])
                if args.wheel_device:
                    cmd.extend(["--device", args.wheel_device])
                rc = run_cmd(cmd, dry_run=bool(args.dry_run))
                if rc != 0:
                    row_failed = True
                    print(f"  [wheel] failed rc={rc}")

        if row_failed:
            fail += 1
        else:
            ok += 1

    print("\n=== Done ===")
    print(f"ok={ok} fail={fail} skip={skip} total={len(rows)}")


if __name__ == "__main__":
    main()
