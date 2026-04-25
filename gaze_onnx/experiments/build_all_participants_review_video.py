#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build one manual review video covering all participants already used in the project.

The video contains:
- a 1 second title card per participant
- a short clip rendered with current gaze inference overlay

Selection strategy:
- choose the current gaze CSV with the longest contiguous valid-class run
- center a short clip on that run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
for _parent in (_HERE.parent, *_HERE.parents):
    if (_parent / "src" / "autodri").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from autodri.common.paths import data_root, models_root, participant_analysis_dir


FPS = 30
TITLE_SEC = 1.0


@dataclass
class ClipSpec:
    participant: str
    video_path: str
    segment_uid: str
    start_sec: float
    duration_sec: float
    roi: Tuple[int, int, int, int]
    model: str
    dominant_class: str
    run_frames: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a manual review video across all used participants")
    p.add_argument(
        "--out-dir",
        default="gaze_onnx/output/review_highlights/all_participants",
        help="Working output directory",
    )
    p.add_argument(
        "--out-video",
        default="gaze_onnx/output/review_highlights/all_participants_review_v1.gaze.mp4",
        help="Final merged review video",
    )
    p.add_argument("--clip-sec", type=float, default=6.0, help="Seconds per participant clip")
    p.add_argument("--title-sec", type=float, default=1.0, help="Seconds for title card")
    p.add_argument("--device-gpu", default="7", help="CUDA_VISIBLE_DEVICES value for clip rendering")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def participants_in_use(root: Path) -> List[str]:
    base = data_root() / "natural_driving"
    out = []
    for p in sorted(base.glob("p*"), key=lambda x: int(x.name[1:])):
        pid = p.name
        if (base / pid / "analysis" / f"{pid}_gaze_map.current.csv").exists():
            out.append(pid)
    if (data_root() / "natural_driving_p1/analysis/p1_gaze_map.segment.csv").exists():
        out = ["p1"] + out
    return out


def get_plan_path(root: Path, participant: str) -> Path:
    if participant == "p1":
        return participant_analysis_dir("p1") / "p1_infer_plan.segment.csv"
    return participant_analysis_dir(participant) / f"{participant}_infer_plan.current.csv"


def infer_model(root: Path, participant: str, summary_model: str) -> str:
    if participant == "p1":
        p = models_root() / "gaze_cls_p1_200shot_driveonly_ft_v1.onnx"
        return str(p)
    v2 = models_root() / f"gaze_cls_{participant}_audit120_driveonly_ft_v2.onnx"
    if v2.exists():
        return str(v2)
    v1 = models_root() / f"gaze_cls_{participant}_200shot_driveonly_ft_v1.onnx"
    if v1.exists():
        return str(v1)
    if summary_model:
        return str((root / summary_model) if not Path(summary_model).is_absolute() else Path(summary_model))
    return str(models_root() / "gaze_cls_yolov8n.onnx")


def best_run_from_gaze_csv(path: Path) -> Tuple[str, int, int]:
    rows = read_csv(path)
    best_cls = ""
    best_start = 0
    best_len = 0
    cur_cls = ""
    cur_start = 0
    cur_len = 0
    valid = {"Forward", "Non-Forward", "In-Car"}
    for i, r in enumerate(rows):
        g = str(r.get("Gaze_Class", "")).strip()
        if g not in valid:
            g = ""
        if g and g == cur_cls:
            cur_len += 1
        else:
            if cur_cls and cur_len > best_len:
                best_cls, best_start, best_len = cur_cls, cur_start, cur_len
            cur_cls = g
            cur_start = i
            cur_len = 1 if g else 0
    if cur_cls and cur_len > best_len:
        best_cls, best_start, best_len = cur_cls, cur_start, cur_len
    return best_cls, best_start, best_len


def choose_clip(root: Path, participant: str, clip_sec: float) -> Optional[ClipSpec]:
    plan_rows = read_csv(get_plan_path(root, participant))
    candidates: List[ClipSpec] = []
    for r in plan_rows:
        gv = str(r.get("gaze_csv", "")).strip()
        if not gv:
            continue
        gpath = (root / gv) if not gv.startswith("/") else Path(gv)
        if not gpath.exists():
            continue
        rows = read_csv(gpath)
        if not rows:
            continue
        run_cls, run_start_idx, run_len = best_run_from_gaze_csv(gpath)
        if not run_cls or run_len <= 0:
            continue
        clip_frames = int(round(clip_sec * FPS))
        center_idx = run_start_idx + run_len // 2
        start_idx = max(0, center_idx - clip_frames // 2)
        if start_idx >= len(rows):
            start_idx = max(0, len(rows) - clip_frames)
        start_sec = float(rows[start_idx].get("Video_Timestamp", rows[start_idx].get("Timestamp", "0")))
        summary_model = ""
        js = Path(str(gpath) + ".summary.json")
        if js.exists():
            try:
                d = json.loads(js.read_text(encoding="utf-8"))
                summary_model = str(d.get("model", "")).strip()
            except Exception:
                pass
        candidates.append(
            ClipSpec(
                participant=participant,
                video_path=str(r["video_path"]),
                segment_uid=str(r.get("segment_uid", participant)),
                start_sec=start_sec,
                duration_sec=clip_sec,
                roi=(
                    int(float(r.get("gaze_roi_x1", "0"))),
                    int(float(r.get("gaze_roi_y1", "0"))),
                    int(float(r.get("gaze_roi_x2", "0"))),
                    int(float(r.get("gaze_roi_y2", "0"))),
                ),
                model=infer_model(root, participant, summary_model),
                dominant_class=run_cls,
                run_frames=run_len,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x.run_frames, x.duration_sec), reverse=True)
    return candidates[0]


def write_title_card(path: Path, participant: str, dominant_class: str, segment_uid: str, clip_sec: float, fps: int) -> None:
    w, h = 1920, 1080
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n_frames = max(1, int(round(TITLE_SEC * fps)))
    for _ in range(n_frames):
        frame = np.full((h, w, 3), (242, 246, 250), dtype=np.uint8)
        cv2.rectangle(frame, (120, 140), (1800, 940), (255, 255, 255), -1)
        cv2.rectangle(frame, (120, 140), (1800, 940), (190, 205, 220), 3)
        cv2.putText(frame, f"{participant}", (180, 360), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (24, 33, 43), 6, cv2.LINE_AA)
        cv2.putText(frame, f"Representative clip for manual review", (180, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (74, 85, 104), 3, cv2.LINE_AA)
        cv2.putText(frame, f"dominant prediction: {dominant_class}", (180, 580), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40, 88, 138), 4, cv2.LINE_AA)
        cv2.putText(frame, f"segment: {segment_uid}", (180, 660), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (74, 85, 104), 3, cv2.LINE_AA)
        cv2.putText(frame, f"clip length: {clip_sec:.1f}s", (180, 730), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (74, 85, 104), 3, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    participants = participants_in_use(root)
    manifest: List[Dict[str, object]] = []
    concat_paths: List[Path] = []

    for participant in participants:
        clip = choose_clip(root, participant, clip_sec=float(args.clip_sec))
        if clip is None:
            continue

        title_path = out_dir / f"{participant}_title.mp4"
        clip_path = out_dir / f"{participant}_clip.gaze.mp4"
        csv_path = out_dir / f"{participant}_clip.gaze.csv"

        write_title_card(title_path, participant, clip.dominant_class, clip.segment_uid, clip.duration_sec, FPS)

        env = dict(**__import__("os").environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.device_gpu)
        subprocess.run(
            [
                sys.executable,
                "gaze_onnx/gaze_state_cls.py",
                "--video",
                clip.video_path,
                "--start-sec",
                f"{clip.start_sec:.3f}",
                "--duration-sec",
                f"{clip.duration_sec:.3f}",
                "--scrfd",
                str(models_root() / "scrfd_person_2.5g.onnx"),
                "--roi",
                str(clip.roi[0]),
                str(clip.roi[1]),
                str(clip.roi[2]),
                str(clip.roi[3]),
                "--face-priority",
                "right_to_left_track",
                "--cls-model",
                clip.model,
                "--out-video",
                str(clip_path),
                "--csv",
                str(csv_path),
            ],
            cwd=root,
            check=True,
            env=env,
        )

        manifest.append(
            {
                "participant": participant,
                "segment_uid": clip.segment_uid,
                "video_path": clip.video_path,
                "start_sec": clip.start_sec,
                "duration_sec": clip.duration_sec,
                "roi": list(clip.roi),
                "model": clip.model,
                "dominant_class": clip.dominant_class,
                "run_frames": clip.run_frames,
                "title_path": str(title_path),
                "clip_path": str(clip_path),
            }
        )
        concat_paths.extend([title_path, clip_path])

    manifest_path = out_dir / "all_participants_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    concat_txt = out_dir / "concat_list.txt"
    with concat_txt.open("w", encoding="utf-8") as f:
        for p in concat_paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")

    out_video = Path(args.out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-an",
            str(out_video),
        ],
        cwd=root,
        check=True,
    )

    print(f"participants={len(manifest)}")
    print(f"manifest={manifest_path}")
    print(f"out_video={out_video}")


if __name__ == "__main__":
    main()
