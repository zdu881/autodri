#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare a dual gaze/wheel ROI review pack from current ROI manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Mapping

import cv2


PARTICIPANTS_DEFAULT = ("p8", "p9", "p10")
MANIFEST_FIELDS = [
    "participant",
    "video_rel",
    "video_abs",
    "ref_raw",
    "ref_grid",
    "frame_idx",
    "timestamp_sec",
    "width",
    "height",
    "gaze_roi_x1",
    "gaze_roi_y1",
    "gaze_roi_x2",
    "gaze_roi_y2",
    "wheel_roi_x1",
    "wheel_roi_y1",
    "wheel_roi_x2",
    "wheel_roi_y2",
    "gaze_review_status",
    "wheel_review_status",
    "roi_note",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_workspace_root() -> Path:
    root = repo_root()
    return root.parent / f"{root.name}_workspace"


def default_current_dir() -> Path:
    return default_workspace_root() / "artifacts" / "manifests" / "current"


def default_out_dir() -> Path:
    return (
        default_workspace_root()
        / "archive"
        / "gaze_onnx_experiments"
        / "roi_refs"
        / "missing_roi_review_p8_p9_p10_bbox"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a dual ROI review pack")
    p.add_argument(
        "--participants",
        nargs="+",
        default=list(PARTICIPANTS_DEFAULT),
        help="Participants to include, e.g. --participants p8 p9 p10",
    )
    p.add_argument("--current-dir", default=str(default_current_dir()))
    p.add_argument("--out-dir", default=str(default_out_dir()))
    p.add_argument("--grid-step", type=int, default=220)
    p.add_argument("--sample-position", choices=("first", "middle"), default="middle")
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for quick smoke packs. 0 means all rows.",
    )
    return p.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv_rows(path: Path, rows: list[Mapping[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def text_int(value: object) -> str:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return ""
    return str(int(float(raw)))


def note_for_pair(gaze: Mapping[str, str], wheel: Mapping[str, str]) -> str:
    parts: list[str] = []
    gaze_uncertain = str(gaze.get("source_uncertain", "")).strip()
    wheel_uncertain = str(wheel.get("source_uncertain", "")).strip()
    wheel_rule = str(wheel.get("inferred_rule", "")).strip()
    gaze_note = str(gaze.get("roi_note", "")).strip()
    wheel_note = str(wheel.get("roi_note", "")).strip()
    if gaze_uncertain:
        parts.append(f"gaze_uncertain={gaze_uncertain}")
    if wheel_uncertain:
        parts.append(f"wheel_uncertain={wheel_uncertain}")
    if wheel_rule:
        parts.append(f"wheel_rule={wheel_rule}")
    if gaze_note:
        parts.append(f"gaze_note={gaze_note}")
    if wheel_note:
        parts.append(f"wheel_note={wheel_note}")
    return "; ".join(parts)


def gather_dual_roi_rows(current_dir: Path | str, participants: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    current = Path(current_dir)
    out: list[dict[str, str]] = []
    for participant in participants:
        participant = str(participant).strip()
        if not participant:
            continue
        gaze_rows = read_csv_rows(current / f"{participant}_gaze_rois.current.csv")
        wheel_rows = read_csv_rows(current / f"{participant}_wheel_rois.current.csv")
        wheel_by_video = {str(row.get("video", "")).strip(): row for row in wheel_rows}
        for gaze in gaze_rows:
            video = str(gaze.get("video", "")).strip()
            if not video or video not in wheel_by_video:
                continue
            wheel = wheel_by_video[video]
            out.append(
                {
                    "participant": str(gaze.get("domain_id", "") or wheel.get("domain_id", "") or participant),
                    "video_abs": video,
                    "gaze_roi_x1": text_int(gaze.get("roi_x1", "")),
                    "gaze_roi_y1": text_int(gaze.get("roi_y1", "")),
                    "gaze_roi_x2": text_int(gaze.get("roi_x2", "")),
                    "gaze_roi_y2": text_int(gaze.get("roi_y2", "")),
                    "wheel_roi_x1": text_int(wheel.get("roi_x1", "")),
                    "wheel_roi_y1": text_int(wheel.get("roi_y1", "")),
                    "wheel_roi_x2": text_int(wheel.get("roi_x2", "")),
                    "wheel_roi_y2": text_int(wheel.get("roi_y2", "")),
                    "roi_note": note_for_pair(gaze, wheel),
                }
            )
    return out


def safe_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = Path(text).stem
    stem = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    stem = stem[:48] if stem else "video"
    return f"{stem}__{digest}"


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


def video_rel(participant: str, video: Path) -> str:
    return f"{participant}/{video.parent.name}/{video.name}"


def build_manifest_rows(
    rows: list[Mapping[str, str]],
    out_dir: Path,
    *,
    grid_step: int,
    sample_position: str,
    max_rows: int = 0,
) -> list[dict[str, str]]:
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, str]] = []
    picked = rows[: max_rows or None]
    for row in picked:
        participant = str(row.get("participant", "")).strip()
        video = Path(str(row.get("video_abs", "")).strip())
        if not video.exists():
            print(f"[WARN] skip missing video: {video}")
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[WARN] skip unreadable video: {video}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_idx = pick_frame_index(total, sample_position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"[WARN] skip unreadable frame: {video}")
            continue

        rel = video_rel(participant, video)
        sid = safe_id(rel)
        raw_path = refs_dir / f"{sid}__raw.jpg"
        grid_path = refs_dir / f"{sid}__grid.jpg"
        grid = frame.copy()
        draw_grid(grid, grid_step)
        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(grid_path), grid)

        out.append(
            {
                "participant": participant,
                "video_rel": rel,
                "video_abs": str(video),
                "ref_raw": str(raw_path.relative_to(out_dir)),
                "ref_grid": str(grid_path.relative_to(out_dir)),
                "frame_idx": str(frame_idx),
                "timestamp_sec": f"{(frame_idx / fps) if fps > 0 else 0.0:.3f}",
                "width": str(width),
                "height": str(height),
                "gaze_roi_x1": str(row.get("gaze_roi_x1", "")),
                "gaze_roi_y1": str(row.get("gaze_roi_y1", "")),
                "gaze_roi_x2": str(row.get("gaze_roi_x2", "")),
                "gaze_roi_y2": str(row.get("gaze_roi_y2", "")),
                "wheel_roi_x1": str(row.get("wheel_roi_x1", "")),
                "wheel_roi_y1": str(row.get("wheel_roi_y1", "")),
                "wheel_roi_x2": str(row.get("wheel_roi_x2", "")),
                "wheel_roi_y2": str(row.get("wheel_roi_y2", "")),
                "gaze_review_status": "pending",
                "wheel_review_status": "pending",
                "roi_note": str(row.get("roi_note", "")),
            }
        )
    return out


def write_readme(path: Path, participants: list[str], rows: int) -> None:
    path.write_text(
        "Dual ROI review pack.\n"
        f"Participants: {', '.join(participants)}\n"
        f"Rows: {rows}\n"
        "Use serve_dual_roi_bbox_review.py to mark original ROI correctness or draw corrected boxes.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    participants = [str(p).strip() for p in args.participants if str(p).strip()]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    current_rows = gather_dual_roi_rows(Path(args.current_dir), participants)
    manifest_rows = build_manifest_rows(
        current_rows,
        out_dir,
        grid_step=int(args.grid_step),
        sample_position=str(args.sample_position),
        max_rows=int(args.max_rows),
    )
    write_csv_rows(out_dir / "roi_label_manifest.csv", manifest_rows, MANIFEST_FIELDS)
    write_readme(out_dir / "README.txt", participants, len(manifest_rows))
    print(f"rows={len(manifest_rows)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
