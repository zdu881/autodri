from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


STATES = ("OFF", "ON", "UNCERTAIN")
DEFAULT_WORKSPACE_ROOT = Path(os.environ.get("AUTODRI_WORKSPACE", "/data/home/sim6g/autodri_workspace"))


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    state_csv: Path
    det_csv: Path
    video_path: Path
    roi: tuple[int, int, int, int]


@dataclass(frozen=True)
class AgreementMetrics:
    accuracy: float
    correct: int
    total: int
    confusion: Counter[tuple[str, str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "correct": self.correct,
            "total": self.total,
            "confusion": {
                f"{teacher}->{pred}": int(count)
                for (teacher, pred), count in sorted(self.confusion.items())
            },
            "per_class": _per_class_metrics(self.confusion),
        }


@dataclass(frozen=True)
class CandidateFrame:
    split: str
    teacher_state: str
    segment: SegmentSpec
    row: Mapping[str, str]


def _per_class_metrics(confusion: Counter[tuple[str, str]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for state in STATES:
        tp = confusion[(state, state)]
        support = sum(confusion[(state, pred)] for pred in STATES)
        predicted = sum(confusion[(teacher, state)] for teacher in STATES)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[state] = {
            "support": int(support),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return out


def resolve_video_path(video_path: str | Path, *, workspace_root: Path = DEFAULT_WORKSPACE_ROOT) -> Path:
    raw = Path(str(video_path).strip()).expanduser()
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    workspace_candidate = Path(workspace_root).expanduser() / raw
    if workspace_candidate.exists():
        return workspace_candidate
    return workspace_candidate


def assign_block_split(
    *,
    segment: SegmentSpec,
    time_sec: float,
    val_ratio: float,
    block_sec: float,
) -> str:
    del segment
    if val_ratio <= 0:
        return "train"
    if val_ratio >= 1:
        return "val"
    block = int(math.floor(max(0.0, float(time_sec)) / max(1e-6, float(block_sec))))
    period = max(2, int(round(1.0 / float(val_ratio))))
    return "val" if block % period == 0 else "train"


def assign_block_split_with_offset(
    *,
    segment: SegmentSpec,
    time_sec: float,
    val_ratio: float,
    block_sec: float,
    offset: int = 0,
) -> str:
    del segment
    if val_ratio <= 0:
        return "train"
    if val_ratio >= 1:
        return "val"
    block = int(math.floor(max(0.0, float(time_sec)) / max(1e-6, float(block_sec))))
    period = max(2, int(round(1.0 / float(val_ratio))))
    return "val" if (block - int(offset)) % period == 0 else "train"


def assign_hash_split(
    *,
    segment: SegmentSpec,
    video_frame: int,
    val_ratio: float,
    seed: int,
) -> str:
    if val_ratio <= 0:
        return "train"
    if val_ratio >= 1:
        return "val"
    key = f"{int(seed)}:{segment.name}:{int(video_frame)}".encode("utf-8")
    bucket = int(hashlib.sha1(key).hexdigest()[:12], 16) / float(16**12)
    return "val" if bucket < float(val_ratio) else "train"


def compute_agreement(rows: Iterable[tuple[str, str]]) -> AgreementMetrics:
    total = 0
    correct = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for teacher_raw, pred_raw in rows:
        teacher = normalize_state(teacher_raw)
        pred = normalize_state(pred_raw)
        total += 1
        if teacher == pred:
            correct += 1
        confusion[(teacher, pred)] += 1
    accuracy = (correct / total) if total else 0.0
    return AgreementMetrics(accuracy=accuracy, correct=correct, total=total, confusion=confusion)


def normalize_state(value: object) -> str:
    state = str(value or "").strip().upper()
    if state in STATES:
        return state
    if state in {"1", "TRUE", "HAND_ON_WHEEL"}:
        return "ON"
    if state in {"0", "FALSE", "HAND_OFF_WHEEL"}:
        return "OFF"
    if state in {"-1", "UNKNOWN", "UNSURE", "UNCLEAR"}:
        return "UNCERTAIN"
    raise ValueError(f"Unsupported wheel state: {value!r}")


def scan_support(
    *,
    state_glob: str,
    det_root: Path | str,
    out_csv: Path | str,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
    min_hand_conf: float = 0.20,
    min_wheel_conf: float = 0.20,
    min_iou: float = 0.05,
    max_off_iou: float | None = None,
    min_state_margin_sec: float = 0.0,
) -> list[dict[str, object]]:
    """Pair state CSVs with teacher detections and summarize clean support."""

    del workspace_root
    det_index = _index_det_csvs(Path(det_root))
    rows: list[dict[str, object]] = []
    for state_raw in sorted(glob.glob(str(state_glob), recursive=True)):
        state_csv = Path(state_raw)
        segment = _segment_name_from_state_csv(state_csv)
        participant = _participant_from_segment_or_path(segment, state_csv)
        det_csv = _find_det_csv(segment=segment, participant=participant, det_index=det_index)
        if det_csv is None:
            continue
        counts = _eligible_counts(
            state_csv,
            min_hand_conf=min_hand_conf,
            min_wheel_conf=min_wheel_conf,
            min_iou=min_iou,
            max_off_iou=max_off_iou,
            min_state_margin_sec=min_state_margin_sec,
        )
        rows.append(
            {
                "participant": participant,
                "segment": segment,
                "state_csv": str(state_csv),
                "det_csv": str(det_csv),
                **counts,
            }
        )

    _write_manifest(Path(out_csv), rows)
    return rows


def build_balanced_dataset(
    *,
    pairs_csv: Path | str,
    out_dir: Path | str,
    train_per_state: int = 1000,
    val_per_state: int = 300,
    min_gap_sec: float = 0.5,
    heldout_mode: str = "segment",
    imgsz: int = 224,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
    seed: int = 3407,
    jpeg_quality: int = 90,
    min_hand_conf: float = 0.20,
    min_wheel_conf: float = 0.20,
    min_iou: float = 0.05,
    max_off_iou: float | None = None,
    min_state_margin_sec: float = 0.0,
) -> dict[str, object]:
    """Build an ON-enriched, class-balanced teacher-state crop dataset."""

    if heldout_mode not in {"segment", "time-block"}:
        raise ValueError(f"Unsupported heldout_mode: {heldout_mode!r}")
    pair_rows = _read_manifest(Path(pairs_csv))
    if not pair_rows:
        raise ValueError(f"No pair rows loaded from {pairs_csv}")

    specs = [_spec_from_pair(row, workspace_root=Path(workspace_root)) for row in pair_rows]
    eligible_by_segment: dict[str, dict[str, list[Mapping[str, str]]]] = {}
    for spec in specs:
        eligible_by_segment[spec.name] = _eligible_rows_by_state(
            spec.state_csv,
            min_gap_sec=min_gap_sec,
            min_hand_conf=min_hand_conf,
            min_wheel_conf=min_wheel_conf,
            min_iou=min_iou,
            max_off_iou=max_off_iou,
            min_state_margin_sec=min_state_margin_sec,
        )

    val_segments: set[str] = set()
    selected: list[CandidateFrame] = []
    if heldout_mode == "segment":
        val_segments = _choose_heldout_segments(
            specs,
            eligible_by_segment,
            per_state_target=max(0, int(val_per_state)),
            seed=int(seed),
        )
        for state in STATES:
            val_candidates = _round_robin_candidates(
                "val",
                state,
                [spec for spec in specs if spec.name in val_segments],
                eligible_by_segment,
                limit=max(0, int(val_per_state)),
            )
            train_candidates = _round_robin_candidates(
                "train",
                state,
                [spec for spec in specs if spec.name not in val_segments],
                eligible_by_segment,
                limit=max(0, int(train_per_state)),
            )
            selected.extend(val_candidates)
            selected.extend(train_candidates)
    elif heldout_mode == "time-block":
        split_rows = _split_eligible_rows_by_time_block(
            specs=specs,
            eligible_by_segment=eligible_by_segment,
            val_ratio=_val_ratio_from_targets(train_per_state=train_per_state, val_per_state=val_per_state),
            seed=int(seed),
        )
        for state in STATES:
            val_candidates = _round_robin_split_candidates(
                "val",
                state,
                specs,
                split_rows,
                limit=max(0, int(val_per_state)),
            )
            train_candidates = _round_robin_split_candidates(
                "train",
                state,
                specs,
                split_rows,
                limit=max(0, int(train_per_state)),
            )
            selected.extend(val_candidates)
            selected.extend(train_candidates)

    summary = _materialize_selected_frames(
        selected,
        out_dir=Path(out_dir),
        imgsz=imgsz,
        jpeg_quality=jpeg_quality,
        seed=seed,
        min_gap_sec=min_gap_sec,
        heldout_mode=heldout_mode,
        requested_train_per_state=train_per_state,
        requested_val_per_state=val_per_state,
        val_segments=val_segments,
        min_state_margin_sec=min_state_margin_sec,
    )
    return summary


def read_segment_spec(
    *,
    state_csv: Path | str,
    det_csv: Path | str,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> SegmentSpec:
    state_path = Path(state_csv).expanduser()
    det_path = Path(det_csv).expanduser()
    with det_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader)
    video = resolve_video_path(first["video_path"], workspace_root=workspace_root)
    roi = (
        int(float(first["roi_x1"])),
        int(float(first["roi_y1"])),
        int(float(first["roi_x2"])),
        int(float(first["roi_y2"])),
    )
    return SegmentSpec(
        name=_safe_name(state_path.stem.replace("tmp_wheel_state_gd_", "")),
        state_csv=state_path,
        det_csv=det_path,
        video_path=video,
        roi=roi,
    )


def _spec_from_pair(row: Mapping[str, str], *, workspace_root: Path) -> SegmentSpec:
    spec = read_segment_spec(
        state_csv=row["state_csv"],
        det_csv=row["det_csv"],
        workspace_root=workspace_root,
    )
    segment = str(row.get("segment", "")).strip() or spec.name
    return SegmentSpec(
        name=_safe_name(segment),
        state_csv=spec.state_csv,
        det_csv=spec.det_csv,
        video_path=spec.video_path,
        roi=spec.roi,
    )


def materialize_dataset(
    segments: Sequence[SegmentSpec],
    *,
    out_dir: Path | str,
    imgsz: int = 224,
    sample_stride: int = 1,
    val_ratio: float = 0.2,
    block_sec: float = 1.0,
    split_mode: str = "block",
    seed: int = 3407,
    jpeg_quality: int = 90,
) -> dict[str, object]:
    import cv2

    root = Path(out_dir).expanduser()
    for split in ("train", "val"):
        for state in STATES:
            (root / split / state).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    counts: Counter[tuple[str, str]] = Counter()
    image_size = max(16, int(imgsz))
    frame_step = max(1, int(sample_stride))

    for segment in segments:
        cap = cv2.VideoCapture(str(segment.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {segment.video_path}")
        try:
            with segment.state_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                state_rows = [
                    row
                    for idx, row in enumerate(reader)
                    if idx % frame_step == 0
                ]
            state_rows.sort(key=lambda row: int(float(row["video_frame"])))
            next_decode_frame: int | None = None
            for row in state_rows:
                state = normalize_state(row.get("stable_state", ""))
                video_frame = int(float(row["video_frame"]))
                if split_mode == "hash":
                    split = assign_hash_split(
                        segment=segment,
                        video_frame=video_frame,
                        val_ratio=val_ratio,
                        seed=seed,
                    )
                elif split_mode == "block":
                    split = assign_block_split(
                        segment=segment,
                        time_sec=float(row["time_sec"]),
                        val_ratio=val_ratio,
                        block_sec=block_sec,
                    )
                else:
                    raise ValueError(f"Unsupported split_mode: {split_mode!r}")
                if should_seek_decode(next_decode_frame=next_decode_frame, video_frame=video_frame):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
                    next_decode_frame = video_frame

                frame = None
                while next_decode_frame <= video_frame:
                    ok, decoded = cap.read()
                    if not ok or decoded is None:
                        raise RuntimeError(f"Failed to read {segment.video_path} frame={video_frame}")
                    frame = decoded
                    next_decode_frame += 1
                if frame is None:
                    raise RuntimeError(f"Failed to decode target frame={video_frame}")

                h, w = frame.shape[:2]
                x1, y1, x2, y2 = _clip_roi(segment.roi, width=w, height=h)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    raise RuntimeError(f"Empty crop for {segment.video_path} frame={video_frame}")
                resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)

                rel = Path(split) / state / f"{segment.name}__f{video_frame:07d}.jpg"
                dst = root / rel
                write_image(dst, resized, jpeg_quality=int(jpeg_quality))
                manifest_rows.append(
                    {
                        "split": split,
                        "teacher_state": state,
                        "image_path": str(dst),
                        "rel_path": rel.as_posix(),
                        "segment": segment.name,
                        "state_csv": str(segment.state_csv),
                        "det_csv": str(segment.det_csv),
                        "video_path": str(segment.video_path),
                        "frame": row.get("frame", ""),
                        "video_frame": video_frame,
                        "time_sec": row["time_sec"],
                        "video_time_sec": row["video_time_sec"],
                        "roi_x1": x1,
                        "roi_y1": y1,
                        "roi_x2": x2,
                        "roi_y2": y2,
                    }
                )
                counts[(split, state)] += 1
        finally:
            cap.release()

    _write_manifest(root / "manifest.csv", manifest_rows)
    summary = {
        "dataset_root": str(root),
        "image_size": image_size,
        "sample_stride": frame_step,
        "val_ratio": float(val_ratio),
        "block_sec": float(block_sec),
        "split_mode": split_mode,
        "seed": int(seed),
        "segments": [segment.name for segment in segments],
        "images_written": len(manifest_rows),
        "counts": {f"{split}/{state}": int(count) for (split, state), count in sorted(counts.items())},
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def predict_manifest(
    *,
    model_path: Path | str,
    manifest_csv: Path | str,
    out_csv: Path | str,
    split: str = "val",
    imgsz: int = 224,
    batch: int = 64,
    device: str = "cuda",
) -> AgreementMetrics:
    from ultralytics import YOLO

    wanted_split = str(split).strip()
    rows = _read_manifest(Path(manifest_csv))
    selected = [row for row in rows if not wanted_split or row["split"] == wanted_split]
    if not selected:
        raise ValueError(f"No manifest rows selected for split={wanted_split!r}")

    model = YOLO(str(model_path))
    image_paths = [row["image_path"] for row in selected]
    results = model.predict(
        source=image_paths,
        imgsz=int(imgsz),
        batch=int(batch),
        device=str(device),
        verbose=False,
    )

    pred_rows: list[dict[str, object]] = []
    agreement_pairs: list[tuple[str, str]] = []
    for row, result in zip(selected, results):
        if result.probs is None:
            raise RuntimeError(f"Classifier result without probabilities: {row['image_path']}")
        pred_idx = int(result.probs.top1)
        pred_name = str(result.names[pred_idx])
        pred_state = normalize_state(pred_name)
        teacher_state = normalize_state(row["teacher_state"])
        confidence = float(result.probs.top1conf.item())
        pred_rows.append(
            {
                **row,
                "pred_state": pred_state,
                "pred_conf": f"{confidence:.6f}",
                "match": int(teacher_state == pred_state),
            }
        )
        agreement_pairs.append((teacher_state, pred_state))

    _write_manifest(Path(out_csv), pred_rows)
    metrics = compute_agreement(agreement_pairs)
    Path(out_csv).with_suffix(".metrics.json").write_text(
        json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def _materialize_selected_frames(
    selected: Sequence[CandidateFrame],
    *,
    out_dir: Path,
    imgsz: int,
    jpeg_quality: int,
    seed: int,
    min_gap_sec: float,
    heldout_mode: str,
    requested_train_per_state: int,
    requested_val_per_state: int,
    val_segments: set[str],
    min_state_margin_sec: float,
) -> dict[str, object]:
    import cv2

    root = Path(out_dir).expanduser()
    for split in ("train", "val"):
        for state in STATES:
            (root / split / state).mkdir(parents=True, exist_ok=True)

    image_size = max(16, int(imgsz))
    manifest_rows: list[dict[str, object]] = []
    counts: Counter[tuple[str, str]] = Counter()
    selected_by_segment: dict[str, list[CandidateFrame]] = defaultdict(list)
    for candidate in selected:
        selected_by_segment[candidate.segment.name].append(candidate)

    for segment_name, candidates in sorted(selected_by_segment.items()):
        segment = candidates[0].segment
        cap = cv2.VideoCapture(str(segment.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {segment.video_path}")
        try:
            by_frame = sorted(
                candidates,
                key=lambda candidate: int(float(candidate.row["video_frame"])),
            )
            next_decode_frame: int | None = None
            for candidate in by_frame:
                row = candidate.row
                state = candidate.teacher_state
                split = candidate.split
                video_frame = int(float(row["video_frame"]))
                if should_seek_decode(next_decode_frame=next_decode_frame, video_frame=video_frame):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
                    next_decode_frame = video_frame
                frame = None
                while next_decode_frame <= video_frame:
                    ok, decoded = cap.read()
                    if not ok or decoded is None:
                        raise RuntimeError(f"Failed to read {segment.video_path} frame={video_frame}")
                    frame = decoded
                    next_decode_frame += 1
                if frame is None:
                    raise RuntimeError(f"Failed to decode target frame={video_frame}")
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = _clip_roi(segment.roi, width=w, height=h)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    raise RuntimeError(f"Empty crop for {segment.video_path} frame={video_frame}")
                resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)

                rel = Path(split) / state / f"{segment_name}__f{video_frame:07d}.jpg"
                dst = root / rel
                write_image(dst, resized, jpeg_quality=int(jpeg_quality))
                manifest_rows.append(
                    {
                        "split": split,
                        "teacher_state": state,
                        "image_path": str(dst),
                        "rel_path": rel.as_posix(),
                        "segment": segment_name,
                        "state_csv": str(segment.state_csv),
                        "det_csv": str(segment.det_csv),
                        "video_path": str(segment.video_path),
                        "frame": row.get("frame", ""),
                        "video_frame": video_frame,
                        "time_sec": row["time_sec"],
                        "video_time_sec": row["video_time_sec"],
                        "roi_x1": x1,
                        "roi_y1": y1,
                        "roi_x2": x2,
                        "roi_y2": y2,
                    }
                )
                counts[(split, state)] += 1
        finally:
            cap.release()

    _write_manifest(root / "manifest.csv", manifest_rows)
    summary = {
        "dataset_root": str(root),
        "image_size": image_size,
        "seed": int(seed),
        "min_gap_sec": float(min_gap_sec),
        "min_state_margin_sec": float(min_state_margin_sec),
        "heldout_mode": heldout_mode,
        "requested_train_per_state": int(requested_train_per_state),
        "requested_val_per_state": int(requested_val_per_state),
        "heldout_segments": sorted(val_segments),
        "images_written": len(manifest_rows),
        "counts": {f"{split}/{state}": int(count) for (split, state), count in sorted(counts.items())},
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _index_det_csvs(det_root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(Path(det_root).expanduser().glob("**/*.wheel.det.csv")):
        out[path.stem.replace(".wheel.det", "")].append(path)
    return out


def _find_det_csv(
    *,
    segment: str,
    participant: str,
    det_index: Mapping[str, Sequence[Path]],
) -> Path | None:
    if segment in det_index:
        paths = list(det_index[segment])
        participant_paths = [path for path in paths if participant and participant in path.parts]
        return (participant_paths or paths)[0]
    tail = re.search(r"(p\d+_seg_\d+)", segment)
    if tail and tail.group(1) in det_index:
        paths = list(det_index[tail.group(1)])
        participant_paths = [path for path in paths if participant and participant in path.parts]
        return (participant_paths or paths)[0]
    return None


def _segment_name_from_state_csv(path: Path) -> str:
    stem = path.stem.replace(".wheel", "")
    match = re.search(r"(p\d+_seg_\d+)", stem)
    if match:
        return match.group(1)
    match = re.search(r"(seg_\d+)", stem)
    return match.group(1) if match else _safe_name(stem)


def _participant_from_segment_or_path(segment: str, path: Path) -> str:
    match = re.search(r"(p\d+)", segment)
    if match:
        return match.group(1)
    for part in reversed(path.parts):
        if re.fullmatch(r"p\d+", part):
            return part
    return ""


def _eligible_counts(
    state_csv: Path,
    *,
    min_hand_conf: float,
    min_wheel_conf: float,
    min_iou: float,
    max_off_iou: float | None,
    min_state_margin_sec: float,
) -> dict[str, int]:
    raw_rows = 0
    clean = Counter()
    with state_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    margins = _state_transition_margins(rows)
    for idx, row in enumerate(rows):
        raw_rows += 1
        state = _eligible_state(
            row,
            min_hand_conf=min_hand_conf,
            min_wheel_conf=min_wheel_conf,
            min_iou=min_iou,
            max_off_iou=max_off_iou,
            state_margin_sec=margins[idx],
            min_state_margin_sec=min_state_margin_sec,
        )
        if state:
            clean[state] += 1
    return {
        "raw_rows": raw_rows,
        "clean_rows": sum(clean.values()),
        "clean_on": clean["ON"],
        "clean_off": clean["OFF"],
        "clean_uncertain": clean["UNCERTAIN"],
    }


def _eligible_rows_by_state(
    state_csv: Path,
    *,
    min_gap_sec: float,
    min_hand_conf: float,
    min_wheel_conf: float,
    min_iou: float,
    max_off_iou: float | None,
    min_state_margin_sec: float,
) -> dict[str, list[Mapping[str, str]]]:
    selected: dict[str, list[Mapping[str, str]]] = {state: [] for state in STATES}
    last_time: dict[str, float] = {state: -float("inf") for state in STATES}
    with state_csv.open("r", encoding="utf-8", newline="") as f:
        rows = sorted(
            csv.DictReader(f),
            key=lambda row: (float(row.get("time_sec", 0) or 0), int(float(row.get("video_frame", 0) or 0))),
        )
    margins = _state_transition_margins(rows)
    for idx, row in enumerate(rows):
        state = _eligible_state(
            row,
            min_hand_conf=min_hand_conf,
            min_wheel_conf=min_wheel_conf,
            min_iou=min_iou,
            max_off_iou=max_off_iou,
            state_margin_sec=margins[idx],
            min_state_margin_sec=min_state_margin_sec,
        )
        if not state:
            continue
        time_sec = float(row.get("time_sec", 0) or 0)
        if time_sec - last_time[state] < float(min_gap_sec):
            continue
        selected[state].append(dict(row))
        last_time[state] = time_sec
    return selected


def _eligible_state(
    row: Mapping[str, str],
    *,
    min_hand_conf: float,
    min_wheel_conf: float,
    min_iou: float,
    max_off_iou: float | None = None,
    state_margin_sec: float | None = None,
    min_state_margin_sec: float = 0.0,
) -> str | None:
    try:
        stable = normalize_state(row.get("stable_state", ""))
        raw = normalize_state(row.get("raw_state", row.get("stable_state", "")))
    except ValueError:
        return None
    if stable != raw:
        return None
    if min_state_margin_sec > 0 and state_margin_sec is not None:
        if state_margin_sec < float(min_state_margin_sec):
            return None
    if stable == "ON":
        if _float(row.get("max_hand_conf", 0)) < min_hand_conf:
            return None
        if _float(row.get("max_wheel_conf", 0)) < min_wheel_conf:
            return None
        if _float(row.get("iou_max", 0)) < min_iou:
            return None
    if stable == "OFF" and max_off_iou is not None:
        if _float(row.get("iou_max", 0)) > float(max_off_iou):
            return None
    return stable


def _state_transition_margins(rows: Sequence[Mapping[str, str]]) -> list[float]:
    """Distance in seconds from each row to the nearest stable-state transition."""

    if not rows:
        return []
    times = [_float(row.get("time_sec", 0)) for row in rows]
    transitions: list[float] = []
    previous: str | None = None
    for idx, row in enumerate(rows):
        stable = str(row.get("stable_state", "")).strip().upper()
        if previous is not None and stable != previous:
            transitions.append(times[idx])
        previous = stable
    if not transitions:
        return [float("inf")] * len(rows)
    margins: list[float] = []
    transition_idx = 0
    for time_sec in times:
        while transition_idx + 1 < len(transitions) and transitions[transition_idx + 1] <= time_sec:
            transition_idx += 1
        best = abs(time_sec - transitions[transition_idx])
        if transition_idx + 1 < len(transitions):
            best = min(best, abs(time_sec - transitions[transition_idx + 1]))
        margins.append(best)
    return margins


def _choose_heldout_segments(
    specs: Sequence[SegmentSpec],
    eligible_by_segment: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
    *,
    per_state_target: int,
    seed: int,
) -> set[str]:
    del seed
    needed = {state: max(0, int(per_state_target)) for state in STATES}
    chosen: set[str] = set()
    if not any(needed.values()):
        return chosen
    scored = []
    for spec in specs:
        counts = {
            state: len(eligible_by_segment[spec.name][state])
            for state in STATES
        }
        score = sum(min(counts[state], needed[state]) for state in STATES)
        scored.append((score, sum(counts.values()), spec.name))
    min_segments = min(len(specs), 2)
    for _, _, name in sorted(scored, reverse=True):
        if all(needed[state] <= 0 for state in STATES):
            if len(chosen) >= min_segments:
                break
        chosen.add(name)
        for state in STATES:
            needed[state] -= len(eligible_by_segment[name][state])
    if not chosen and specs:
        chosen.add(specs[-1].name)
    return chosen


def _round_robin_candidates(
    split: str,
    state: str,
    specs: Sequence[SegmentSpec],
    eligible_by_segment: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
    *,
    limit: int,
) -> list[CandidateFrame]:
    if limit <= 0:
        return []
    by_segment = [
        (spec, list(eligible_by_segment[spec.name][state]))
        for spec in specs
        if eligible_by_segment[spec.name][state]
    ]
    out: list[CandidateFrame] = []
    idx = 0
    while len(out) < limit and by_segment:
        next_round = []
        for spec, rows in by_segment:
            if idx < len(rows):
                out.append(CandidateFrame(split, state, spec, rows[idx]))
                next_round.append((spec, rows))
                if len(out) >= limit:
                    break
        idx += 1
        by_segment = next_round
    return out


def _val_ratio_from_targets(*, train_per_state: int, val_per_state: int) -> float:
    total = max(0, int(train_per_state)) + max(0, int(val_per_state))
    if total <= 0:
        return 0.0
    return max(0.0, min(0.25, int(val_per_state) / total))


def _split_eligible_rows_by_time_block(
    *,
    specs: Sequence[SegmentSpec],
    eligible_by_segment: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
    val_ratio: float,
    block_sec: float = 1.0,
    seed: int = 3407,
) -> dict[str, dict[str, dict[str, list[Mapping[str, str]]]]]:
    out: dict[str, dict[str, dict[str, list[Mapping[str, str]]]]] = {}
    for spec in specs:
        offset = _time_block_offset(segment=spec, val_ratio=val_ratio, seed=seed)
        out[spec.name] = {
            state: {"train": [], "val": []}
            for state in STATES
        }
        for state in STATES:
            for row in eligible_by_segment[spec.name][state]:
                split = assign_block_split_with_offset(
                    segment=spec,
                    time_sec=float(row.get("time_sec", 0) or 0),
                    val_ratio=val_ratio,
                    block_sec=block_sec,
                    offset=offset,
                )
                out[spec.name][state][split].append(row)
    return out


def _time_block_offset(*, segment: SegmentSpec, val_ratio: float, seed: int) -> int:
    if val_ratio <= 0 or val_ratio >= 1:
        return 0
    period = max(2, int(round(1.0 / float(val_ratio))))
    key = f"{int(seed)}:{segment.name}".encode("utf-8")
    return int(hashlib.sha1(key).hexdigest()[:8], 16) % period


def _round_robin_split_candidates(
    split: str,
    state: str,
    specs: Sequence[SegmentSpec],
    split_rows: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]]],
    *,
    limit: int,
) -> list[CandidateFrame]:
    if limit <= 0:
        return []
    by_segment = [
        (spec, list(split_rows[spec.name][state][split]))
        for spec in specs
        if split_rows[spec.name][state][split]
    ]
    out: list[CandidateFrame] = []
    idx = 0
    while len(out) < limit and by_segment:
        next_round = []
        for spec, rows in by_segment:
            if idx < len(rows):
                out.append(CandidateFrame(split, state, spec, rows[idx]))
                next_round.append((spec, rows))
                if len(out) >= limit:
                    break
        idx += 1
        by_segment = next_round
    return out


def _float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _clip_roi(roi: tuple[int, int, int, int], *, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(1, min(int(x2), width))
    y2 = max(1, min(int(y2), height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI after clipping: {roi}")
    return x1, y1, x2, y2


def write_image(path: Path | str, image: object, *, jpeg_quality: int = 90) -> None:
    import cv2

    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {dst}")


def should_seek_decode(
    *,
    next_decode_frame: int | None,
    video_frame: int,
    max_gap_frames: int = 300,
) -> bool:
    if next_decode_frame is None:
        return True
    if int(video_frame) < int(next_decode_frame):
        return True
    return int(video_frame) - int(next_decode_frame) > int(max_gap_frames)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "segment"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate GroundingDINO wheel-state distillation datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Materialize teacher-state ROI crops for YOLO classification.")
    build.add_argument("--state-csv", action="append", required=True)
    build.add_argument("--det-csv", action="append", required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    build.add_argument("--imgsz", type=int, default=224)
    build.add_argument("--sample-stride", type=int, default=1)
    build.add_argument("--val-ratio", type=float, default=0.2)
    build.add_argument("--block-sec", type=float, default=1.0)
    build.add_argument("--split-mode", choices=["block", "hash"], default="block")
    build.add_argument("--seed", type=int, default=3407)
    build.add_argument("--jpeg-quality", type=int, default=90)
    build.set_defaults(func=_cmd_build)

    predict = sub.add_parser("predict", help="Predict a manifest split and compute teacher agreement.")
    predict.add_argument("--model", required=True)
    predict.add_argument("--manifest", required=True)
    predict.add_argument("--out-csv", required=True)
    predict.add_argument("--split", default="val")
    predict.add_argument("--imgsz", type=int, default=224)
    predict.add_argument("--batch", type=int, default=64)
    predict.add_argument("--device", default="cuda")
    predict.set_defaults(func=_cmd_predict)

    scan = sub.add_parser("scan-support", help="Scan paired teacher state/detection CSVs for clean ON/OFF/UNCERTAIN support.")
    scan.add_argument("--state-glob", required=True)
    scan.add_argument("--det-root", required=True)
    scan.add_argument("--out-csv", required=True)
    scan.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    scan.add_argument("--min-hand-conf", type=float, default=0.20)
    scan.add_argument("--min-wheel-conf", type=float, default=0.20)
    scan.add_argument("--min-iou", type=float, default=0.05)
    scan.add_argument("--max-off-iou", type=float, default=None)
    scan.add_argument("--min-state-margin-sec", type=float, default=0.0)
    scan.set_defaults(func=_cmd_scan_support)

    balanced = sub.add_parser("build-balanced", help="Build an ON-enriched balanced heldout distillation dataset.")
    balanced.add_argument("--pairs-csv", required=True)
    balanced.add_argument("--out-dir", required=True)
    balanced.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    balanced.add_argument("--min-gap-sec", type=float, default=0.5)
    balanced.add_argument("--train-per-state", type=int, default=1000)
    balanced.add_argument("--val-per-state", type=int, default=300)
    balanced.add_argument("--heldout-mode", choices=["segment", "time-block"], default="segment")
    balanced.add_argument("--imgsz", type=int, default=224)
    balanced.add_argument("--seed", type=int, default=3407)
    balanced.add_argument("--jpeg-quality", type=int, default=90)
    balanced.add_argument("--min-hand-conf", type=float, default=0.20)
    balanced.add_argument("--min-wheel-conf", type=float, default=0.20)
    balanced.add_argument("--min-iou", type=float, default=0.05)
    balanced.add_argument("--max-off-iou", type=float, default=None)
    balanced.add_argument("--min-state-margin-sec", type=float, default=0.0)
    balanced.set_defaults(func=_cmd_build_balanced)

    return parser.parse_args(argv)


def _cmd_build(args: argparse.Namespace) -> None:
    if len(args.state_csv) != len(args.det_csv):
        raise SystemExit("--state-csv and --det-csv must be passed the same number of times.")
    workspace_root = Path(args.workspace_root).expanduser()
    segments = [
        read_segment_spec(state_csv=state_csv, det_csv=det_csv, workspace_root=workspace_root)
        for state_csv, det_csv in zip(args.state_csv, args.det_csv)
    ]
    summary = materialize_dataset(
        segments,
        out_dir=args.out_dir,
        imgsz=args.imgsz,
        sample_stride=args.sample_stride,
        val_ratio=args.val_ratio,
        block_sec=args.block_sec,
        split_mode=args.split_mode,
        seed=args.seed,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _cmd_predict(args: argparse.Namespace) -> None:
    metrics = predict_manifest(
        model_path=args.model,
        manifest_csv=args.manifest,
        out_csv=args.out_csv,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))


def _cmd_scan_support(args: argparse.Namespace) -> None:
    rows = scan_support(
        state_glob=args.state_glob,
        det_root=args.det_root,
        out_csv=args.out_csv,
        workspace_root=Path(args.workspace_root).expanduser(),
        min_hand_conf=args.min_hand_conf,
        min_wheel_conf=args.min_wheel_conf,
        min_iou=args.min_iou,
        max_off_iou=args.max_off_iou,
        min_state_margin_sec=args.min_state_margin_sec,
    )
    print(json.dumps({"out_csv": args.out_csv, "pairs": len(rows)}, ensure_ascii=False, indent=2))


def _cmd_build_balanced(args: argparse.Namespace) -> None:
    summary = build_balanced_dataset(
        pairs_csv=args.pairs_csv,
        out_dir=args.out_dir,
        train_per_state=args.train_per_state,
        val_per_state=args.val_per_state,
        min_gap_sec=args.min_gap_sec,
        heldout_mode=args.heldout_mode,
        imgsz=args.imgsz,
        workspace_root=Path(args.workspace_root).expanduser(),
        seed=args.seed,
        jpeg_quality=args.jpeg_quality,
        min_hand_conf=args.min_hand_conf,
        min_wheel_conf=args.min_wheel_conf,
        min_iou=args.min_iou,
        max_off_iou=args.max_off_iou,
        min_state_margin_sec=args.min_state_margin_sec,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
