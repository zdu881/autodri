from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
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
        }


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
                if next_decode_frame is None or video_frame < next_decode_frame:
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


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
