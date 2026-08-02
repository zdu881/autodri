from __future__ import annotations

from pathlib import Path

from autodri.workflows.wheel_state_distill import (
    SegmentSpec,
    assign_block_split,
    assign_block_split_with_offset,
    assign_hash_split,
    build_balanced_dataset,
    compute_agreement,
    resolve_video_path,
    scan_support,
    should_seek_decode,
    write_image,
)


def test_resolve_video_path_uses_workspace_for_relative_data_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    video = workspace / "data" / "natural_driving_p1" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")

    resolved = resolve_video_path("data/natural_driving_p1/clip.mp4", workspace_root=workspace)

    assert resolved == video


def test_assign_block_split_holds_out_whole_time_blocks() -> None:
    spec = SegmentSpec(
        name="probe",
        state_csv=Path("state.csv"),
        det_csv=Path("det.csv"),
        video_path=Path("video.mp4"),
        roi=(0, 0, 100, 100),
    )

    splits = [
        assign_block_split(segment=spec, time_sec=t, val_ratio=0.25, block_sec=1.0)
        for t in (0.0, 0.4, 0.9, 1.0, 1.4, 2.0, 2.4, 4.0)
    ]

    assert splits[:3] == ["val", "val", "val"]
    assert splits[3:7] == ["train", "train", "train", "train"]
    assert splits[7:] == ["val"]


def test_assign_block_split_with_offset_shifts_validation_blocks() -> None:
    spec = SegmentSpec(
        name="probe",
        state_csv=Path("state.csv"),
        det_csv=Path("det.csv"),
        video_path=Path("video.mp4"),
        roi=(0, 0, 100, 100),
    )

    splits = [
        assign_block_split_with_offset(
            segment=spec,
            time_sec=t,
            val_ratio=0.25,
            block_sec=1.0,
            offset=1,
        )
        for t in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    ]

    assert splits == ["train", "val", "train", "train", "train", "val"]


def test_assign_hash_split_is_deterministic_and_frame_level() -> None:
    spec = SegmentSpec(
        name="probe",
        state_csv=Path("state.csv"),
        det_csv=Path("det.csv"),
        video_path=Path("video.mp4"),
        roi=(0, 0, 100, 100),
    )

    first = assign_hash_split(segment=spec, video_frame=123, val_ratio=0.2, seed=3407)
    second = assign_hash_split(segment=spec, video_frame=123, val_ratio=0.2, seed=3407)
    changed_seed = assign_hash_split(segment=spec, video_frame=123, val_ratio=0.2, seed=1)

    assert first == second
    assert {first, changed_seed} <= {"train", "val"}


def test_compute_agreement_reports_accuracy_and_confusion() -> None:
    rows = [
        ("OFF", "OFF"),
        ("OFF", "ON"),
        ("ON", "ON"),
        ("UNCERTAIN", "UNCERTAIN"),
    ]

    metrics = compute_agreement(rows)

    assert metrics.accuracy == 0.75
    assert metrics.correct == 3
    assert metrics.total == 4
    assert metrics.confusion[("OFF", "ON")] == 1
    metrics_dict = metrics.to_dict()
    assert metrics_dict["per_class"]["ON"]["support"] == 1
    assert metrics_dict["per_class"]["ON"]["precision"] == 0.5
    assert metrics_dict["per_class"]["ON"]["recall"] == 1.0
    assert metrics_dict["per_class"]["ON"]["f1"] == 2 / 3


def test_write_image_creates_parent_directories(tmp_path: Path) -> None:
    import cv2  # noqa: F401
    import numpy as np

    dst = tmp_path / "missing" / "OFF" / "frame.jpg"

    write_image(dst, np.zeros((8, 8, 3), dtype=np.uint8), jpeg_quality=90)

    assert dst.exists()


def test_should_seek_decode_for_large_frame_gaps() -> None:
    assert should_seek_decode(next_decode_frame=None, video_frame=100)
    assert not should_seek_decode(next_decode_frame=100, video_frame=120, max_gap_frames=300)
    assert should_seek_decode(next_decode_frame=100, video_frame=500, max_gap_frames=300)
    assert should_seek_decode(next_decode_frame=500, video_frame=100, max_gap_frames=300)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_video(path: Path, *, frames: int = 16) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (32, 32),
    )
    assert writer.isOpened()
    try:
        for i in range(frames):
            frame = np.full((32, 32, 3), i * 10 % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _state_rows(states: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, state in enumerate(states):
        rows.append(
            {
                "frame": idx,
                "time_sec": f"{idx * 0.5:.3f}",
                "video_time_sec": f"{idx * 0.5:.3f}",
                "video_frame": idx,
                "raw_state": state,
                "stable_state": state,
                "raw_hand_on_wheel": 1 if state == "ON" else 0,
                "stable_hand_on_wheel": 1 if state == "ON" else 0,
                "iou_max": "0.080" if state == "ON" else "0.000",
                "max_hand_conf": "0.800" if state != "UNCERTAIN" else "0.000",
                "max_wheel_conf": "0.850" if state != "UNCERTAIN" else "0.000",
            }
        )
    return rows


def test_scan_support_pairs_state_and_detection_csvs_with_clean_counts(tmp_path: Path) -> None:
    video = tmp_path / "videos" / "clip.mp4"
    _write_video(video, frames=8)
    state_csv = tmp_path / "states" / "drive__p99_seg_00001.wheel.csv"
    rows = _state_rows(["ON", "OFF", "UNCERTAIN", "ON"])
    rows[3]["raw_state"] = "OFF"
    _write_csv(state_csv, rows)
    det_csv = tmp_path / "det" / "p99" / "p99_seg_00001.wheel.det.csv"
    _write_csv(
        det_csv,
        [
            {
                "video_path": str(video),
                "roi_x1": 0,
                "roi_y1": 0,
                "roi_x2": 16,
                "roi_y2": 16,
            }
        ],
    )

    out_csv = tmp_path / "support.csv"
    support = scan_support(
        state_glob=str(tmp_path / "states" / "*.wheel.csv"),
        det_root=tmp_path / "det",
        out_csv=out_csv,
        workspace_root=tmp_path,
    )

    assert out_csv.exists()
    assert len(support) == 1
    assert support[0]["participant"] == "p99"
    assert support[0]["segment"] == "p99_seg_00001"
    assert support[0]["clean_on"] == 1
    assert support[0]["clean_off"] == 1
    assert support[0]["clean_uncertain"] == 1
    assert support[0]["clean_rows"] == 3


def test_scan_support_can_exclude_off_frames_near_contact_boundary(tmp_path: Path) -> None:
    video = tmp_path / "videos" / "clip.mp4"
    _write_video(video, frames=8)
    state_csv = tmp_path / "states" / "drive__p99_seg_00001.wheel.csv"
    rows = _state_rows(["OFF", "OFF", "ON"])
    rows[0]["iou_max"] = "0.020"
    rows[1]["iou_max"] = "0.000"
    _write_csv(state_csv, rows)
    det_csv = tmp_path / "det" / "p99" / "p99_seg_00001.wheel.det.csv"
    _write_csv(
        det_csv,
        [
            {
                "video_path": str(video),
                "roi_x1": 0,
                "roi_y1": 0,
                "roi_x2": 16,
                "roi_y2": 16,
            }
        ],
    )

    support = scan_support(
        state_glob=str(tmp_path / "states" / "*.wheel.csv"),
        det_root=tmp_path / "det",
        out_csv=tmp_path / "support.csv",
        workspace_root=tmp_path,
        max_off_iou=0.010,
    )

    assert support[0]["clean_off"] == 1


def test_scan_support_can_exclude_frames_near_stable_state_transitions(tmp_path: Path) -> None:
    video = tmp_path / "videos" / "clip.mp4"
    _write_video(video, frames=8)
    state_csv = tmp_path / "states" / "drive__p99_seg_00001.wheel.csv"
    _write_csv(state_csv, _state_rows(["OFF", "OFF", "ON", "ON", "ON"]))
    det_csv = tmp_path / "det" / "p99" / "p99_seg_00001.wheel.det.csv"
    _write_csv(
        det_csv,
        [
            {
                "video_path": str(video),
                "roi_x1": 0,
                "roi_y1": 0,
                "roi_x2": 16,
                "roi_y2": 16,
            }
        ],
    )

    support = scan_support(
        state_glob=str(tmp_path / "states" / "*.wheel.csv"),
        det_root=tmp_path / "det",
        out_csv=tmp_path / "support.csv",
        workspace_root=tmp_path,
        min_state_margin_sec=0.75,
    )

    assert support[0]["clean_off"] == 1
    assert support[0]["clean_on"] == 1


def test_build_balanced_dataset_uses_segment_holdout_and_min_gap(tmp_path: Path) -> None:
    video = tmp_path / "videos" / "clip.mp4"
    _write_video(video, frames=16)
    pair_rows: list[dict[str, object]] = []
    for segment_idx, segment_name in enumerate(("p99_seg_00001", "p99_seg_00002", "p99_seg_00003")):
        state_csv = tmp_path / "states" / f"drive__{segment_name}.wheel.csv"
        det_csv = tmp_path / "det" / "p99" / f"{segment_name}.wheel.det.csv"
        _write_csv(state_csv, _state_rows(["ON", "ON", "OFF", "UNCERTAIN", "ON", "OFF"]))
        _write_csv(
            det_csv,
            [
                {
                    "video_path": str(video),
                    "roi_x1": 0,
                    "roi_y1": 0,
                    "roi_x2": 16,
                    "roi_y2": 16,
                }
            ],
        )
        pair_rows.append(
            {
                "participant": "p99",
                "segment": segment_name,
                "state_csv": str(state_csv),
                "det_csv": str(det_csv),
                "clean_on": 3,
                "clean_off": 2,
                "clean_uncertain": 1,
                "clean_rows": 6,
                "raw_rows": 6,
            }
        )
    pairs_csv = tmp_path / "pairs.csv"
    _write_csv(pairs_csv, pair_rows)

    summary = build_balanced_dataset(
        pairs_csv=pairs_csv,
        out_dir=tmp_path / "balanced",
        train_per_state=2,
        val_per_state=2,
        min_gap_sec=1.0,
        heldout_mode="segment",
        imgsz=16,
        workspace_root=tmp_path,
    )

    assert summary["counts"]["val/ON"] == 2
    assert summary["counts"]["train/ON"] == 2
    manifest = tmp_path / "balanced" / "manifest.csv"
    import csv

    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    val_segments = {row["segment"] for row in rows if row["split"] == "val"}
    train_segments = {row["segment"] for row in rows if row["split"] == "train"}
    assert len(val_segments) >= 2
    assert val_segments.isdisjoint(train_segments)
    on_times = sorted(float(row["time_sec"]) for row in rows if row["teacher_state"] == "ON")
    assert 0.5 not in on_times


def test_build_balanced_dataset_can_hold_out_time_blocks_within_each_segment(tmp_path: Path) -> None:
    video = tmp_path / "videos" / "clip.mp4"
    _write_video(video, frames=24)
    pair_rows: list[dict[str, object]] = []
    states = ["ON", "OFF", "UNCERTAIN", "ON", "OFF", "UNCERTAIN", "ON", "OFF", "UNCERTAIN"]
    for segment_name in ("p99_seg_00001", "p99_seg_00002"):
        state_csv = tmp_path / "states" / f"drive__{segment_name}.wheel.csv"
        det_csv = tmp_path / "det" / "p99" / f"{segment_name}.wheel.det.csv"
        _write_csv(state_csv, _state_rows(states))
        _write_csv(
            det_csv,
            [
                {
                    "video_path": str(video),
                    "roi_x1": 0,
                    "roi_y1": 0,
                    "roi_x2": 16,
                    "roi_y2": 16,
                }
            ],
        )
        pair_rows.append(
            {
                "participant": "p99",
                "segment": segment_name,
                "state_csv": str(state_csv),
                "det_csv": str(det_csv),
                "clean_on": 3,
                "clean_off": 3,
                "clean_uncertain": 3,
                "clean_rows": 9,
                "raw_rows": 9,
            }
        )
    pairs_csv = tmp_path / "pairs.csv"
    _write_csv(pairs_csv, pair_rows)

    summary = build_balanced_dataset(
        pairs_csv=pairs_csv,
        out_dir=tmp_path / "balanced_blocks",
        train_per_state=3,
        val_per_state=2,
        min_gap_sec=0.0,
        heldout_mode="time-block",
        imgsz=16,
        workspace_root=tmp_path,
    )

    assert summary["heldout_mode"] == "time-block"
    assert summary["counts"]["val/ON"] == 2
    assert summary["counts"]["train/ON"] == 3
    manifest = tmp_path / "balanced_blocks" / "manifest.csv"
    import csv

    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    val_segments = {row["segment"] for row in rows if row["split"] == "val"}
    train_segments = {row["segment"] for row in rows if row["split"] == "train"}
    assert val_segments == {"p99_seg_00001", "p99_seg_00002"}
    assert train_segments == {"p99_seg_00001", "p99_seg_00002"}
    for segment_name in val_segments | train_segments:
        val_blocks = {
            int(float(row["time_sec"]) // 1)
            for row in rows
            if row["segment"] == segment_name and row["split"] == "val"
        }
        train_blocks = {
            int(float(row["time_sec"]) // 1)
            for row in rows
            if row["segment"] == segment_name and row["split"] == "train"
        }
        assert val_blocks
        assert train_blocks
        assert val_blocks.isdisjoint(train_blocks)


def test_build_balanced_dataset_respects_state_transition_margin(tmp_path: Path) -> None:
    video = tmp_path / "videos" / "clip.mp4"
    _write_video(video, frames=12)
    segment_name = "p99_seg_00001"
    state_csv = tmp_path / "states" / f"drive__{segment_name}.wheel.csv"
    det_csv = tmp_path / "det" / "p99" / f"{segment_name}.wheel.det.csv"
    _write_csv(state_csv, _state_rows(["OFF", "OFF", "ON", "ON", "ON", "OFF", "OFF"]))
    _write_csv(
        det_csv,
        [
            {
                "video_path": str(video),
                "roi_x1": 0,
                "roi_y1": 0,
                "roi_x2": 16,
                "roi_y2": 16,
            }
        ],
    )
    pairs_csv = tmp_path / "pairs.csv"
    _write_csv(
        pairs_csv,
        [
            {
                "participant": "p99",
                "segment": segment_name,
                "state_csv": str(state_csv),
                "det_csv": str(det_csv),
                "clean_on": 3,
                "clean_off": 4,
                "clean_uncertain": 0,
                "clean_rows": 7,
                "raw_rows": 7,
            }
        ],
    )

    summary = build_balanced_dataset(
        pairs_csv=pairs_csv,
        out_dir=tmp_path / "balanced_margin",
        train_per_state=1,
        val_per_state=1,
        min_gap_sec=0.0,
        heldout_mode="time-block",
        imgsz=16,
        workspace_root=tmp_path,
        min_state_margin_sec=0.75,
    )

    assert summary["min_state_margin_sec"] == 0.75
    manifest = tmp_path / "balanced_margin" / "manifest.csv"
    import csv

    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {float(row["time_sec"]) for row in rows} <= {0.0, 2.0, 3.0}
