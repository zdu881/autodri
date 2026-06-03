from __future__ import annotations

from pathlib import Path

from autodri.workflows.wheel_state_distill import (
    SegmentSpec,
    assign_block_split,
    assign_hash_split,
    compute_agreement,
    resolve_video_path,
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


def test_write_image_creates_parent_directories(tmp_path: Path) -> None:
    import cv2  # noqa: F401
    import numpy as np

    dst = tmp_path / "missing" / "OFF" / "frame.jpg"

    write_image(dst, np.zeros((8, 8, 3), dtype=np.uint8), jpeg_quality=90)

    assert dst.exists()
