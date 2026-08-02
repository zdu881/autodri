from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_dual_roi_review_rows_merge_gaze_and_wheel_current_manifests(tmp_path: Path) -> None:
    prep = _load_module("gaze_onnx/experiments/prepare_dual_roi_review_pack.py")
    current = tmp_path / "current"
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    _write_rows(
        current / "p8_gaze_rois.current.csv",
        [
            {
                "domain_id": "p8",
                "video": str(video),
                "roi_x1": 1900,
                "roi_y1": 660,
                "roi_x2": 3300,
                "roi_y2": 1400,
                "n_samples": 1,
                "source_swapped": 0,
                "source_uncertain": 1,
            }
        ],
    )
    _write_rows(
        current / "p8_wheel_rois.current.csv",
        [
            {
                "domain_id": "p8",
                "video": str(video),
                "roi_x1": 0,
                "roi_y1": 0,
                "roi_x2": 1900,
                "roi_y2": 1100,
                "n_samples": 1,
                "source_swapped": 0,
                "source_uncertain": 1,
                "inferred_rule": "dual_panel_opposite_from_right",
            }
        ],
    )

    rows = prep.gather_dual_roi_rows(current, ["p8"])

    assert rows == [
        {
            "participant": "p8",
            "video_abs": str(video),
            "gaze_roi_x1": "1900",
            "gaze_roi_y1": "660",
            "gaze_roi_x2": "3300",
            "gaze_roi_y2": "1400",
            "wheel_roi_x1": "0",
            "wheel_roi_y1": "0",
            "wheel_roi_x2": "1900",
            "wheel_roi_y2": "1100",
            "roi_note": "gaze_uncertain=1; wheel_uncertain=1; wheel_rule=dual_panel_opposite_from_right",
        }
    ]


def test_dual_roi_server_exports_manual_csvs_by_participant(tmp_path: Path) -> None:
    server = _load_module("gaze_onnx/experiments/serve_dual_roi_bbox_review.py")
    items = [
        server.Item(
            idx=0,
            participant="p8",
            video_rel="p8/a.mp4",
            video_abs="/videos/a.mp4",
            ref_raw="refs/a.jpg",
            ref_grid="refs/a_grid.jpg",
            frame_idx=10,
            timestamp_sec=0.4,
            width=3840,
            height=2160,
            gaze_roi_x1="1900",
            gaze_roi_y1="660",
            gaze_roi_x2="3300",
            gaze_roi_y2="1400",
            wheel_roi_x1="0",
            wheel_roi_y1="0",
            wheel_roi_x2="1900",
            wheel_roi_y2="1100",
            gaze_review_status="original_correct",
            wheel_review_status="corrected",
            roi_note="checked",
        ),
        server.Item(
            idx=1,
            participant="p9",
            video_rel="p9/b.mp4",
            video_abs="/videos/b.mp4",
            ref_raw="refs/b.jpg",
            ref_grid="refs/b_grid.jpg",
            frame_idx=20,
            timestamp_sec=0.8,
            width=3840,
            height=2160,
            gaze_roi_x1="1900",
            gaze_roi_y1="660",
            gaze_roi_x2="3300",
            gaze_roi_y2="1400",
            wheel_roi_x1="0",
            wheel_roi_y1="0",
            wheel_roi_x2="1900",
            wheel_roi_y2="1100",
            gaze_review_status="uncertain",
            wheel_review_status="original_correct",
            roi_note="needs replay",
        ),
    ]

    server.write_export_csvs(tmp_path, items)

    p8_gaze = _read_rows(tmp_path / "p8_gaze_rois.manual.csv")
    p8_wheel = _read_rows(tmp_path / "p8_wheel_rois.manual.csv")
    p9_gaze = _read_rows(tmp_path / "p9_gaze_rois.manual.csv")
    assert p8_gaze[0]["domain_id"] == "p8"
    assert p8_gaze[0]["video"] == "/videos/a.mp4"
    assert p8_gaze[0]["roi_note"] == "checked; gaze_review_status=original_correct"
    assert p8_wheel[0]["roi_note"] == "checked; wheel_review_status=corrected"
    assert p9_gaze[0]["domain_id"] == "p9"
