from __future__ import annotations

import csv
from pathlib import Path

import autodri.workflows.autoui_experiments as autoui_exp
from autodri.workflows.autoui_experiments import (
    find_wheel_review_det_csv,
    generate_wheel_review_frame,
    join_wheel_validation,
    prepare_fewshot_datasets,
    prepare_wheel_review_rows,
    resolve_wheel_review_video,
    write_deployment_matrix_report,
    write_fewshot_curve_report,
    summarize_roi_audit,
    summarize_wheel_validation,
    write_wheel_review_results,
)


MANIFEST_FIELDS = ["split", "label", "domain", "frame_id", "timestamp", "video", "src_rel", "dst_rel", "augmented"]


def _write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_cls_dataset(root: Path, participant: str) -> Path:
    dataset = root / participant
    rows: list[dict[str, object]] = []
    labels = ["Forward", "In-Car", "Non-Forward"]
    for idx in range(12):
        label = labels[idx % len(labels)]
        split = "val" if idx >= 9 else "train"
        rel = f"{split}/{label}/{participant}_{idx}.jpg"
        (dataset / rel).parent.mkdir(parents=True, exist_ok=True)
        (dataset / rel).write_bytes(b"image")
        rows.append(
            {
                "split": split,
                "label": label,
                "domain": participant,
                "frame_id": idx,
                "timestamp": float(idx * 31),
                "video": f"{participant}_{idx}.mp4",
                "src_rel": rel,
                "dst_rel": rel,
                "augmented": 0,
            }
        )
    _write_rows(dataset / "split_manifest.csv", rows, MANIFEST_FIELDS)
    return dataset


def test_prepare_fewshot_datasets_uses_budgeted_train_and_frozen_test(tmp_path: Path) -> None:
    source = _write_cls_dataset(tmp_path, "p1")

    summary = prepare_fewshot_datasets(
        {"p1": source},
        out_dir=tmp_path / "fewshot",
        budgets=[6],
        seeds=[13],
        labels=("Forward", "In-Car", "Non-Forward"),
        internal_val_ratio=0.34,
    )

    assert summary[0]["budget"] == 6
    assert summary[0]["selected_label_count"] == 6
    dataset = Path(str(summary[0]["data_dir"]))
    rows = _read_rows(dataset / "split_manifest.csv")
    split_counts = {split: sum(1 for row in rows if row["split"] == split) for split in {"train", "internal_val", "test"}}
    assert split_counts["test"] == 3
    assert split_counts["train"] + split_counts["internal_val"] == 6
    assert all(row["source_split"] == "val" for row in rows if row["split"] == "test")
    assert (dataset / "val").is_symlink()


def test_prepare_fewshot_datasets_excludes_train_groups_overlapping_frozen_test(tmp_path: Path) -> None:
    source = tmp_path / "p1"
    rows: list[dict[str, object]] = []
    specs = [
        ("train", "Forward", "shared.mp4", 10.0),
        ("train", "In-Car", "shared.mp4", 40.0),
        ("train", "Non-Forward", "shared.mp4", 95.0),
        ("val", "Forward", "shared.mp4", 10.0),
        ("val", "In-Car", "other.mp4", 80.0),
    ]
    for idx, (split, label, video, timestamp) in enumerate(specs):
        rel = f"{split}/{label}/p1_{idx}.jpg"
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / rel).write_bytes(b"image")
        rows.append(
            {
                "split": split,
                "label": label,
                "domain": "p1",
                "frame_id": idx,
                "timestamp": timestamp,
                "video": video,
                "src_rel": rel,
                "dst_rel": rel,
                "augmented": 0,
            }
        )
    _write_rows(source / "split_manifest.csv", rows, MANIFEST_FIELDS)

    summary = prepare_fewshot_datasets(
        {"p1": source},
        out_dir=tmp_path / "fewshot",
        budgets=[2],
        seeds=[13],
        labels=("Forward", "In-Car", "Non-Forward"),
        targets=("p1",),
        internal_val_ratio=0.5,
    )

    dataset = Path(str(summary[0]["data_dir"]))
    manifest_rows = _read_rows(dataset / "split_manifest.csv")
    train_groups = {
        (row["domain"], row["video"], int(float(row["timestamp"]) // 30))
        for row in manifest_rows
        if row["split"] in {"train", "internal_val"}
    }
    test_groups = {
        (row["domain"], row["video"], int(float(row["timestamp"]) // 30))
        for row in manifest_rows
        if row["split"] == "test"
    }
    assert train_groups.isdisjoint(test_groups)
    assert all(row["source_split"] == "val" for row in manifest_rows if row["split"] == "test")


def test_prepare_fewshot_datasets_keeps_every_selected_label_in_train(tmp_path: Path) -> None:
    source = tmp_path / "p1"
    rows: list[dict[str, object]] = []
    train_labels = ["Forward"] * 20 + ["In-Car"] * 4 + ["Non-Forward"]
    val_labels = ["Forward", "In-Car", "Non-Forward"]
    for idx, label in enumerate(train_labels + val_labels):
        split = "train" if idx < len(train_labels) else "val"
        rel = f"{split}/{label}/p1_{idx}.jpg"
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / rel).write_bytes(b"image")
        rows.append(
            {
                "split": split,
                "label": label,
                "domain": "p1",
                "frame_id": idx,
                "timestamp": float(idx * 31),
                "video": f"video_{idx}.mp4",
                "src_rel": rel,
                "dst_rel": rel,
                "augmented": 0,
            }
        )
    _write_rows(source / "split_manifest.csv", rows, MANIFEST_FIELDS)

    summary = prepare_fewshot_datasets(
        {"p1": source},
        out_dir=tmp_path / "fewshot",
        budgets=[25],
        seeds=[101],
        labels=("Forward", "In-Car", "Non-Forward"),
        targets=("p1",),
        internal_val_ratio=0.2,
    )

    dataset = Path(str(summary[0]["data_dir"]))
    manifest_rows = _read_rows(dataset / "split_manifest.csv")
    train_labels_seen = {row["label"] for row in manifest_rows if row["split"] == "train"}
    assert train_labels_seen == {"Forward", "In-Car", "Non-Forward"}


def test_summarize_roi_audit_computes_human_metrics_and_manual_corrections(tmp_path: Path) -> None:
    manifest = tmp_path / "roi_manifest.csv"
    _write_rows(
        manifest,
        [
            {"participant": "p1", "video_or_source": "a.mp4", "frame_or_reference": "r1", "artifact_path": "a.jpg", "proposed_status": "pending", "notes": ""},
            {"participant": "p1", "video_or_source": "b.mp4", "frame_or_reference": "r2", "artifact_path": "b.jpg", "proposed_status": "confirmed_swap", "notes": ""},
        ],
    )
    review = tmp_path / "roi_review_results.csv"
    _write_rows(
        review,
        [
            {"participant": "p1", "video_or_source": "a.mp4", "frame_or_reference": "r1", "human_valid": "yes", "wrong_panel": "no", "clipping": "no"},
            {"participant": "p1", "video_or_source": "b.mp4", "frame_or_reference": "r2", "human_valid": "no", "wrong_panel": "yes", "clipping": "yes"},
        ],
    )
    current = tmp_path / "current" / "p1_gaze_rois.current.csv"
    manual = tmp_path / "manual" / "p1_gaze_rois.manual.csv"
    roi_fields = ["domain_id", "video", "roi_x1", "roi_y1", "roi_x2", "roi_y2", "n_samples", "source_swapped", "source_uncertain", "roi_note"]
    _write_rows(
        current,
        [{"domain_id": "d1", "video": "a.mp4", "roi_x1": 1, "roi_y1": 2, "roi_x2": 3, "roi_y2": 4, "n_samples": 1, "source_swapped": 0, "source_uncertain": 0, "roi_note": ""}],
        roi_fields,
    )
    _write_rows(
        manual,
        [{"domain_id": "d1", "video": "a.mp4", "roi_x1": 1, "roi_y1": 2, "roi_x2": 5, "roi_y2": 4, "n_samples": 1, "source_swapped": 0, "source_uncertain": 0, "roi_note": ""}],
        roi_fields,
    )

    outputs = summarize_roi_audit(
        manifest_csv=manifest,
        out_dir=tmp_path / "out",
        review_results_csv=review,
        current_roi_dir=tmp_path / "current",
        manual_roi_csvs=[manual],
    )

    summary = _read_rows(outputs["summary_csv"])[0]
    assert summary["reviewed_rows"] == "2"
    assert summary["valid_rate"] == "0.500000"
    assert summary["wrong_panel_rate"] == "0.500000"
    assert summary["clipping_rate"] == "0.500000"
    assert summary["manual_correction_rate"] == "1.000000"
    diff_rows = _read_rows(outputs["manual_vs_current_csv"])
    assert diff_rows[0]["coords_changed"] == "1"


def test_wheel_validation_join_and_summary_compute_agreement(tmp_path: Path) -> None:
    manifest = tmp_path / "wheel_validation_manifest.csv"
    _write_rows(
        manifest,
        [
            {"participant": "p1", "source_csv": "seg1.wheel.csv", "state": "ON", "timestamp_or_row": "last_row", "review_priority": "high", "notes": ""},
            {"participant": "p1", "source_csv": "seg2.wheel.csv", "state": "OFF", "timestamp_or_row": "last_row", "review_priority": "high", "notes": ""},
        ],
    )
    wheel_dir = tmp_path / "wheel"
    state_fields = ["frame", "time_sec", "video_time_sec", "stable_state", "stable_hand_on_wheel"]
    _write_rows(
        wheel_dir / "seg1.wheel.csv",
        [{"frame": 1, "time_sec": 0.1, "video_time_sec": 10.0, "stable_state": "ON", "stable_hand_on_wheel": 1}],
        state_fields,
    )
    _write_rows(
        wheel_dir / "seg2.wheel.csv",
        [{"frame": 2, "time_sec": 0.2, "video_time_sec": 20.0, "stable_state": "ON", "stable_hand_on_wheel": 1}],
        state_fields,
    )
    wheel_map = tmp_path / "p1_wheel_map.current.csv"
    _write_rows(
        wheel_map,
        [
            {"video_path": "video1.mp4", "wheel_csv": str(wheel_dir / "seg1.wheel.csv"), "segment_uid": "s1"},
            {"video_path": "video2.mp4", "wheel_csv": str(wheel_dir / "seg2.wheel.csv"), "segment_uid": "s2"},
        ],
    )
    human = tmp_path / "human.csv"
    _write_rows(
        human,
        [
            {"participant": "p1", "source_csv": "seg1.wheel.csv", "human_state": "ON"},
            {"participant": "p1", "source_csv": "seg2.wheel.csv", "human_state": "OFF"},
        ],
    )

    joined = join_wheel_validation(manifest, tmp_path / "joined.csv", wheel_maps=[wheel_map])
    summary = summarize_wheel_validation(joined, tmp_path / "wheel_summary.csv", human_results_csv=human)

    joined_rows = _read_rows(joined)
    assert joined_rows[0]["video_path"] == "video1.mp4"
    assert joined_rows[1]["model_state"] == "ON"
    summary_rows = _read_rows(summary)
    assert summary_rows[0]["reviewed_rows"] == "2"
    assert summary_rows[0]["agreement_rate"] == "0.500000"
    assert summary_rows[0]["false_on"] == "1"
    assert summary_rows[0]["false_off"] == "0"


def test_wheel_review_helpers_write_human_results_and_whitelist_videos(tmp_path: Path) -> None:
    joined = tmp_path / "joined.csv"
    allowed_video = tmp_path / "video.mp4"
    allowed_video.write_bytes(b"video")
    _write_rows(
        joined,
        [
            {
                "participant": "p1",
                "source_csv": "seg1.wheel.csv",
                "video_path": str(allowed_video),
                "model_state": "OFF",
                "model_video_time_sec": "12.5",
                "segment_start_sec": "10.0",
                "segment_end_sec": "20.0",
            }
        ],
    )

    rows = prepare_wheel_review_rows(joined)

    assert rows[0]["row_id"] == "0"
    assert rows[0]["video_url"] == "/video/0"
    assert rows[0]["model_state"] == "OFF"
    assert resolve_wheel_review_video(joined, "0") == allowed_video
    assert resolve_wheel_review_video(joined, "../secret") is None

    out = tmp_path / "wheel_validation_human_review.csv"
    write_wheel_review_results(
        out,
        [
            {"participant": "p1", "source_csv": "seg1.wheel.csv", "human_state": "ON", "human_notes": "visible final frame"},
            {"participant": "p2", "source_csv": "seg2.wheel.csv", "human_state": "invalid", "human_notes": "ignored"},
        ],
    )

    written = _read_rows(out)
    assert written == [
        {"participant": "p1", "source_csv": "seg1.wheel.csv", "human_state": "ON", "human_notes": "visible final frame"}
    ]


def test_wheel_review_frame_generation_uses_nearest_groundingdino_detection(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    det_csv = tmp_path / "wheel_teacher_det" / "p1" / "seg1.wheel.det.csv"
    joined = tmp_path / "joined.csv"
    frame = autoui_exp.np.full((80, 120, 3), 240, dtype=autoui_exp.np.uint8)
    writer = autoui_exp.cv2.VideoWriter(str(video), autoui_exp.cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (120, 80))
    assert writer.isOpened()
    writer.write(frame)
    writer.release()
    _write_rows(
        det_csv,
        [
            {
                "video_path": str(video),
                "frame": 0,
                "video_frame": 0,
                "time_sec": 0.0,
                "video_time_sec": 0.0,
                "roi_x1": 10,
                "roi_y1": 8,
                "roi_x2": 90,
                "roi_y2": 70,
                "class_name": "hand",
                "confidence": 0.9,
                "x1": 20,
                "y1": 20,
                "x2": 40,
                "y2": 45,
            }
        ],
    )
    _write_rows(
        joined,
        [
            {
                "participant": "p1",
                "source_csv": "prefix__seg1.wheel.csv",
                "segment_uid": "seg1",
                "video_path": str(video),
                "model_frame": "0",
                "model_video_time_sec": "0.0",
            }
        ],
    )

    rows = prepare_wheel_review_rows(joined, det_root=tmp_path / "wheel_teacher_det")
    out = generate_wheel_review_frame(joined, "0", out_dir=tmp_path / "frames", det_root=tmp_path / "wheel_teacher_det")

    assert rows[0]["frame_url"] == "/frame/0"
    assert find_wheel_review_det_csv(_read_rows(joined)[0], det_root=tmp_path / "wheel_teacher_det") == det_csv
    assert out is not None and out.exists()
    image = autoui_exp.cv2.imread(str(out))
    assert image is not None
    assert image.shape[0] == 80
    assert image.shape[1] == 120

    _write_rows(
        joined,
        [
            {
                "participant": "p1",
                "source_csv": "prefix__seg1.wheel.csv",
                "segment_uid": "seg1",
                "video_path": "",
                "model_frame": "0",
                "model_video_time_sec": "0.0",
            }
        ],
    )
    fallback_rows = prepare_wheel_review_rows(joined, det_root=tmp_path / "wheel_teacher_det")
    fallback_out = generate_wheel_review_frame(joined, "0", out_dir=tmp_path / "frames_fallback", det_root=tmp_path / "wheel_teacher_det")

    assert fallback_rows[0]["frame_url"] == "/frame/0"
    assert fallback_out is not None and fallback_out.exists()


def test_wheel_review_rows_mark_single_panel_roi_as_unverified(tmp_path: Path) -> None:
    joined = tmp_path / "joined.csv"
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    det_csv = tmp_path / "wheel_teacher_det" / "p13" / "p13_seg_00141.wheel.det.csv"
    _write_rows(
        det_csv,
        [
            {
                "video_path": str(video),
                "frame": "0",
                "video_frame": "0",
                "video_time_sec": "0.0",
                "roi_x1": "950",
                "roi_y1": "300",
                "roi_x2": "1650",
                "roi_y2": "700",
            }
        ],
    )
    _write_rows(
        tmp_path / "artifacts" / "manifests" / "current" / "p13_wheel_rois.current.csv",
        [
            {
                "domain_id": "p13",
                "video": str(video),
                "roi_x1": "950",
                "roi_y1": "300",
                "roi_x2": "1650",
                "roi_y2": "700",
                "n_samples": "1",
                "inferred_rule": "single_panel_same_as_gaze",
            }
        ],
    )
    _write_rows(
        joined,
        [
            {
                "participant": "p13",
                "source_csv": "20250429_192927-201727__p13_seg_00141.wheel.csv",
                "segment_uid": "p13_seg_00141",
                "video_path": str(video),
            }
        ],
    )

    rows = prepare_wheel_review_rows(joined, workspace_root=tmp_path, det_root=tmp_path / "wheel_teacher_det")

    assert rows[0]["roi_review_status"] == "unverified_single_panel_same_as_gaze"
    assert rows[0]["roi_review_trusted"] == 0
    assert rows[0]["roi_coords"] == "950,300,1650,700"


def test_wheel_review_rows_use_layout_candidate_for_suspect_wheel_roi(tmp_path: Path) -> None:
    joined = tmp_path / "joined.csv"
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    det_csv = tmp_path / "wheel_teacher_det" / "p13" / "p13_seg_00141.wheel.det.csv"
    _write_rows(
        det_csv,
        [
            {
                "video_path": str(video),
                "frame": "0",
                "video_frame": "0",
                "video_time_sec": "0.0",
                "roi_x1": "950",
                "roi_y1": "300",
                "roi_x2": "1650",
                "roi_y2": "700",
            }
        ],
    )
    _write_rows(
        tmp_path / "artifacts" / "manifests" / "current" / "p13_wheel_rois.current.csv",
        [
            {
                "domain_id": "p13",
                "video": str(video),
                "roi_x1": "950",
                "roi_y1": "300",
                "roi_x2": "1650",
                "roi_y2": "700",
                "n_samples": "1",
                "inferred_rule": "single_panel_same_as_gaze",
            }
        ],
    )
    _write_rows(
        joined,
        [
            {
                "participant": "p13",
                "source_csv": "20250429_192927-201727__p13_seg_00141.wheel.csv",
                "segment_uid": "p13_seg_00141",
                "video_path": str(video),
            }
        ],
    )

    rows = prepare_wheel_review_rows(joined, workspace_root=tmp_path, det_root=tmp_path / "wheel_teacher_det")

    assert rows[0]["roi_coords"] == "950,300,1650,700"
    assert rows[0]["wheel_evidence_roi_coords"] == "0,0,960,700"
    assert rows[0]["wheel_evidence_roi_source"] == "layout_candidate_from_review_grid"
    assert rows[0]["wheel_evidence_roi_trusted"] == 0
    assert rows[0]["wheel_evidence_roi_status"] == "candidate_wheel_panel_unverified"


def test_wheel_review_rows_use_p11_left_panel_candidate(tmp_path: Path) -> None:
    joined = tmp_path / "joined.csv"
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    det_csv = tmp_path / "wheel_teacher_det" / "p11" / "p11_seg_00601.wheel.det.csv"
    _write_rows(
        det_csv,
        [
            {
                "video_path": str(video),
                "frame": "0",
                "video_frame": "0",
                "video_time_sec": "0.0",
                "roi_x1": "720",
                "roi_y1": "0",
                "roi_x2": "1440",
                "roi_y2": "1080",
            }
        ],
    )
    _write_rows(
        tmp_path / "artifacts" / "manifests" / "current" / "p11_wheel_rois.current.csv",
        [
            {
                "domain_id": "p11",
                "video": str(video),
                "roi_x1": "720",
                "roi_y1": "0",
                "roi_x2": "1440",
                "roi_y2": "1080",
                "n_samples": "1",
                "inferred_rule": "single_panel_same_as_gaze",
            }
        ],
    )
    _write_rows(
        joined,
        [
            {
                "participant": "p11",
                "source_csv": "20250522_214140-220809__p11_seg_00601.wheel.csv",
                "segment_uid": "p11_seg_00601",
                "video_path": str(video),
            }
        ],
    )

    rows = prepare_wheel_review_rows(joined, workspace_root=tmp_path, det_root=tmp_path / "wheel_teacher_det")

    assert rows[0]["roi_review_status"] == "unverified_single_panel_same_as_gaze"
    assert rows[0]["roi_coords"] == "720,0,1440,1080"
    assert rows[0]["wheel_evidence_roi_coords"] == "0,0,720,540"
    assert rows[0]["wheel_evidence_roi_source"] == "layout_candidate_from_review_grid"


def test_wheel_review_page_labels_pipeline_state_as_pipeline_not_model() -> None:
    html = autoui_exp._wheel_review_html()

    assert "Pipeline stable_state" in html
    assert "Seed label" in html
    assert "GroundingDINO frame" in html
    assert "Wheel evidence" in html
    assert "<img id=\"frameImage\"" in html
    assert "<video" not in html
    assert ">MODEL<" not in html


def test_wheel_review_frame_caption_marks_unverified_roi_and_does_not_draw_it(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    rectangles: list[tuple[int, int, int]] = []

    def fake_put_text(image, text, *args, **kwargs):
        calls.append(str(text))
        return image

    def fake_rectangle(image, pt1, pt2, color, *args, **kwargs):
        rectangles.append(tuple(color))
        return image

    monkeypatch.setattr(autoui_exp.cv2, "putText", fake_put_text)
    monkeypatch.setattr(autoui_exp.cv2, "rectangle", fake_rectangle)
    frame = autoui_exp.np.full((80, 120, 3), 240, dtype=autoui_exp.np.uint8)
    det_rows = [
        {
            "frame": "0",
            "video_time_sec": "0.0",
            "roi_x1": "10",
            "roi_y1": "8",
            "roi_x2": "90",
            "roi_y2": "70",
            "class_name": "hand",
            "confidence": "0.9",
            "x1": "20",
            "y1": "20",
            "x2": "40",
            "y2": "45",
        }
    ]

    autoui_exp._draw_wheel_review_frame(
        frame,
        det_rows,
        {
            "participant": "p13",
            "source_csv": "seg1.wheel.csv",
            "roi_review_trusted": "0",
            "roi_review_status": "unverified_single_panel_same_as_gaze",
        },
    )

    assert any("ROI unverified" in text for text in calls)
    assert (230, 215, 80) not in rectangles


def test_wheel_review_frame_draws_candidate_evidence_roi_without_verified_roi(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    rectangles: list[tuple[int, int, int]] = []

    def fake_put_text(image, text, *args, **kwargs):
        calls.append(str(text))
        return image

    def fake_rectangle(image, pt1, pt2, color, *args, **kwargs):
        rectangles.append(tuple(color))
        return image

    monkeypatch.setattr(autoui_exp.cv2, "putText", fake_put_text)
    monkeypatch.setattr(autoui_exp.cv2, "rectangle", fake_rectangle)
    frame = autoui_exp.np.full((80, 120, 3), 240, dtype=autoui_exp.np.uint8)
    det_rows = [
        {
            "frame": "0",
            "video_time_sec": "0.0",
            "roi_x1": "10",
            "roi_y1": "8",
            "roi_x2": "90",
            "roi_y2": "70",
            "class_name": "hand",
            "confidence": "0.9",
            "x1": "20",
            "y1": "20",
            "x2": "40",
            "y2": "45",
        }
    ]

    autoui_exp._draw_wheel_review_frame(
        frame,
        det_rows,
        {
            "participant": "p13",
            "source_csv": "seg1.wheel.csv",
            "roi_review_trusted": "0",
            "roi_review_status": "unverified_single_panel_same_as_gaze",
            "wheel_evidence_roi_coords": "0,0,96,70",
            "wheel_evidence_roi_status": "candidate_wheel_panel_unverified",
        },
    )

    assert any("Wheel evidence" in text for text in calls)
    assert (230, 215, 80) not in rectangles
    assert (80, 180, 245) in rectangles


def test_predict_matrix_only_schedules_training_successes(tmp_path: Path, monkeypatch) -> None:
    matrix_rows = [
        {
            "dataset": "ds",
            "target_participant": "p1",
            "data_dir": "data",
            "seed": "13",
            "model": "resnet50",
            "trainer": "torchvision_timm",
            "run_name": "success_run",
            "run_dir": "runs/success_run",
        },
        {
            "dataset": "ds",
            "target_participant": "p1",
            "data_dir": "data",
            "seed": "13",
            "model": "resnet50",
            "trainer": "torchvision_timm",
            "run_name": "pending_run",
            "run_dir": "runs/pending_run",
        },
    ]
    _write_rows(tmp_path / "run_matrix.csv", matrix_rows)
    _write_rows(
        tmp_path / "train_status.csv",
        [
            {
                "index": 1,
                "run_name": "success_run",
                "dataset": "ds",
                "target_participant": "p1",
                "model": "resnet50",
                "seed": "13",
                "status": "success",
                "returncode": 0,
                "start_time": "",
                "end_time": "",
                "log_path": "",
            }
        ],
    )
    scheduled: list[str] = []

    def fake_predict(root, index, row, device):
        scheduled.append(row["run_name"])
        return {
            "index": index,
            "run_name": row["run_name"],
            "dataset": row["dataset"],
            "target_participant": row["target_participant"],
            "model": row["model"],
            "seed": row["seed"],
            "status": "success",
            "returncode": 0,
            "start_time": "",
            "end_time": "",
            "log_path": "",
        }

    monkeypatch.setattr(autoui_exp, "_run_predict_one", fake_predict)

    rc = autoui_exp.run_predict_matrix(root=tmp_path, gpus=["cpu"], keep_going=True)

    assert rc == 0
    assert scheduled == ["success_run"]


def test_prepare_matrix_no_class_weights_flag_is_enabled_by_default(tmp_path: Path) -> None:
    source = _write_cls_dataset(tmp_path, "p1")

    prepare_fewshot_datasets(
        {"p1": source},
        out_dir=tmp_path / "fewshot",
        budgets=[6],
        seeds=[13],
        labels=("Forward", "In-Car", "Non-Forward"),
        targets=("p1",),
        models=(autoui_exp.ExperimentModel("resnet50", "convnet", "resnet", "torchvision_timm", "resnet50"),),
    )

    rows = _read_rows(tmp_path / "fewshot" / "run_matrix.csv")
    assert "--no-class-weights" in rows[0]["train_command"]


def test_prepare_parser_keeps_no_class_weights_semantics() -> None:
    parser = autoui_exp.build_parser()

    default_args = parser.parse_args(["prepare-fewshot"])
    no_weight_args = parser.parse_args(["prepare-fewshot", "--no-class-weights"])
    weighted_args = parser.parse_args(["prepare-fewshot", "--class-weights"])

    assert default_args.no_class_weights is True
    assert no_weight_args.no_class_weights is True
    assert weighted_args.no_class_weights is False


def test_write_fewshot_curve_report_aggregates_budget_metrics(tmp_path: Path) -> None:
    _write_rows(
        tmp_path / "eval" / "metrics.csv",
        [
            {"dataset": "fewshot_p1_b25", "model": "yolov8s-cls", "seed": "13", "primary3_macro_f1": "0.50", "primary3_balanced_accuracy": "0.60", "primary3_event_acc": "0.70"},
            {"dataset": "fewshot_p1_b25", "model": "resnet50", "seed": "13", "primary3_macro_f1": "0.55", "primary3_balanced_accuracy": "0.65", "primary3_event_acc": "0.75"},
            {"dataset": "fewshot_p2_b25", "model": "resnet50", "seed": "29", "primary3_macro_f1": "0.65", "primary3_balanced_accuracy": "0.75", "primary3_event_acc": "0.85"},
            {"dataset": "fewshot_p1_b50", "model": "resnet50", "seed": "13", "primary3_macro_f1": "0.80", "primary3_balanced_accuracy": "0.82", "primary3_event_acc": "0.90"},
        ],
    )

    outputs = write_fewshot_curve_report(tmp_path)

    rows = _read_rows(outputs["summary_csv"])
    resnet25 = next(row for row in rows if row["model"] == "resnet50" and row["budget"] == "25")
    assert resnet25["runs"] == "2"
    assert resnet25["primary3_macro_f1_mean"] == "0.600000"
    assert outputs["tex"].exists()
    assert outputs["curve_svg"].exists()


def test_write_deployment_matrix_report_summarizes_existing_latency_and_parity(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs_torch" / "resnet_run"
    run_dir.mkdir(parents=True)
    (run_dir / "best.onnx").write_bytes(b"onnx")
    _write_rows(
        tmp_path / "run_matrix.csv",
        [
            {
                "dataset": "ds",
                "target_participant": "p1",
                "data_dir": str(tmp_path / "data"),
                "seed": "13",
                "model": "resnet50",
                "trainer": "torchvision_timm",
                "run_name": "resnet_run",
                "run_dir": str(run_dir),
            }
        ],
    )
    _write_rows(
        tmp_path / "train_status.csv",
        [
            {
                "index": 1,
                "run_name": "resnet_run",
                "dataset": "ds",
                "target_participant": "p1",
                "model": "resnet50",
                "seed": "13",
                "status": "success",
                "returncode": 0,
                "start_time": "",
                "end_time": "",
                "log_path": "",
            }
        ],
    )
    _write_rows(
        tmp_path / "deployment" / "latency_resnet_run.csv",
        [{"model": "resnet50", "batch_size": "1", "latency_p50_ms": "1.0", "latency_p95_ms": "2.0", "throughput_img_s": "100.0"}],
    )
    _write_rows(
        tmp_path / "deployment" / "parity_resnet_run.csv",
        [{"aligned_total": "10", "top1_matches": "10", "top1_parity": "1.000000"}],
    )

    outputs = write_deployment_matrix_report(tmp_path)

    rows = _read_rows(outputs["summary_csv"])
    assert rows[0]["run_name"] == "resnet_run"
    assert rows[0]["onnx_exists"] == "1"
    assert rows[0]["top1_parity"] == "1.000000"
    assert rows[0]["latency_b1_p50_ms"] == "1.0"


def test_deployment_provider_selection_distinguishes_cpu_and_gpu() -> None:
    assert autoui_exp._onnx_providers_for_device("cpu") == ["CPUExecutionProvider"]
    assert autoui_exp._onnx_providers_for_device("0") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert autoui_exp._onnx_env_for_device("2")["CUDA_VISIBLE_DEVICES"] == "2"


def test_yolo_export_command_requests_dynamic_onnx(tmp_path: Path, monkeypatch) -> None:
    weights = tmp_path / "runs_yolo" / "yolo_run" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"pt")
    commands: list[list[str]] = []

    def fake_run_checked(cmd, *, timeout):
        commands.append([str(x) for x in cmd])
        (weights / "best.onnx").write_bytes(b"onnx")

    monkeypatch.setattr(autoui_exp, "_run_checked", fake_run_checked)

    out = autoui_exp.ensure_run_onnx(
        {
            "trainer": "ultralytics",
            "run_dir": str(tmp_path / "runs_yolo" / "yolo_run"),
            "data_dir": str(tmp_path / "data"),
        },
        device="cpu",
    )

    assert out == weights / "best.onnx"
    assert "--dynamic" in commands[0]


def test_yolo_existing_static_onnx_is_refreshed_for_deployment(tmp_path: Path, monkeypatch) -> None:
    weights = tmp_path / "runs_yolo" / "yolo_run" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"pt")
    (weights / "best.onnx").write_bytes(b"static")
    commands: list[list[str]] = []

    def fake_run_checked(cmd, *, timeout):
        commands.append([str(x) for x in cmd])
        (weights / "best.onnx").write_bytes(b"dynamic")

    monkeypatch.setattr(autoui_exp, "_run_checked", fake_run_checked)
    monkeypatch.setattr(autoui_exp, "_onnx_has_dynamic_batch", lambda path: False)

    out = autoui_exp.ensure_run_onnx(
        {
            "trainer": "ultralytics",
            "run_dir": str(tmp_path / "runs_yolo" / "yolo_run"),
            "data_dir": str(tmp_path / "data"),
        },
        device="cpu",
    )

    assert out == weights / "best.onnx"
    assert commands
    assert "--dynamic" in commands[0]
