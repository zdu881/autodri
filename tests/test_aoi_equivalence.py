from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from autodri.aoi.equivalence import (
    DEFAULT_LABELS,
    PRIMARY_LABELS,
    PredictionRow,
    assign_internal_validation,
    compute_top1_parity,
    compute_event_accuracy,
    compute_frame_metrics,
    default_model_specs,
    generate_run_matrix,
    holm_adjust,
    mcnemar_exact,
    load_split_manifest,
    paired_bootstrap_delta,
    summarize_latency,
    validate_split_integrity,
)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "label",
                "domain",
                "frame_id",
                "timestamp",
                "video",
                "src_rel",
                "dst_rel",
                "augmented",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_internal_validation_split_keeps_frozen_val_as_test_and_augmented_rows_in_train(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset" / "split_manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "split": "train",
                "label": "Forward",
                "domain": "d1",
                "frame_id": "1",
                "timestamp": "1.0",
                "video": "v1.mp4",
                "src_rel": "images/a.jpg",
                "dst_rel": "train/Forward/a.jpg",
                "augmented": "0",
            },
            {
                "split": "train",
                "label": "Forward",
                "domain": "d1",
                "frame_id": "1",
                "timestamp": "1.0",
                "video": "v1.mp4",
                "src_rel": "images/a.jpg",
                "dst_rel": "train/Forward/a_aug.jpg",
                "augmented": "1",
            },
            {
                "split": "train",
                "label": "In-Car",
                "domain": "d1",
                "frame_id": "2",
                "timestamp": "31.0",
                "video": "v2.mp4",
                "src_rel": "images/b.jpg",
                "dst_rel": "train/In-Car/b.jpg",
                "augmented": "0",
            },
            {
                "split": "train",
                "label": "Non-Forward",
                "domain": "d2",
                "frame_id": "3",
                "timestamp": "61.0",
                "video": "v3.mp4",
                "src_rel": "images/c.jpg",
                "dst_rel": "train/Non-Forward/c.jpg",
                "augmented": "0",
            },
            {
                "split": "val",
                "label": "Other",
                "domain": "d3",
                "frame_id": "4",
                "timestamp": "91.0",
                "video": "v4.mp4",
                "src_rel": "images/d.jpg",
                "dst_rel": "val/Other/d.jpg",
                "augmented": "0",
            },
        ],
    )

    samples = load_split_manifest(manifest)
    assignment = assign_internal_validation(samples, val_ratio=0.34, seed=3)
    report = validate_split_integrity(samples, assignment)

    assert assignment["val/Other/d.jpg"] == "test"
    assert assignment["train/Forward/a_aug.jpg"] == "train"
    assert "internal_val" in set(assignment.values())
    assert report["group_leak_count"] == 0
    assert report["augmented_not_train_count"] == 0


def test_internal_validation_split_covers_each_eligible_label_when_grouped_data_is_imbalanced(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    frame_id = 0
    for label, count in [("Forward", 5), ("In-Car", 3), ("Non-Forward", 1)]:
        for idx in range(count):
            frame_id += 1
            rows.append(
                {
                    "split": "train",
                    "label": label,
                    "domain": "d1",
                    "frame_id": str(frame_id),
                    "timestamp": str(float(frame_id * 31)),
                    "video": f"{label}_{idx}.mp4",
                    "src_rel": f"images/{label}_{idx}.jpg",
                    "dst_rel": f"train/{label}/{label}_{idx}.jpg",
                    "augmented": "0",
                }
            )
    rows.append(
        {
            "split": "train",
            "label": "Non-Forward",
            "domain": "d1",
            "frame_id": "100",
            "timestamp": "3100.0",
            "video": "non_forward_aug.mp4",
            "src_rel": "images/non_forward_aug.jpg",
            "dst_rel": "train/Non-Forward/non_forward_aug.jpg",
            "augmented": "1",
        }
    )
    rows.append(
        {
            "split": "val",
            "label": "Other",
            "domain": "d2",
            "frame_id": "200",
            "timestamp": "6200.0",
            "video": "frozen.mp4",
            "src_rel": "images/frozen.jpg",
            "dst_rel": "val/Other/frozen.jpg",
            "augmented": "0",
        }
    )
    manifest = tmp_path / "dataset" / "split_manifest.csv"
    _write_manifest(manifest, rows)

    samples = load_split_manifest(manifest)
    assignment = assign_internal_validation(samples, val_ratio=0.2, seed=5)
    report = validate_split_integrity(samples, assignment)
    internal_val_labels = {
        sample.label for sample in samples if assignment[sample.dst_rel] == "internal_val"
    }

    assert {"Forward", "In-Car", "Non-Forward"} <= internal_val_labels
    assert assignment["train/Non-Forward/non_forward_aug.jpg"] == "train"
    assert assignment["val/Other/frozen.jpg"] == "test"
    assert report["group_leak_count"] == 0
    assert report["augmented_not_train_count"] == 0


def test_frame_and_event_metrics_separate_primary_three_from_other() -> None:
    rows = [
        PredictionRow("ds", "split", "m", 1, "a.jpg", "Forward", "Forward", "d", "v", 1.0),
        PredictionRow("ds", "split", "m", 1, "b.jpg", "In-Car", "Forward", "d", "v", 2.0),
        PredictionRow("ds", "split", "m", 1, "c.jpg", "Non-Forward", "Non-Forward", "d", "v", 33.0),
        PredictionRow("ds", "split", "m", 1, "d.jpg", "Other", "Other", "d", "v", 34.0),
    ]

    metrics = compute_frame_metrics(rows, labels=DEFAULT_LABELS, primary_labels=PRIMARY_LABELS)
    event_acc, event_total = compute_event_accuracy(rows, window_sec=30.0)

    assert math.isclose(metrics["primary3_accuracy"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(metrics["primary3_balanced_accuracy"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(metrics["primary3_macro_f1"], (2 / 3 + 0 + 1) / 3, rel_tol=1e-6)
    assert math.isclose(metrics["all4_accuracy"], 3 / 4, rel_tol=1e-6)
    assert event_total == 2
    assert math.isclose(event_acc, 1.0, rel_tol=1e-6)


def test_model_matrix_contains_yolo_convnet_and_light_vit_runs() -> None:
    specs = default_model_specs()
    matrix = generate_run_matrix(["holdout_car1", "holdout_car2", "stratified"], seeds=[1, 2], specs=specs)

    assert len(matrix) == 48
    assert {"yolov8n-cls", "yolov8s-cls", "yolov8m-cls"} <= {s.name for s in specs}
    assert {"resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny", "deit_tiny"} <= {
        s.name for s in specs
    }
    assert any(run.family == "yolo" for run in matrix)
    assert any(run.arch_group == "light_vit" for run in matrix)


def test_paired_bootstrap_delta_reports_noninferiority_for_identical_predictions() -> None:
    baseline = [
        PredictionRow("ds", "split", "yolo", 1, "a.jpg", "Forward", "Forward", "d", "v1", 1.0),
        PredictionRow("ds", "split", "yolo", 1, "b.jpg", "In-Car", "In-Car", "d", "v1", 2.0),
        PredictionRow("ds", "split", "yolo", 1, "c.jpg", "Non-Forward", "Non-Forward", "d", "v2", 33.0),
    ]
    candidate = [
        PredictionRow("ds", "split", "resnet50", 1, r.image_path, r.label, r.pred, r.domain, r.video, r.timestamp)
        for r in baseline
    ]

    result = paired_bootstrap_delta(
        candidate,
        baseline,
        metric_name="primary3_macro_f1",
        delta_margin=0.03,
        n_boot=100,
        seed=1,
    )

    assert math.isclose(result.observed_delta, 0.0, abs_tol=1e-12)
    assert result.ci_low >= 0.0
    assert result.noninferior is True


def test_latency_summary_reports_milliseconds_and_throughput() -> None:
    summary = summarize_latency([0.01, 0.02, 0.03], batch_size=2)

    assert math.isclose(summary["latency_p50_ms"], 20.0, rel_tol=1e-6)
    assert summary["latency_p95_ms"] > 20.0
    assert math.isclose(summary["throughput_img_s"], 100.0, rel_tol=1e-6)


def test_top1_parity_aligns_prediction_csvs_by_image_path() -> None:
    reference = [
        PredictionRow("ds", "val", "torch", 1, "a.jpg", "Forward", "Forward", "d", "v", 1.0),
        PredictionRow("ds", "val", "torch", 1, "b.jpg", "In-Car", "In-Car", "d", "v", 2.0),
    ]
    candidate = [
        PredictionRow("ds", "val", "onnx", 1, "b.jpg", "In-Car", "Forward", "d", "v", 2.0),
        PredictionRow("ds", "val", "onnx", 1, "a.jpg", "Forward", "Forward", "d", "v", 1.0),
    ]

    parity = compute_top1_parity(reference, candidate)

    assert parity["aligned_total"] == 2
    assert parity["top1_matches"] == 1
    assert math.isclose(parity["top1_parity"], 0.5, rel_tol=1e-6)


def test_mcnemar_and_holm_adjustment_for_paired_comparisons() -> None:
    baseline = [
        PredictionRow("ds", "val", "yolo", 1, "a.jpg", "Forward", "Forward", "d", "v", 1.0),
        PredictionRow("ds", "val", "yolo", 1, "b.jpg", "In-Car", "Forward", "d", "v", 2.0),
    ]
    candidate = [
        PredictionRow("ds", "val", "resnet50", 1, "a.jpg", "Forward", "Forward", "d", "v", 1.0),
        PredictionRow("ds", "val", "resnet50", 1, "b.jpg", "In-Car", "In-Car", "d", "v", 2.0),
    ]

    result = mcnemar_exact(candidate, baseline)
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert result["candidate_only_correct"] == 1
    assert result["baseline_only_correct"] == 0
    assert math.isclose(result["p_value"], 1.0, rel_tol=1e-6)
    assert adjusted == [0.03, 0.06, 0.06]


def test_onnx_preprocessing_uses_bilinear_resize_like_torchvision(tmp_path: Path) -> None:
    from autodri.workflows.aoi_equivalence import _load_normalized_image

    image_path = tmp_path / "tiny.png"
    image = Image.new("RGB", (2, 2))
    image.putdata([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
    image.save(image_path)

    actual = _load_normalized_image(image_path, 4)
    resized = np.asarray(image.resize((4, 4), resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    expected = np.transpose(
        (resized - np.asarray([0.485, 0.456, 0.406], dtype=np.float32))
        / np.asarray([0.229, 0.224, 0.225], dtype=np.float32),
        (2, 0, 1),
    )

    assert np.allclose(actual, expected)


def test_yolo_onnx_preprocessing_uses_ultralytics_classification_normalization(tmp_path: Path) -> None:
    from autodri.workflows.aoi_equivalence import _load_normalized_image
    from torchvision import transforms

    image_path = tmp_path / "wide.png"
    image = Image.new("RGB", (4, 2))
    image.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (64, 64, 64),
            (192, 192, 192),
        ]
    )
    image.save(image_path)

    actual = _load_normalized_image(image_path, 2, preprocess="yolo-cls")
    expected = transforms.Compose(
        [
            transforms.Resize(2, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((2, 2)),
            transforms.ToTensor(),
        ]
    )(image).numpy()

    assert actual.shape == (3, 2, 2)
    assert np.allclose(actual, expected)
