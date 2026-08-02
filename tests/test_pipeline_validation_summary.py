from __future__ import annotations

import csv
from pathlib import Path

from autodri.workflows.pipeline_validation_summary import (
    PipelineValidationInputs,
    build_pipeline_validation_summary,
    summarize_gaze_finetune_validation,
    summarize_hand_validation,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_summarizes_gaze_finetune_validation_from_current_models(tmp_path: Path) -> None:
    participants_csv = tmp_path / "reports" / "participants_results_summary.current.csv"
    runs_root = tmp_path / "runs_cls"
    datasets_root = tmp_path / "datasets"

    _write_csv(
        participants_csv,
        [
            {
                "participant": "p1",
                "current_model": "/workspace/models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx",
                "gaze_segments_done": "2",
                "n_windows": "10",
                "coverage_ok_windows": "9",
                "coverage_fail_windows": "1",
            },
            {
                "participant": "p2",
                "current_model": "/workspace/models/gaze_cls_p2_200shot_driveonly_ft_v1.onnx",
                "gaze_segments_done": "3",
                "n_windows": "12",
                "coverage_ok_windows": "12",
                "coverage_fail_windows": "0",
            },
            {
                "participant": "p10",
                "current_model": "models/gaze_cls_yolov8n.onnx",
                "gaze_segments_done": "1",
                "n_windows": "5",
                "coverage_ok_windows": "5",
                "coverage_fail_windows": "0",
            },
        ],
    )
    _write_csv(
        runs_root / "gaze_p1_200shot_driveonly_ft_v1_gpu1" / "results.csv",
        [
            {"epoch": 1, "metrics/accuracy_top1": "0.5"},
            {"epoch": 2, "metrics/accuracy_top1": "0.8"},
        ],
    )
    _write_csv(
        runs_root / "gaze_p2_200shot_driveonly_ft_v1_gpu2" / "results.csv",
        [
            {"epoch": 1, "metrics/accuracy_top1": "0.6"},
            {"epoch": 2, "metrics/accuracy_top1": "0.9"},
        ],
    )
    _write_csv(
        datasets_root / "cls_dataset_p1_200shot_driveonly_v1" / "split_manifest.csv",
        [{"split": "train"}, {"split": "val"}, {"split": "val"}, {"split": "val"}, {"split": "val"}, {"split": "val"}],
    )
    _write_csv(
        datasets_root / "cls_dataset_p2_200shot_driveonly_v1" / "split_manifest.csv",
        [
            {"split": "train"},
            {"split": "val"},
            {"split": "val"},
            {"split": "val"},
            {"split": "val"},
            {"split": "val"},
            {"split": "val"},
            {"split": "val"},
            {"split": "val"},
            {"split": "val"},
            {"split": "val"},
        ],
    )

    summary = summarize_gaze_finetune_validation(participants_csv, runs_root, datasets_root)

    assert summary.aggregate["models_with_finetune_validation"] == "2"
    assert summary.aggregate["validation_samples"] == "15"
    assert summary.aggregate["validation_correct_estimate"] == "13"
    assert summary.aggregate["weighted_best_top1"] == "0.8667"
    assert summary.aggregate["mean_best_top1"] == "0.8500"
    assert [row["status"] for row in summary.rows] == ["ok", "ok", "not_participant_finetune_model"]


def test_summarizes_hand_validation_reviews_and_distillation_predictions(tmp_path: Path) -> None:
    agreement_csv = tmp_path / "wheel_validation_agreement.current.csv"
    second_pass_csv = tmp_path / "wheel_validation_p11plus_confirmed_20260524_summary.csv"
    distill_csv = tmp_path / "distill_predictions.csv"

    _write_csv(
        agreement_csv,
        [
            {
                "joined_rows": "40",
                "reviewed_rows": "24",
                "agreement": "18",
                "agreement_rate": "0.75",
                "false_on": "4",
                "false_off": "2",
            }
        ],
    )
    _write_csv(
        second_pass_csv,
        [
            {
                "scope": "corrected",
                "participants": "7",
                "rows": "21",
                "on_rows": "6",
                "off_rows": "6",
                "uncertain_rows": "9",
                "exact_agreement": "21",
                "exact_agreement_rate": "1.0",
                "binary_rows": "12",
                "false_on": "0",
                "false_off": "0",
            }
        ],
    )
    _write_csv(
        distill_csv,
        [
            {"split": "val", "teacher_state": "ON", "pred_state": "ON", "match": "1"},
            {"split": "val", "teacher_state": "ON", "pred_state": "OFF", "match": "0"},
            {"split": "val", "teacher_state": "OFF", "pred_state": "OFF", "match": "1"},
            {"split": "val", "teacher_state": "OFF", "pred_state": "ON", "match": "0"},
            {"split": "val", "teacher_state": "UNCERTAIN", "pred_state": "UNCERTAIN", "match": "1"},
            {"split": "train", "teacher_state": "ON", "pred_state": "ON", "match": "1"},
        ],
    )

    summary = summarize_hand_validation(agreement_csv, second_pass_csv, distill_csv)

    assert summary.review["initial_review_agreement"] == "18/24"
    assert summary.review["initial_review_agreement_rate"] == "0.7500"
    assert summary.review["second_pass_agreement"] == "21/21"
    assert summary.review["second_pass_agreement_rate"] == "1.0000"
    assert summary.distillation["agreement"] == "3/5"
    assert summary.distillation["agreement_rate"] == "0.6000"
    assert summary.distillation["f1_ON"] == "0.5000"
    assert summary.distillation["f1_OFF"] == "0.5000"
    assert summary.distillation["f1_UNCERTAIN"] == "1.0000"


def test_build_pipeline_validation_summary_writes_csv_and_markdown(tmp_path: Path) -> None:
    participants_csv = tmp_path / "reports" / "participants_results_summary.current.csv"
    wheel_summary_csv = tmp_path / "reports" / "wheel_results_summary.current.csv"
    runs_root = tmp_path / "runs_cls"
    datasets_root = tmp_path / "datasets"
    agreement_csv = tmp_path / "reports" / "wheel_validation_agreement.current.csv"
    second_pass_csv = tmp_path / "reports" / "wheel_validation_p11plus_confirmed_20260524_summary.csv"
    distill_csv = tmp_path / "distill_predictions.csv"
    out_csv = tmp_path / "out" / "pipeline_validation_summary.csv"
    out_md = tmp_path / "out" / "pipeline_validation_summary.md"

    _write_csv(
        participants_csv,
        [
            {
                "participant": "p1",
                "current_model": "/workspace/models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx",
                "gaze_segments_done": "2",
                "n_windows": "10",
                "coverage_ok_windows": "9",
                "coverage_fail_windows": "1",
                "gaze_total_frames": "100",
            }
        ],
    )
    _write_csv(wheel_summary_csv, [{"participant": "p1", "wheel_done": "2", "state_rows": "99"}])
    _write_csv(runs_root / "gaze_p1_200shot_driveonly_ft_v1_gpu1" / "results.csv", [{"epoch": 1, "metrics/accuracy_top1": "1.0"}])
    _write_csv(datasets_root / "cls_dataset_p1_200shot_driveonly_v1" / "split_manifest.csv", [{"split": "val"}])
    _write_csv(
        agreement_csv,
        [{"joined_rows": "1", "reviewed_rows": "1", "agreement": "1", "agreement_rate": "1.0", "false_on": "0", "false_off": "0"}],
    )
    _write_csv(
        second_pass_csv,
        [
            {
                "scope": "corrected",
                "participants": "1",
                "rows": "1",
                "on_rows": "1",
                "off_rows": "0",
                "uncertain_rows": "0",
                "exact_agreement": "1",
                "exact_agreement_rate": "1.0",
                "binary_rows": "1",
                "false_on": "0",
                "false_off": "0",
            }
        ],
    )
    _write_csv(distill_csv, [{"split": "val", "teacher_state": "ON", "pred_state": "ON", "match": "1"}])

    build_pipeline_validation_summary(
        PipelineValidationInputs(
            participants_summary_csv=participants_csv,
            wheel_summary_csv=wheel_summary_csv,
            gaze_runs_root=runs_root,
            gaze_datasets_root=datasets_root,
            wheel_agreement_csv=agreement_csv,
            wheel_second_pass_summary_csv=second_pass_csv,
            wheel_distill_predictions_csv=distill_csv,
            out_csv=out_csv,
            out_md=out_md,
        )
    )

    assert out_csv.exists()
    assert out_md.exists()
    text = out_md.read_text(encoding="utf-8")
    assert "Gaze fine-tuned validation" in text
    assert "weighted best top-1: 1.0000" in text
    assert "Hand-on-wheel validation" in text
