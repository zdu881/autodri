#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summarize available validation evidence for the gaze/wheel pipeline.

The generated report deliberately separates model validation, extraction
coverage, human review agreement, and teacher-student agreement. The repository
does not currently contain human gold-standard labels for final 20 s behavioral
window metrics, so this workflow does not report a fabricated end-to-end
behavioral accuracy.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from autodri.common.paths import artifacts_root, reports_root, repo_root, resolve_workspace_or_repo_path


STATE_ORDER = ("OFF", "ON", "UNCERTAIN")


@dataclass(frozen=True)
class GazeValidationSummary:
    rows: list[dict[str, str]]
    aggregate: dict[str, str]


@dataclass(frozen=True)
class HandValidationSummary:
    review: dict[str, str]
    distillation: dict[str, str]
    initial_distillation: dict[str, str]


@dataclass(frozen=True)
class PipelineValidationInputs:
    participants_summary_csv: Path
    wheel_summary_csv: Path
    gaze_runs_root: Path
    gaze_datasets_root: Path
    wheel_agreement_csv: Path
    wheel_second_pass_summary_csv: Path
    wheel_distill_predictions_csv: Path
    out_csv: Path
    out_md: Path
    wheel_initial_distill_predictions_csv: Path | None = None
    window_metrics_summary_csv: Path | None = None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _as_int(raw: object, default: int = 0) -> int:
    try:
        text = str(raw).strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def _as_float(raw: object, default: float = 0.0) -> float:
    try:
        text = str(raw).strip()
        return float(text) if text else default
    except Exception:
        return default


def _fmt4(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _safe_rate(num: int, den: int) -> str:
    return _fmt4(num / den) if den else ""


def _model_token(participant: str, current_model: str) -> str | None:
    name = Path(str(current_model).strip()).name
    if not name.startswith("gaze_cls_") or not name.endswith(".onnx"):
        return None
    token = name.removeprefix("gaze_cls_").removesuffix(".onnx")
    if not token.startswith(f"{participant}_"):
        return None
    return token


def _dataset_token_from_model_token(token: str) -> str:
    return token.replace("_ft_", "_")


def _find_results_csv(runs_root: Path, token: str) -> Path | None:
    candidates = sorted(runs_root.glob(f"gaze_{token}*/results.csv"))
    return candidates[0] if candidates else None


def _find_manifest_csv(datasets_root: Path, token: str) -> Path | None:
    direct = datasets_root / f"cls_dataset_{_dataset_token_from_model_token(token)}" / "split_manifest.csv"
    if direct.exists():
        return direct
    participant = token.split("_", 1)[0]
    candidates = sorted(datasets_root.glob(f"cls_dataset_{participant}_*/split_manifest.csv"))
    return candidates[0] if candidates else None


def _best_and_final_top1(results_csv: Path) -> tuple[float | None, float | None]:
    values: list[float] = []
    for row in read_csv(results_csv):
        if "metrics/accuracy_top1" not in row:
            continue
        values.append(_as_float(row["metrics/accuracy_top1"], default=-1.0))
    values = [v for v in values if v >= 0.0]
    if not values:
        return None, None
    return max(values), values[-1]


def _count_val_samples(manifest_csv: Path) -> int:
    return sum(1 for row in read_csv(manifest_csv) if str(row.get("split", "")).strip().lower() == "val")


def summarize_gaze_finetune_validation(
    participants_summary_csv: Path,
    runs_root: Path,
    datasets_root: Path,
) -> GazeValidationSummary:
    rows: list[dict[str, str]] = []
    best_values: list[float] = []
    total_val = 0
    total_correct = 0

    for participant_row in read_csv(participants_summary_csv):
        participant = str(participant_row.get("participant", "")).strip()
        current_model = str(participant_row.get("current_model", "")).strip()
        row = {
            "participant": participant,
            "current_model": current_model,
            "run_results_csv": "",
            "dataset_manifest_csv": "",
            "val_n": "",
            "best_top1": "",
            "final_top1": "",
            "best_correct_estimate": "",
            "status": "",
        }

        token = _model_token(participant, current_model)
        if token is None:
            row["status"] = "not_participant_finetune_model"
            rows.append(row)
            continue

        results_csv = _find_results_csv(runs_root, token)
        manifest_csv = _find_manifest_csv(datasets_root, token)
        if results_csv is None:
            row["status"] = "missing_results_csv"
            rows.append(row)
            continue
        if manifest_csv is None:
            row["run_results_csv"] = str(results_csv)
            row["status"] = "missing_split_manifest"
            rows.append(row)
            continue

        best, final = _best_and_final_top1(results_csv)
        val_n = _count_val_samples(manifest_csv)
        if best is None or final is None or val_n == 0:
            row.update(
                {
                    "run_results_csv": str(results_csv),
                    "dataset_manifest_csv": str(manifest_csv),
                    "val_n": str(val_n) if val_n else "",
                    "status": "missing_validation_metric",
                }
            )
            rows.append(row)
            continue

        correct = int(round(best * val_n))
        total_val += val_n
        total_correct += correct
        best_values.append(best)
        row.update(
            {
                "run_results_csv": str(results_csv),
                "dataset_manifest_csv": str(manifest_csv),
                "val_n": str(val_n),
                "best_top1": _fmt4(best),
                "final_top1": _fmt4(final),
                "best_correct_estimate": str(correct),
                "status": "ok",
            }
        )
        rows.append(row)

    aggregate = {
        "models_with_finetune_validation": str(len(best_values)),
        "participants_in_summary": str(len(rows)),
        "validation_samples": str(total_val),
        "validation_correct_estimate": str(total_correct),
        "weighted_best_top1": _safe_rate(total_correct, total_val),
        "mean_best_top1": _fmt4(statistics.fmean(best_values)) if best_values else "",
        "median_best_top1": _fmt4(statistics.median(best_values)) if best_values else "",
        "min_best_top1": _fmt4(min(best_values)) if best_values else "",
        "max_best_top1": _fmt4(max(best_values)) if best_values else "",
        "participants_without_finetune_validation": str(sum(1 for row in rows if row["status"] != "ok")),
    }
    return GazeValidationSummary(rows=rows, aggregate=aggregate)


def _count_wheel_map_outputs(window_summary_csv: Path | None) -> dict[str, int]:
    if window_summary_csv is None or not window_summary_csv.exists():
        return {}

    planned = 0
    existing = 0
    state_rows = 0
    seen_maps: set[Path] = set()
    for summary_row in read_csv(window_summary_csv):
        raw_map = str(summary_row.get("wheel_map_csv", "")).strip()
        if not raw_map:
            continue
        map_csv = Path(raw_map).expanduser()
        if not map_csv.is_absolute():
            map_csv = resolve_workspace_or_repo_path(raw_map)
        if map_csv in seen_maps or not map_csv.exists():
            continue
        seen_maps.add(map_csv)
        for map_row in read_csv(map_csv):
            raw_wheel = str(map_row.get("wheel_csv", "")).strip()
            if not raw_wheel:
                continue
            planned += 1
            wheel_csv = Path(raw_wheel).expanduser()
            if not wheel_csv.is_absolute():
                wheel_csv = resolve_workspace_or_repo_path(raw_wheel)
            if not wheel_csv.exists():
                continue
            existing += 1
            with wheel_csv.open("r", encoding="utf-8-sig", newline="") as f:
                state_rows += max(sum(1 for _ in f) - 1, 0)
    return {
        "wheel_segments_planned_from_maps": planned,
        "wheel_segments_done": existing,
        "wheel_state_rows": state_rows,
    }


def summarize_extraction_coverage(
    participants_summary_csv: Path,
    wheel_summary_csv: Path,
    window_summary_csv: Path | None = None,
) -> dict[str, str]:
    participant_rows = read_csv(participants_summary_csv)
    wheel_rows = read_csv(wheel_summary_csv)
    window_rows = read_csv(window_summary_csv) if window_summary_csv else []
    gaze_segments = sum(_as_int(row.get("gaze_segments_done")) for row in participant_rows)
    gaze_frames = sum(_as_int(row.get("gaze_total_frames")) for row in participant_rows)
    coverage_source = window_rows if window_rows else participant_rows
    ok_windows = sum(_as_int(row.get("coverage_ok_windows")) for row in coverage_source)
    fail_windows = sum(_as_int(row.get("coverage_fail_windows")) for row in coverage_source)
    zero_windows = sum(_as_int(row.get("coverage_zero_windows")) for row in coverage_source)
    generated_windows = sum(_as_int(row.get("windows_total")) for row in window_rows)

    wheel_map_counts = _count_wheel_map_outputs(window_summary_csv)
    wheel_segments = wheel_map_counts.get("wheel_segments_done", sum(_as_int(row.get("wheel_done")) for row in wheel_rows))
    wheel_rows_total = wheel_map_counts.get("wheel_state_rows", sum(_as_int(row.get("state_rows")) for row in wheel_rows))
    teacher_det_done = sum(_as_int(row.get("teacher_det_done")) for row in wheel_rows)

    return {
        "participants": str(len(participant_rows)),
        "gaze_segments_done": str(gaze_segments),
        "gaze_frame_rows": str(gaze_frames),
        "generated_windows": str(generated_windows) if generated_windows else "",
        "coverage_ok_windows": str(ok_windows),
        "coverage_fail_windows": str(fail_windows),
        "coverage_zero_windows": str(zero_windows),
        "coverage_ok_rate": _safe_rate(ok_windows, ok_windows + fail_windows),
        "wheel_segments_planned_from_maps": str(wheel_map_counts.get("wheel_segments_planned_from_maps", "")),
        "wheel_segments_done": str(wheel_segments),
        "wheel_state_rows": str(wheel_rows_total),
        "wheel_teacher_det_done": str(teacher_det_done),
    }


def _first_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return rows[0] if rows else {}


def _state(text: object) -> str:
    value = str(text).strip().upper().replace("-", "_")
    if value in {"HAND_ON_WHEEL", "TRUE", "1"}:
        return "ON"
    if value in {"HAND_OFF_WHEEL", "FALSE", "0"}:
        return "OFF"
    if "UNCERTAIN" in value:
        return "UNCERTAIN"
    return value


def summarize_distillation_predictions(path: Path) -> dict[str, str]:
    rows = [row for row in read_csv(path) if str(row.get("split", "val")).strip().lower() == "val"]
    if not rows:
        return {
            "prediction_csv": str(path),
            "validation_rows": "0",
            "agreement": "",
            "agreement_rate": "",
        }

    labels = list(STATE_ORDER)
    actual = [_state(row.get("teacher_state", "")) for row in rows]
    pred = [_state(row.get("pred_state", "")) for row in rows]
    correct = sum(1 for a, p in zip(actual, pred) if a == p)
    out = {
        "prediction_csv": str(path),
        "validation_rows": str(len(rows)),
        "agreement": f"{correct}/{len(rows)}",
        "agreement_rate": _safe_rate(correct, len(rows)),
    }

    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for a, p in zip(actual, pred) if a == label and p == label)
        fp = sum(1 for a, p in zip(actual, pred) if a != label and p == label)
        fn = sum(1 for a, p in zip(actual, pred) if a == label and p != label)
        support = sum(1 for a in actual if a == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        out[f"support_{label}"] = str(support)
        out[f"precision_{label}"] = _fmt4(precision)
        out[f"recall_{label}"] = _fmt4(recall)
        out[f"f1_{label}"] = _fmt4(f1)
    out["macro_f1"] = _fmt4(statistics.fmean(f1_values))
    return out


def summarize_hand_validation(
    agreement_csv: Path,
    second_pass_summary_csv: Path,
    distill_predictions_csv: Path,
    initial_distill_predictions_csv: Path | None = None,
) -> HandValidationSummary:
    agreement = _first_row(agreement_csv)
    second_pass = _first_row(second_pass_summary_csv)

    initial_reviewed = _as_int(agreement.get("reviewed_rows"))
    initial_agreement = _as_int(agreement.get("agreement"))
    second_reviewed = _as_int(second_pass.get("rows"))
    second_agreement = _as_int(second_pass.get("exact_agreement"))

    review = {
        "initial_review_joined_rows": str(_as_int(agreement.get("joined_rows"))),
        "initial_review_reviewed_rows": str(initial_reviewed),
        "initial_review_agreement": f"{initial_agreement}/{initial_reviewed}" if initial_reviewed else "",
        "initial_review_agreement_rate": _safe_rate(initial_agreement, initial_reviewed),
        "initial_review_false_on": str(_as_int(agreement.get("false_on"))),
        "initial_review_false_off": str(_as_int(agreement.get("false_off"))),
        "second_pass_scope": str(second_pass.get("scope", "")),
        "second_pass_participants": str(_as_int(second_pass.get("participants"))),
        "second_pass_rows": str(second_reviewed),
        "second_pass_binary_rows": str(_as_int(second_pass.get("binary_rows"))),
        "second_pass_agreement": f"{second_agreement}/{second_reviewed}" if second_reviewed else "",
        "second_pass_agreement_rate": _safe_rate(second_agreement, second_reviewed),
        "second_pass_false_on": str(_as_int(second_pass.get("false_on"))),
        "second_pass_false_off": str(_as_int(second_pass.get("false_off"))),
    }
    initial_distillation = (
        summarize_distillation_predictions(initial_distill_predictions_csv) if initial_distill_predictions_csv else {}
    )
    return HandValidationSummary(
        review=review,
        distillation=summarize_distillation_predictions(distill_predictions_csv),
        initial_distillation=initial_distillation,
    )


def _metric_rows(scope: str, metrics: Mapping[str, str]) -> list[dict[str, str]]:
    return [{"scope": scope, "metric": key, "value": value} for key, value in metrics.items()]


def _write_summary_csv(
    path: Path,
    gaze: GazeValidationSummary,
    coverage: Mapping[str, str],
    hand: HandValidationSummary,
) -> None:
    rows: list[dict[str, str]] = []
    rows.extend(_metric_rows("gaze_finetune_validation", gaze.aggregate))
    rows.extend(_metric_rows("extraction_coverage", coverage))
    rows.extend(_metric_rows("hand_review", hand.review))
    if hand.initial_distillation:
        rows.extend(_metric_rows("hand_initial_teacher_student", hand.initial_distillation))
    rows.extend(_metric_rows("hand_clean_teacher_student", hand.distillation))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _md_table(rows: Iterable[Mapping[str, str]], columns: list[str]) -> str:
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(col, "")) for col in columns) + "|")
    return "\n".join(lines)


def _write_markdown(
    path: Path,
    gaze: GazeValidationSummary,
    coverage: Mapping[str, str],
    hand: HandValidationSummary,
) -> None:
    ok_rows = [row for row in gaze.rows if row["status"] == "ok"]
    missing_rows = [row for row in gaze.rows if row["status"] != "ok"]
    lines = [
        "# Pipeline Validation Summary",
        "",
        "This report summarizes available quantitative evidence for the current gaze and hand-on-wheel pipeline. "
        "It is not an end-to-end accuracy estimate for final 20 s behavioral metrics because the current artifacts "
        "do not include human gold-standard labels for those derived window metrics.",
        "",
        "## Gaze fine-tuned validation",
        "",
        f"- models with participant-specific validation: {gaze.aggregate['models_with_finetune_validation']}/"
        f"{gaze.aggregate['participants_in_summary']}",
        f"- validation samples: {gaze.aggregate['validation_samples']}",
        f"- weighted best top-1: {gaze.aggregate['weighted_best_top1']}",
        f"- mean best top-1: {gaze.aggregate['mean_best_top1']}",
        f"- range: {gaze.aggregate['min_best_top1']} to {gaze.aggregate['max_best_top1']}",
        "",
    ]
    if ok_rows:
        lines.extend(
            [
                _md_table(
                    ok_rows,
                    ["participant", "val_n", "best_top1", "final_top1", "best_correct_estimate"],
                ),
                "",
            ]
        )
    if missing_rows:
        lines.extend(
            [
                "Models or participants not included in that aggregate:",
                "",
                _md_table(missing_rows, ["participant", "current_model", "status"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Extraction coverage",
            "",
            _md_table(_metric_rows("coverage", coverage), ["metric", "value"]),
            "",
            "## Hand-on-wheel validation",
            "",
            f"- initial final-state review agreement: {hand.review['initial_review_agreement']} "
            f"({hand.review['initial_review_agreement_rate']}); false ON/OFF: "
            f"{hand.review['initial_review_false_on']}/{hand.review['initial_review_false_off']}",
            f"- corrected second-pass agreement: {hand.review['second_pass_agreement']} "
            f"({hand.review['second_pass_agreement_rate']}); binary rows: "
            f"{hand.review['second_pass_binary_rows']}; false ON/OFF: "
            f"{hand.review['second_pass_false_on']}/{hand.review['second_pass_false_off']}",
            "",
        ]
    )
    if hand.initial_distillation:
        lines.extend(
            [
                "Initial teacher-student state agreement:",
                "",
                _md_table(_metric_rows("initial_teacher_student", hand.initial_distillation), ["metric", "value"]),
                "",
            ]
        )
    lines.extend(
        [
            "Clean teacher-student state agreement:",
            "",
            _md_table(_metric_rows("clean_teacher_student", hand.distillation), ["metric", "value"]),
            "",
            "## Recommended wording",
            "",
            "The current artifacts support component-level validation rather than an end-to-end behavioral-metric "
            "accuracy claim. We therefore report participant-specific gaze validation accuracy, processing coverage, "
            "hand-state review agreement, and hand teacher-student agreement as separate reliability checks.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_pipeline_validation_summary(inputs: PipelineValidationInputs) -> tuple[Path, Path]:
    gaze = summarize_gaze_finetune_validation(
        inputs.participants_summary_csv,
        inputs.gaze_runs_root,
        inputs.gaze_datasets_root,
    )
    coverage = summarize_extraction_coverage(
        inputs.participants_summary_csv,
        inputs.wheel_summary_csv,
        inputs.window_metrics_summary_csv,
    )
    hand = summarize_hand_validation(
        inputs.wheel_agreement_csv,
        inputs.wheel_second_pass_summary_csv,
        inputs.wheel_distill_predictions_csv,
        inputs.wheel_initial_distill_predictions_csv,
    )
    _write_summary_csv(inputs.out_csv, gaze, coverage, hand)
    _write_markdown(inputs.out_md, gaze, coverage, hand)
    return inputs.out_csv, inputs.out_md


def default_inputs() -> PipelineValidationInputs:
    reports = reports_root(create=True)
    artifacts = artifacts_root()
    return PipelineValidationInputs(
        participants_summary_csv=reports / "participants_results_summary.current.csv",
        wheel_summary_csv=reports / "wheel_results_summary.current.csv",
        gaze_runs_root=artifacts / "runs" / "classify" / "gaze_onnx" / "experiments" / "runs_cls",
        gaze_datasets_root=repo_root() / "gaze_onnx" / "experiments",
        wheel_agreement_csv=reports / "wheel_validation_agreement.current.csv",
        wheel_second_pass_summary_csv=reports / "wheel_validation_p11plus_confirmed_20260524_summary.csv",
        wheel_initial_distill_predictions_csv=artifacts
        / "wheel_state_distill_20260603"
        / "yolov8n_cls_state_hash_codex_verify_val_predictions.csv",
        wheel_distill_predictions_csv=artifacts
        / "wheel_state_distill_on_enriched_20260608"
        / "yolov8s_contact_strict_2500_320_margin05_retrained_val_predictions.csv",
        out_csv=reports / "pipeline_validation_summary.current.csv",
        out_md=reports / "pipeline_validation_summary.current.md",
        window_metrics_summary_csv=reports / "all_participants_window_metrics_summary.current.csv",
    )


def parse_args() -> argparse.Namespace:
    defaults = default_inputs()
    p = argparse.ArgumentParser(description="Build a validation summary for current gaze and hand-on-wheel artifacts")
    p.add_argument("--participants-summary-csv", default=str(defaults.participants_summary_csv))
    p.add_argument("--wheel-summary-csv", default=str(defaults.wheel_summary_csv))
    p.add_argument("--gaze-runs-root", default=str(defaults.gaze_runs_root))
    p.add_argument("--gaze-datasets-root", default=str(defaults.gaze_datasets_root))
    p.add_argument("--wheel-agreement-csv", default=str(defaults.wheel_agreement_csv))
    p.add_argument("--wheel-second-pass-summary-csv", default=str(defaults.wheel_second_pass_summary_csv))
    p.add_argument("--wheel-initial-distill-predictions-csv", default=str(defaults.wheel_initial_distill_predictions_csv))
    p.add_argument("--wheel-distill-predictions-csv", default=str(defaults.wheel_distill_predictions_csv))
    p.add_argument("--window-metrics-summary-csv", default=str(defaults.window_metrics_summary_csv))
    p.add_argument("--out-csv", default=str(defaults.out_csv))
    p.add_argument("--out-md", default=str(defaults.out_md))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_csv, out_md = build_pipeline_validation_summary(
        PipelineValidationInputs(
            participants_summary_csv=Path(args.participants_summary_csv).expanduser(),
            wheel_summary_csv=Path(args.wheel_summary_csv).expanduser(),
            gaze_runs_root=Path(args.gaze_runs_root).expanduser(),
            gaze_datasets_root=Path(args.gaze_datasets_root).expanduser(),
            wheel_agreement_csv=Path(args.wheel_agreement_csv).expanduser(),
            wheel_second_pass_summary_csv=Path(args.wheel_second_pass_summary_csv).expanduser(),
            wheel_initial_distill_predictions_csv=Path(args.wheel_initial_distill_predictions_csv).expanduser(),
            wheel_distill_predictions_csv=Path(args.wheel_distill_predictions_csv).expanduser(),
            out_csv=Path(args.out_csv).expanduser(),
            out_md=Path(args.out_md).expanduser(),
            window_metrics_summary_csv=Path(args.window_metrics_summary_csv).expanduser(),
        )
    )
    print(f"out_csv={out_csv}")
    print(f"out_md={out_md}")


if __name__ == "__main__":
    main()
