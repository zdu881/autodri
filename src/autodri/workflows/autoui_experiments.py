from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import mimetypes
import os
import random
import shlex
import shutil
import statistics
import subprocess
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

from autodri.aoi.equivalence import (
    PRIMARY_LABELS,
    assign_internal_validation,
    load_split_manifest,
    write_csv_rows,
)
from autodri.aoi.participant_lopo import build_lopo_dataset
from autodri.common.paths import workspace_root

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - only needed for review-frame rendering.
    cv2 = None
    np = None


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = workspace_root()
DEFAULT_AUTOWIP_ROOT = WORKSPACE_ROOT / "artifacts" / "autoui_wip_experiments_20260523"
DEFAULT_MULTI_SEED_ROOT = DEFAULT_AUTOWIP_ROOT / "lopo_all14_multiseed_primary3"
DEFAULT_FEWSHOT_ROOT = DEFAULT_AUTOWIP_ROOT / "fewshot_participant_curve_primary3"
DEFAULT_REPORT_ROOT = WORKSPACE_ROOT / "artifacts" / "reports"
DEFAULT_WHEEL_DET_ROOT = WORKSPACE_ROOT / "data" / "wheel_teacher_det"
WHEEL_REVIEW_FRAME_VERSION = "v4_wheel_candidate"
SUSPECT_WHEEL_ROI_RULES = {"single_panel_same_as_gaze"}
SUSPECT_WHEEL_ROI_PARTICIPANTS = {"p11", "p13", "p14", "p15", "p16", "p17", "p18"}
WHEEL_EVIDENCE_CANDIDATE_ROIS = {
    "p11": (0, 0, 720, 540),
    "p13": (0, 0, 960, 700),
    "p14": (0, 0, 960, 700),
    "p15": (0, 0, 960, 700),
    "p16": (0, 0, 960, 700),
    "p17": (0, 0, 960, 700),
    "p18": (0, 0, 960, 700),
}
DEFAULT_SEEDS = (13, 29, 43, 71, 101)
DEFAULT_BUDGETS = (25, 50, 100, 200)
DEFAULT_TARGETS = ("p1", "p2", "p4", "p6", "p7", "p8", "p9", "p11", "p13", "p14", "p15", "p16", "p17", "p18")


@dataclass(frozen=True)
class ExperimentModel:
    model: str
    family: str
    arch_group: str
    trainer: str
    base_model: str


DEFAULT_MODELS = (
    ExperimentModel("yolov8s-cls", "yolo", "yolo", "ultralytics", "yolov8s-cls.pt"),
    ExperimentModel("resnet50", "convnet", "resnet", "torchvision_timm", "resnet50"),
    ExperimentModel("efficientnet_b0", "convnet", "efficientnet", "torchvision_timm", "efficientnet_b0"),
    ExperimentModel("deit_tiny", "vit", "light_vit", "torchvision_timm", "deit_tiny_patch16_224"),
)


def default_participant_datasets(experiments_dir: Path | None = None) -> dict[str, Path]:
    base = experiments_dir or (REPO_ROOT / "gaze_onnx" / "experiments")
    return {
        "p1": base / "cls_dataset_p1_200shot_driveonly_v1",
        "p2": base / "cls_dataset_p2_200shot_driveonly_v1",
        "p4": base / "cls_dataset_p4_200shot_driveonly_v1",
        "p6": base / "cls_dataset_p6_200shot_driveonly_v1",
        "p7": base / "cls_dataset_p7_200shot_driveonly_v1",
        "p8": base / "cls_dataset_p8_200shot_driveonly_v1",
        "p9": base / "cls_dataset_p9_200shot_driveonly_v1",
        "p11": base / "cls_dataset_p11_200shot_driveonly_v3",
        "p13": base / "cls_dataset_p13_200shot_driveonly_v1",
        "p14": base / "cls_dataset_p14_200shot_driveonly_v2_newroi",
        "p15": base / "cls_dataset_p15_200shot_driveonly_v1",
        "p16": base / "cls_dataset_p16_200shot_driveonly_v1",
        "p17": base / "cls_dataset_p17_200shot_driveonly_v1",
        "p18": base / "cls_dataset_p18_200shot_driveonly_v1",
    }


def prepare_lopo_matrix(
    participant_datasets: Mapping[str, Path | str],
    *,
    out_dir: Path | str,
    targets: Sequence[str] = DEFAULT_TARGETS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    labels: Sequence[str] = PRIMARY_LABELS,
    models: Sequence[ExperimentModel] = DEFAULT_MODELS,
    epochs: int = 50,
    batch: int = 32,
    workers: int = 4,
    yolo_aug_preset: str = "baseline",
    no_class_weights: bool = True,
) -> list[dict[str, object]]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    _assert_participant_datasets(participant_datasets)

    support_rows = label_support_rows(participant_datasets)
    write_csv_rows(root / "label_support_summary.csv", support_rows)

    dataset_rows: list[dict[str, object]] = []
    for target in targets:
        if target not in participant_datasets:
            raise ValueError(f"Unknown holdout participant {target!r}; known={sorted(participant_datasets)}")
        for seed in seeds:
            data_dir = root / "lopo_datasets" / f"holdout_{target}_seed{int(seed)}"
            summary = build_lopo_dataset(
                participant_datasets,
                holdout_participant=target,
                out_dir=data_dir,
                labels=labels,
            )
            dataset_rows.append(
                {
                    "dataset": f"lopo_{target}",
                    "target_participant": target,
                    "seed": int(seed),
                    "data_dir": str(data_dir),
                    "source_participants": ",".join(sorted(participant_datasets)),
                    "labels": ",".join(labels),
                    **summary,
                }
            )
    write_csv_rows(root / "lopo_dataset_summary.csv", dataset_rows)

    matrix = _run_matrix_rows(
        dataset_rows,
        root=root,
        models=models,
        labels=labels,
        epochs=epochs,
        batch=batch,
        workers=workers,
        yolo_aug_preset=yolo_aug_preset,
        no_class_weights=no_class_weights,
        name_prefix="",
    )
    write_csv_rows(root / "run_matrix.csv", matrix)
    _write_command_script(root / "commands.sh", [str(row["train_command"]) for row in matrix])
    return matrix


def prepare_fewshot_datasets(
    participant_datasets: Mapping[str, Path | str],
    *,
    out_dir: Path | str,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    labels: Sequence[str] = PRIMARY_LABELS,
    targets: Sequence[str] | None = None,
    models: Sequence[ExperimentModel] = DEFAULT_MODELS,
    internal_val_ratio: float = 0.2,
    epochs: int = 50,
    batch: int = 32,
    workers: int = 4,
    yolo_aug_preset: str = "baseline",
    no_class_weights: bool = True,
) -> list[dict[str, object]]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    _assert_participant_datasets(participant_datasets)
    picked_targets = tuple(targets or sorted(participant_datasets))
    label_set = set(labels)

    support_rows = label_support_rows(participant_datasets)
    write_csv_rows(root / "label_support_summary.csv", support_rows)

    dataset_rows: list[dict[str, object]] = []
    for participant in picked_targets:
        if participant not in participant_datasets:
            raise ValueError(f"Unknown participant {participant!r}; known={sorted(participant_datasets)}")
        source_dir = Path(participant_datasets[participant])
        source_samples = load_split_manifest(source_dir / "split_manifest.csv")
        frozen_test = [
            sample
            for sample in source_samples
            if sample.split == "val" and sample.label in label_set and not sample.augmented
        ]
        frozen_group_keys = {sample.group_key() for sample in frozen_test}
        eligible_train = [
            sample
            for sample in source_samples
            if sample.split == "train"
            and sample.label in label_set
            and not sample.augmented
            and sample.group_key() not in frozen_group_keys
        ]
        if not eligible_train:
            raise ValueError(f"No eligible train samples for {participant}: {source_dir}")
        if not frozen_test:
            raise ValueError(f"No frozen val/test samples for {participant}: {source_dir}")

        for budget in budgets:
            for seed in seeds:
                selected = _stratified_group_sample(
                    eligible_train,
                    n=min(int(budget), len(eligible_train)),
                    seed=int(seed),
                )
                assignment = assign_internal_validation(selected, val_ratio=internal_val_ratio, seed=int(seed))
                assignment = _keep_labels_in_train(selected, assignment, labels)
                data_dir = root / "fewshot_datasets" / f"{participant}_budget{int(budget)}_seed{int(seed)}"
                counts = _materialize_fewshot_dataset(
                    source_dir,
                    data_dir,
                    participant=participant,
                    selected_train=selected,
                    frozen_test=frozen_test,
                    train_assignment=assignment,
                    labels=labels,
                )
                label_counts = Counter(sample.label for sample in selected)
                dataset_rows.append(
                    {
                        "dataset": f"fewshot_{participant}_b{int(budget)}",
                        "participant": participant,
                        "budget": int(budget),
                        "seed": int(seed),
                        "data_dir": str(data_dir),
                        "selected_label_count": len(selected),
                        "available_train_count": len(eligible_train),
                        "frozen_test_count": len(frozen_test),
                        "labels": ",".join(labels),
                        "selected_labels": _format_counter(label_counts),
                        **counts,
                    }
                )
    write_csv_rows(root / "fewshot_dataset_summary.csv", dataset_rows)

    matrix = _run_matrix_rows(
        dataset_rows,
        root=root,
        models=models,
        labels=labels,
        epochs=epochs,
        batch=batch,
        workers=workers,
        yolo_aug_preset=yolo_aug_preset,
        no_class_weights=no_class_weights,
        name_prefix="fewshot_",
    )
    write_csv_rows(root / "run_matrix.csv", matrix)
    _write_command_script(root / "commands.sh", [str(row["train_command"]) for row in matrix])
    return dataset_rows


def label_support_rows(participant_datasets: Mapping[str, Path | str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participant, raw_path in sorted(participant_datasets.items()):
        samples = load_split_manifest(Path(raw_path) / "split_manifest.csv")
        for split in ("train", "val"):
            split_rows = [sample for sample in samples if sample.split == split and not sample.augmented]
            counts = Counter(sample.label for sample in split_rows)
            row: dict[str, object] = {"participant": participant, "split": split, "total": len(split_rows)}
            for label in ("Forward", "In-Car", "Non-Forward", "Other"):
                row[label] = counts.get(label, 0)
            rows.append(row)
    return rows


def run_train_matrix(
    *,
    root: Path | str,
    gpus: Sequence[str],
    matrix_csv: Path | str | None = None,
    keep_going: bool = False,
    status_csv: Path | str | None = None,
) -> int:
    root_path = Path(root)
    matrix_path = Path(matrix_csv) if matrix_csv else root_path / "run_matrix.csv"
    status_path = Path(status_csv) if status_csv else root_path / "train_status.csv"
    rows = _read_csv(matrix_path)
    completed = _load_success_names(status_path)
    pending = deque((idx, row) for idx, row in enumerate(rows, 1) if row["run_name"] not in completed)
    gpu_list = [str(gpu) for gpu in gpus] or ["cpu"]
    print(f"[{_now()}] train matrix={len(rows)} completed={len(completed)} pending={len(pending)} gpus={','.join(gpu_list)}", flush=True)

    failed = False
    futures = {}
    with ThreadPoolExecutor(max_workers=len(gpu_list)) as pool:
        for gpu in gpu_list:
            if not pending:
                break
            idx, row = pending.popleft()
            print(f"[{_now()}] start {idx}/{len(rows)} {row['run_name']} gpu={gpu}", flush=True)
            futures[pool.submit(_run_training_one, root_path, idx, row, gpu)] = gpu
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                gpu = futures.pop(future)
                result = future.result()
                _append_csv(status_path, result, STATUS_FIELDS)
                print(
                    f"[{result['end_time']}] {result['status']} {result['index']}/{len(rows)} "
                    f"{result['run_name']} gpu={gpu} rc={result['returncode']}",
                    flush=True,
                )
                if int(result["returncode"]) != 0:
                    failed = True
                if pending and (keep_going or not failed):
                    idx, row = pending.popleft()
                    print(f"[{_now()}] start {idx}/{len(rows)} {row['run_name']} gpu={gpu}", flush=True)
                    futures[pool.submit(_run_training_one, root_path, idx, row, gpu)] = gpu
    return 1 if failed else 0


def run_predict_matrix(
    *,
    root: Path | str,
    gpus: Sequence[str] = (),
    matrix_csv: Path | str | None = None,
    keep_going: bool = False,
    status_csv: Path | str | None = None,
) -> int:
    root_path = Path(root)
    matrix_path = Path(matrix_csv) if matrix_csv else root_path / "run_matrix.csv"
    status_path = Path(status_csv) if status_csv else root_path / "predict_status.csv"
    rows = _read_csv(matrix_path)
    completed = _load_success_names(status_path)
    train_success = _load_success_names(root_path / "train_status.csv")
    pred_dir = root_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pending = deque(
        (idx, row)
        for idx, row in enumerate(rows, 1)
        if row["run_name"] in train_success
        and row["run_name"] not in completed
        and not (pred_dir / f"{row['run_name']}.csv").exists()
    )
    devices = [str(gpu) for gpu in gpus] or ["cpu"]
    print(
        f"[{_now()}] predict matrix={len(rows)} train_success={len(train_success)} "
        f"completed={len(completed)} pending={len(pending)} devices={','.join(devices)}",
        flush=True,
    )

    failed = False
    futures = {}
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for device in devices:
            if not pending:
                break
            idx, row = pending.popleft()
            futures[pool.submit(_run_predict_one, root_path, idx, row, device)] = device
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                device = futures.pop(future)
                result = future.result()
                _append_csv(status_path, result, STATUS_FIELDS)
                print(
                    f"[{result['end_time']}] {result['status']} {result['index']}/{len(rows)} "
                    f"{result['run_name']} device={device} rc={result['returncode']}",
                    flush=True,
                )
                if int(result["returncode"]) != 0:
                    failed = True
                if pending and (keep_going or not failed):
                    idx, row = pending.popleft()
                    futures[pool.submit(_run_predict_one, root_path, idx, row, device)] = device
    return 1 if failed else 0


def run_stats(root: Path | str, *, n_boot: int = 1000, metric: str = "primary3_macro_f1") -> None:
    root_path = Path(root)
    pred_files = sorted(str(path) for path in (root_path / "predictions").glob("*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction CSV files under {root_path / 'predictions'}")
    _run_checked(
        [
            sys.executable,
            "-m",
            "autodri.cli.aoi_equivalence",
            "metrics",
            "--predictions",
            *pred_files,
            "--out-dir",
            str(root_path / "eval"),
        ],
        timeout=600,
    )
    _run_checked(
        [
            sys.executable,
            "-m",
            "autodri.cli.aoi_equivalence",
            "equivalence",
            "--predictions",
            *pred_files,
            "--out-dir",
            str(root_path / "stats"),
            "--metric",
            metric,
            "--delta-margin",
            "0.03",
            "--n-boot",
            str(int(n_boot)),
            "--seed",
            "42",
        ],
        timeout=1800,
    )
    _write_tost_ci_summary(root_path / "stats" / "equivalence_results.csv", root_path / "stats" / "tost_ci_summary.csv")


def write_matrix_report(root: Path | str, *, title: str = "AutoUI AOI Experiment") -> dict[str, Path]:
    root_path = Path(root)
    metrics_path = root_path / "eval" / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = _read_csv(metrics_path)
    by_model: dict[str, list[float]] = defaultdict(list)
    by_dataset_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in metrics:
        value = _safe_float(row.get("primary3_macro_f1", ""))
        by_model[row["model"]].append(value)
        by_dataset_model[(row["dataset"], row["model"])].append(value)

    summary_rows = []
    for model, values in sorted(by_model.items(), key=lambda item: -statistics.fmean(item[1])):
        summary_rows.append(
            {
                "model": model,
                "runs": len(values),
                "primary3_macro_f1_mean": f"{statistics.fmean(values):.6f}",
                "primary3_macro_f1_std": f"{statistics.pstdev(values):.6f}" if len(values) > 1 else "0.000000",
            }
        )
    model_summary = root_path / "paper_model_summary.csv"
    write_csv_rows(model_summary, summary_rows)

    dataset_rows = []
    for (dataset, model), values in sorted(by_dataset_model.items()):
        dataset_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "runs": len(values),
                "primary3_macro_f1_mean": f"{statistics.fmean(values):.6f}",
            }
        )
    dataset_summary = root_path / "paper_dataset_model_summary.csv"
    write_csv_rows(dataset_summary, dataset_rows)

    tex_path = root_path / "paper_model_summary.tex"
    tex_lines = [
        "\\begin{tabular}{lrr}\n",
        "\\toprule\n",
        "Model & Runs & Primary3 Macro-F1 \\\\\n",
        "\\midrule\n",
    ]
    for row in summary_rows:
        tex_lines.append(
            f"{_latex_escape(str(row['model']))} & {row['runs']} & "
            f"{float(row['primary3_macro_f1_mean']):.3f} $\\pm$ {float(row['primary3_macro_f1_std']):.3f} \\\\\n"
        )
    tex_lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    tex_path.write_text("".join(tex_lines), encoding="utf-8")

    report_path = root_path / "autoui_experiment_report.md"
    lines = [f"# {title}\n\n", f"- Metrics: `{metrics_path}`\n", f"- Model summary: `{model_summary}`\n"]
    conclusion = root_path / "stats" / "conclusion.md"
    if conclusion.exists():
        lines.extend(["\n## Equivalence Conclusion\n\n", conclusion.read_text(encoding="utf-8")])
    report_path.write_text("".join(lines), encoding="utf-8")
    return {"model_summary_csv": model_summary, "dataset_summary_csv": dataset_summary, "tex": tex_path, "report": report_path}


def write_fewshot_curve_report(root: Path | str) -> dict[str, Path]:
    root_path = Path(root)
    metrics_path = root_path / "eval" / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(metrics_path):
        budget = _budget_from_row(row)
        if budget is None:
            continue
        grouped[(budget, row["model"])].append(row)

    summary_rows = []
    for (budget, model), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "budget": budget,
                "model": model,
                "runs": len(rows),
                "primary3_macro_f1_mean": f"{_mean_metric(rows, 'primary3_macro_f1'):.6f}",
                "primary3_macro_f1_std": f"{_std_metric(rows, 'primary3_macro_f1'):.6f}",
                "primary3_balanced_accuracy_mean": f"{_mean_metric(rows, 'primary3_balanced_accuracy'):.6f}",
                "primary3_event_acc_mean": f"{_mean_metric(rows, 'primary3_event_acc'):.6f}",
            }
        )

    summary_csv = root_path / "fewshot_curve_summary.csv"
    write_csv_rows(summary_csv, summary_rows)
    tex_path = root_path / "fewshot_curve_summary.tex"
    _write_fewshot_curve_tex(tex_path, summary_rows)
    curve_svg = root_path / "fewshot_curve_primary3_macro_f1.svg"
    _write_fewshot_curve_svg(curve_svg, summary_rows)
    return {"summary_csv": summary_csv, "tex": tex_path, "curve_svg": curve_svg}


def run_deployment_matrix(
    *,
    root: Path | str,
    devices: Sequence[str] = ("cpu",),
    matrix_csv: Path | str | None = None,
    keep_going: bool = False,
    warmup: int = 10,
    iterations: int = 100,
) -> int:
    root_path = Path(root)
    rows = _read_csv(Path(matrix_csv) if matrix_csv else root_path / "run_matrix.csv")
    train_success = _load_success_names(root_path / "train_status.csv")
    completed = _load_success_names(root_path / "deployment_status.csv")
    pending = deque((idx, row) for idx, row in enumerate(rows, 1) if row["run_name"] in train_success and row["run_name"] not in completed)
    device_list = [str(device) for device in devices] or ["cpu"]
    print(
        f"[{_now()}] deployment matrix={len(rows)} train_success={len(train_success)} "
        f"completed={len(completed)} pending={len(pending)} devices={','.join(device_list)}",
        flush=True,
    )
    failed = False
    futures = {}
    with ThreadPoolExecutor(max_workers=len(device_list)) as pool:
        for device in device_list:
            if not pending:
                break
            idx, row = pending.popleft()
            futures[pool.submit(_run_deployment_one, root_path, idx, row, device, int(warmup), int(iterations))] = device
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                device = futures.pop(future)
                result = future.result()
                _append_csv(root_path / "deployment_status.csv", result, STATUS_FIELDS)
                print(
                    f"[{result['end_time']}] {result['status']} {result['index']}/{len(rows)} "
                    f"{result['run_name']} device={device} rc={result['returncode']}",
                    flush=True,
                )
                if int(result["returncode"]) != 0:
                    failed = True
                if pending and (keep_going or not failed):
                    idx, row = pending.popleft()
                    futures[pool.submit(_run_deployment_one, root_path, idx, row, device, int(warmup), int(iterations))] = device
    write_deployment_matrix_report(root_path)
    return 1 if failed else 0


def run_deployment_one_from_matrix(
    *,
    root: Path | str,
    index: int,
    run_name: str,
    device: str = "cpu",
    warmup: int = 10,
    iterations: int = 100,
) -> None:
    root_path = Path(root)
    matrix = _read_csv(root_path / "run_matrix.csv")
    by_name = {row["run_name"]: row for row in matrix}
    row = by_name.get(run_name)
    if row is None:
        raise ValueError(f"Unknown run_name {run_name!r}")
    onnx_path = ensure_run_onnx(row, device=device)
    deployment_dir = root_path / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    torch_pred = root_path / "predictions" / f"{run_name}.csv"
    if not torch_pred.exists():
        _run_checked(_predict_command_for_row(root_path, row, device=device, out_csv=torch_pred), timeout=900)
    onnx_pred = deployment_dir / f"onnx_predictions_{run_name}.csv"
    labels_json = str(Path(row.get("run_dir", "")) / "labels.json") if row.get("trainer") != "ultralytics" else ""
    _run_checked(
        [
            sys.executable,
            "-m",
            "autodri.cli.aoi_equivalence",
            "predict-onnx",
            "--onnx",
            str(onnx_path),
            "--data",
            row["data_dir"],
            "--dataset-name",
            row["dataset"],
            "--model-name",
            f"{row['model']}_onnx",
            "--seed",
            row["seed"],
            "--manifest-split",
            "test",
            "--out-csv",
            str(onnx_pred),
            "--batch",
            "64",
            "--preprocess",
            "yolo-cls" if row.get("trainer") == "ultralytics" else "torchvision",
            "--providers",
            *_onnx_providers_for_device(device),
            *(["--labels-json", labels_json] if labels_json and Path(labels_json).exists() else []),
        ],
        timeout=900,
    )
    _run_checked(
        [
            sys.executable,
            "-m",
            "autodri.cli.aoi_equivalence",
            "parity",
            "--reference-predictions",
            str(torch_pred),
            "--candidate-predictions",
            str(onnx_pred),
            "--out-csv",
            str(deployment_dir / f"parity_{run_name}.csv"),
        ],
        timeout=300,
    )
    _run_checked(
        [
            sys.executable,
            "-m",
            "autodri.cli.aoi_equivalence",
            "benchmark-onnx",
            "--onnx",
            str(onnx_path),
            "--model-name",
            row["model"],
            "--out-csv",
            str(deployment_dir / f"latency_{run_name}.csv"),
            "--batch-sizes",
            "1",
            "32",
            "--warmup",
            str(int(warmup)),
            "--iterations",
            str(int(iterations)),
            "--providers",
            *_onnx_providers_for_device(device),
        ],
        timeout=1800,
    )


def ensure_run_onnx(row: Mapping[str, str], *, device: str = "cpu") -> Path:
    onnx_path = _onnx_path_for_run(row)
    if onnx_path.exists() and (row.get("trainer") != "ultralytics" or _onnx_has_dynamic_batch(onnx_path)):
        return onnx_path
    if row.get("trainer") == "ultralytics":
        weights = _weights_path_for_yolo(row)
        if not weights.exists():
            raise FileNotFoundError(weights)
        _run_checked(
            [
                sys.executable,
                "-m",
                "autodri.cli.train_gaze_cls",
                "--mode",
                "export",
                "--data",
                row["data_dir"],
                "--weights",
                str(weights),
                "--imgsz",
                "224",
                "--device",
                device,
                "--dynamic",
            ],
            timeout=900,
        )
    else:
        checkpoint = _checkpoint_path_for_torch(row)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        from autodri.workflows.train_aoi_backbone import export_onnx

        export_onnx(checkpoint, onnx_path, device="cpu" if device == "cpu" else f"cuda:{device}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"Expected ONNX export was not created: {onnx_path}")
    return onnx_path


def _onnx_has_dynamic_batch(path: Path) -> bool:
    try:
        import onnx
    except ImportError:
        return False
    try:
        model = onnx.load(str(path))
    except Exception:
        return False
    if not model.graph.input:
        return False
    dims = model.graph.input[0].type.tensor_type.shape.dim
    return bool(dims and dims[0].dim_param)


def write_deployment_matrix_report(root: Path | str) -> dict[str, Path]:
    root_path = Path(root)
    matrix_rows = _read_csv(root_path / "run_matrix.csv")
    train_success = _load_success_names(root_path / "train_status.csv")
    rows = []
    for row in matrix_rows:
        run_name = row["run_name"]
        onnx_path = _onnx_path_for_run(row)
        latency = _latency_summary_for_run(root_path, run_name)
        parity = _parity_summary_for_run(root_path, run_name)
        out = {
            "run_name": run_name,
            "dataset": row.get("dataset", ""),
            "target_participant": row.get("target_participant", ""),
            "budget": row.get("budget", ""),
            "model": row.get("model", ""),
            "trainer": row.get("trainer", ""),
            "seed": row.get("seed", ""),
            "train_success": int(run_name in train_success),
            "onnx_path": str(onnx_path),
            "onnx_exists": int(onnx_path.exists()),
            "model_size_mb": f"{onnx_path.stat().st_size / (1024 * 1024):.6f}" if onnx_path.exists() else "",
            "top1_parity": parity.get("top1_parity", ""),
            "parity_aligned_total": parity.get("aligned_total", ""),
            **latency,
        }
        rows.append(out)
    summary_csv = root_path / "deployment_summary.csv"
    write_csv_rows(summary_csv, rows)
    tex_path = root_path / "deployment_summary.tex"
    _write_deployment_tex(tex_path, rows)
    return {"summary_csv": summary_csv, "tex": tex_path}


def summarize_roi_audit(
    *,
    manifest_csv: Path | str,
    out_dir: Path | str,
    review_results_csv: Path | str | None = None,
    current_roi_dir: Path | str | None = None,
    manual_roi_csvs: Sequence[Path | str] = (),
) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_rows = _read_csv(Path(manifest_csv))
    merged = _merge_review_rows(manifest_rows, _read_csv(Path(review_results_csv)) if review_results_csv else [])

    status_counts = Counter(row.get("proposed_status", "") or "missing" for row in merged)
    status_rows = [{"proposed_status": key, "count": count} for key, count in sorted(status_counts.items())]
    status_csv = out_path / "roi_audit_status_counts.csv"
    write_csv_rows(status_csv, status_rows)

    manual_diff_rows = _manual_vs_current_roi_rows(current_roi_dir=current_roi_dir, manual_roi_csvs=manual_roi_csvs)
    manual_vs_current_csv = out_path / "roi_grounding_manual_vs_current.current.csv"
    write_csv_rows(manual_vs_current_csv, manual_diff_rows)
    by_participant_rows = _roi_by_participant_rows(merged, manual_diff_rows)
    by_participant_csv = out_path / "roi_grounding_audit_by_participant.current.csv"
    write_csv_rows(by_participant_csv, by_participant_rows)

    summary = _roi_summary_row(merged, manual_diff_rows)
    summary_csv = out_path / "roi_grounding_audit_summary.current.csv"
    write_csv_rows(summary_csv, [summary])
    return {
        "summary_csv": summary_csv,
        "by_participant_csv": by_participant_csv,
        "manual_vs_current_csv": manual_vs_current_csv,
        "status_counts_csv": status_csv,
    }


def discover_default_wheel_maps() -> list[Path]:
    paths = sorted((WORKSPACE_ROOT / "data" / "natural_driving").glob("p*/analysis/p*_wheel_map.current.csv"))
    p1 = WORKSPACE_ROOT / "data" / "natural_driving_p1" / "analysis" / "p1_wheel_map.segment.manual_roi_20260414.csv"
    if p1.exists():
        paths.append(p1)
    return paths


def discover_default_segment_manifests() -> list[Path]:
    paths = sorted((WORKSPACE_ROOT / "data" / "natural_driving").glob("p*/analysis/p*_segments.current.csv"))
    p1 = WORKSPACE_ROOT / "data" / "natural_driving_p1" / "analysis" / "p1_segments.parsed.csv"
    if p1.exists():
        paths.append(p1)
    return paths


def join_wheel_validation(
    manifest_csv: Path | str,
    out_csv: Path | str,
    *,
    wheel_maps: Sequence[Path | str] | None = None,
    segment_manifests: Sequence[Path | str] | None = None,
    workspace_root: Path | str = WORKSPACE_ROOT,
) -> Path:
    workspace = Path(workspace_root)
    maps = _load_wheel_maps([Path(path) for path in (wheel_maps or discover_default_wheel_maps())], workspace_root=workspace)
    segments = _load_segment_rows([Path(path) for path in (segment_manifests or discover_default_segment_manifests())])

    joined_rows: list[dict[str, object]] = []
    for row in _read_csv(Path(manifest_csv)):
        participant = row.get("participant", "")
        source_csv = row.get("source_csv", "")
        map_row = maps.get((participant, Path(source_csv).name)) or maps.get(("", Path(source_csv).name)) or {}
        wheel_csv = _resolve_workspace_path(str(map_row.get("wheel_csv", "")), workspace)
        selected = _select_wheel_state_row(wheel_csv, row.get("timestamp_or_row", "last_row")) if wheel_csv else {}
        segment = segments.get(str(map_row.get("segment_uid", "")), {})
        model_state = _normalize_wheel_state(
            selected.get("stable_state", "") or selected.get("raw_state", "") or selected.get("stable_hand_on_wheel", "")
        )
        joined_rows.append(
            {
                **row,
                "segment_uid": map_row.get("segment_uid", ""),
                "video_path": map_row.get("video_path", "") or segment.get("video_path", ""),
                "wheel_csv": str(wheel_csv) if wheel_csv else str(map_row.get("wheel_csv", "")),
                "wheel_csv_exists": int(bool(wheel_csv and wheel_csv.exists())),
                "model_state": model_state,
                "model_hand_on_wheel": _state_to_binary(model_state),
                "model_frame": selected.get("frame", ""),
                "model_video_time_sec": selected.get("video_time_sec", selected.get("time_sec", "")),
                "segment_start_sec": segment.get("start_sec", ""),
                "segment_end_sec": segment.get("end_sec", ""),
            }
        )
    out_path = Path(out_csv)
    write_csv_rows(out_path, joined_rows)
    return out_path


def summarize_wheel_validation(
    joined_csv: Path | str,
    out_csv: Path | str,
    *,
    human_results_csv: Path | str | None = None,
) -> Path:
    joined_rows = _read_csv(Path(joined_csv))
    human_rows = _read_csv(Path(human_results_csv)) if human_results_csv else []
    human_by_key = {
        _wheel_review_key(row): row
        for row in human_rows
    }
    reviewed = []
    for row in joined_rows:
        merged = dict(row)
        human = human_by_key.get(_wheel_review_key(row), {})
        merged.update({f"human_{key}": value for key, value in human.items() if key not in merged})
        human_state = _extract_human_wheel_state(row, human)
        model_state = _normalize_wheel_state(row.get("model_state", "") or row.get("state", ""))
        if human_state in {"ON", "OFF"} and model_state in {"ON", "OFF"}:
            reviewed.append((row, model_state, human_state))

    false_on = sum(1 for _, model, human in reviewed if model == "ON" and human == "OFF")
    false_off = sum(1 for _, model, human in reviewed if model == "OFF" and human == "ON")
    correct = sum(1 for _, model, human in reviewed if model == human)
    summary_rows = [
        {
            "joined_rows": len(joined_rows),
            "reviewed_rows": len(reviewed),
            "agreement": correct,
            "agreement_rate": _rate(correct, len(reviewed)),
            "false_on": false_on,
            "false_on_rate": _rate(false_on, len(reviewed)),
            "false_off": false_off,
            "false_off_rate": _rate(false_off, len(reviewed)),
            "missing_human_labels": int(not bool(reviewed)),
        }
    ]
    out_path = Path(out_csv)
    write_csv_rows(out_path, summary_rows)
    by_participant = _wheel_by_participant_state(reviewed)
    if by_participant:
        write_csv_rows(out_path.with_name(out_path.stem + "_by_participant_state.csv"), by_participant)
    return out_path


WHEEL_REVIEW_FIELDS = ("participant", "source_csv", "human_state", "human_notes")


def prepare_wheel_review_rows(
    joined_csv: Path | str,
    *,
    human_results_csv: Path | str | None = None,
    workspace_root: Path | str = WORKSPACE_ROOT,
    det_root: Path | str = DEFAULT_WHEEL_DET_ROOT,
) -> list[dict[str, object]]:
    joined_path = Path(joined_csv)
    joined_rows = _read_csv(joined_path)
    human_rows = _read_csv(Path(human_results_csv)) if human_results_csv and Path(human_results_csv).exists() else []
    human_by_key = {_wheel_review_key(row): row for row in human_rows}
    out: list[dict[str, object]] = []
    for idx, row in enumerate(joined_rows):
        video = _resolve_wheel_review_row_video(row, workspace_root=Path(workspace_root))
        det_csv = find_wheel_review_det_csv(row, det_root=det_root)
        if (not video or not video.exists()) and det_csv:
            video = _resolve_video_from_det_csv(det_csv, workspace_root=Path(workspace_root))
        human = human_by_key.get(_wheel_review_key(row), {})
        human_state = _extract_human_wheel_state(row, human)
        roi_info = resolve_wheel_review_roi(row, video=video, det_csv=det_csv, workspace_root=Path(workspace_root))
        out.append(
            {
                "row_id": str(idx),
                "participant": row.get("participant", ""),
                "source_csv": row.get("source_csv", ""),
                "expected_state": _normalize_wheel_state(row.get("state", "")),
                "review_priority": row.get("review_priority", ""),
                "notes": row.get("notes", ""),
                "segment_uid": row.get("segment_uid", ""),
                "video_path": row.get("video_path", ""),
                "video_exists": int(bool(video and video.exists())),
                "video_url": f"/video/{idx}" if video and video.exists() else "",
                "frame_url": f"/frame/{idx}" if det_csv and video and video.exists() else "",
                "det_csv": str(det_csv) if det_csv else "",
                "det_csv_exists": int(bool(det_csv and det_csv.exists())),
                "wheel_csv": row.get("wheel_csv", ""),
                "wheel_csv_exists": row.get("wheel_csv_exists", ""),
                "model_state": _normalize_wheel_state(row.get("model_state", "")),
                "model_hand_on_wheel": row.get("model_hand_on_wheel", ""),
                "model_frame": row.get("model_frame", ""),
                "model_video_time_sec": row.get("model_video_time_sec", ""),
                "segment_start_sec": row.get("segment_start_sec", ""),
                "segment_end_sec": row.get("segment_end_sec", ""),
                **roi_info,
                "human_state": human_state,
                "human_notes": human.get("human_notes", human.get("notes", "")),
            }
        )
    return out


def resolve_wheel_review_roi(
    row: Mapping[str, str],
    *,
    video: Path | None = None,
    det_csv: Path | None = None,
    workspace_root: Path | str = WORKSPACE_ROOT,
) -> dict[str, object]:
    participant = str(row.get("participant", "")).strip()
    manifest_row = _find_current_wheel_roi_row(participant, video=video, row=row, workspace_root=Path(workspace_root))
    det_row = _first_csv_row(det_csv)
    roi = _extract_roi_tuple(manifest_row) or _extract_roi_tuple(det_row)
    source = "none"
    note = ""
    trusted = False
    if manifest_row and _extract_roi_tuple(manifest_row):
        source = "current_wheel_roi_manifest"
        note = str(manifest_row.get("inferred_rule", "") or manifest_row.get("roi_note", "")).strip()
    elif det_row and _extract_roi_tuple(det_row):
        source = "groundingdino_det_csv"
        note = str(det_row.get("roi_note", "") or det_row.get("inferred_rule", "")).strip()
    status = "missing"
    lower_note = note.lower()
    if roi:
        status = "verified_or_explicit"
        trusted = True
        if lower_note in SUSPECT_WHEEL_ROI_RULES or (
            participant in SUSPECT_WHEEL_ROI_PARTICIPANTS and tuple(roi) in _suspect_single_panel_rois()
        ):
            status = "unverified_single_panel_same_as_gaze"
            trusted = False
        elif "manual" in lower_note or "verified" in lower_note or "explicit" in lower_note:
            status = "verified_or_explicit"
            trusted = True
        elif source == "groundingdino_det_csv" and participant in SUSPECT_WHEEL_ROI_PARTICIPANTS:
            status = "unverified_det_csv_roi"
            trusted = False
    return {
        "roi_coords": _format_roi_tuple(roi),
        "roi_review_status": status,
        "roi_review_source": source,
        "roi_review_note": note,
        "roi_review_trusted": int(trusted),
        **resolve_wheel_evidence_roi(participant, roi=roi, roi_status=status),
    }


def resolve_wheel_evidence_roi(
    participant: str,
    *,
    roi: tuple[int, int, int, int] | None,
    roi_status: str,
) -> dict[str, object]:
    if roi and roi_status != "unverified_single_panel_same_as_gaze":
        return {
            "wheel_evidence_roi_coords": _format_roi_tuple(roi),
            "wheel_evidence_roi_status": "same_as_verified_roi",
            "wheel_evidence_roi_source": "current_wheel_roi_manifest",
            "wheel_evidence_roi_trusted": 1,
            "wheel_evidence_roi_note": "Wheel evidence uses the verified or explicit wheel ROI.",
        }
    candidate = WHEEL_EVIDENCE_CANDIDATE_ROIS.get(participant)
    if candidate:
        return {
            "wheel_evidence_roi_coords": _format_roi_tuple(candidate),
            "wheel_evidence_roi_status": "candidate_wheel_panel_unverified",
            "wheel_evidence_roi_source": "layout_candidate_from_review_grid",
            "wheel_evidence_roi_trusted": 0,
            "wheel_evidence_roi_note": (
                "Review-only candidate wheel/cockpit panel from prior grid inspection; "
                "the persisted wheel ROI remains unverified."
            ),
        }
    return {
        "wheel_evidence_roi_coords": "",
        "wheel_evidence_roi_status": "missing",
        "wheel_evidence_roi_source": "none",
        "wheel_evidence_roi_trusted": 0,
        "wheel_evidence_roi_note": "",
    }


def find_wheel_review_det_csv(
    row: Mapping[str, str],
    *,
    det_root: Path | str = DEFAULT_WHEEL_DET_ROOT,
) -> Path | None:
    root = Path(det_root)
    participant = str(row.get("participant", "")).strip()
    source_csv = Path(str(row.get("source_csv", ""))).name
    segment_uid = str(row.get("segment_uid", "")).strip()
    candidates: list[Path] = []
    if segment_uid and segment_uid.lower() != "nan":
        candidates.extend(
            [
                root / participant / f"{segment_uid}.wheel.det.csv",
                root / "p1_manual_roi_20260414" / f"{segment_uid}.wheel.det.csv",
            ]
        )
    if "__" in source_csv:
        suffix = source_csv.split("__", 1)[1].replace(".wheel.csv", "")
        candidates.append(root / participant / f"{suffix}.wheel.det.csv")
        if "_seg_" in suffix:
            prefix = suffix.split("_seg_", 1)[0]
            candidates.append(root / prefix / f"{suffix}.wheel.det.csv")
            if prefix.startswith("p") and prefix != participant:
                participant_suffix = suffix.replace(prefix + "_", participant + "_", 1)
                candidates.append(root / participant / f"{participant_suffix}.wheel.det.csv")
    source_stem = source_csv.replace(".wheel.csv", "")
    candidates.append(root / participant / f"{source_stem}.wheel.det.csv")
    for path in candidates:
        if path.exists():
            return path
    return None


def generate_wheel_review_frame(
    joined_csv: Path | str,
    row_id: str,
    *,
    out_dir: Path | str = DEFAULT_REPORT_ROOT / "wheel_review_frames",
    workspace_root: Path | str = WORKSPACE_ROOT,
    det_root: Path | str = DEFAULT_WHEEL_DET_ROOT,
) -> Path | None:
    if cv2 is None:
        raise ImportError("opencv-python is required to render wheel review frames")
    if not str(row_id).isdigit():
        return None
    joined_rows = _read_csv(Path(joined_csv))
    idx = int(row_id)
    if idx < 0 or idx >= len(joined_rows):
        return None
    row = joined_rows[idx]
    video = _resolve_wheel_review_row_video(row, workspace_root=Path(workspace_root))
    det_csv = find_wheel_review_det_csv(row, det_root=det_root)
    if (not video or not video.exists()) and det_csv:
        video = _resolve_video_from_det_csv(det_csv, workspace_root=Path(workspace_root))
    if not video or not video.exists() or not det_csv:
        return None
    out_path = Path(out_dir) / f"row_{idx:03d}_{WHEEL_REVIEW_FRAME_VERSION}_{Path(str(row.get('source_csv', 'row'))).stem}.jpg"
    if out_path.exists():
        return out_path
    det_rows = _read_csv(det_csv)
    if not det_rows:
        return None
    target_frame = _safe_float(row.get("model_frame", "nan"))
    target_time = _safe_float(row.get("model_video_time_sec", "nan"))
    selected_rows = _nearest_detection_rows(det_rows, target_frame=target_frame, target_time=target_time)
    if not selected_rows:
        return None
    selected = selected_rows[0]
    frame_idx = int(round(_safe_float(selected.get("video_frame", selected.get("frame", "0")))))
    frame = _read_video_frame(video, frame_idx)
    if frame is None:
        fallback_frame = int(round(_safe_float(selected.get("frame", "0"))))
        frame = _read_video_frame(video, fallback_frame)
    if frame is None:
        return None
    roi_info = resolve_wheel_review_roi(row, video=video, det_csv=det_csv, workspace_root=workspace_root)
    annotated = _draw_wheel_review_frame(frame, selected_rows, {**row, **roi_info})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    return out_path


def resolve_wheel_review_video(
    joined_csv: Path | str,
    row_id: str,
    *,
    workspace_root: Path | str = WORKSPACE_ROOT,
) -> Path | None:
    if not str(row_id).isdigit():
        return None
    rows = _read_csv(Path(joined_csv))
    idx = int(row_id)
    if idx < 0 or idx >= len(rows):
        return None
    path = _resolve_wheel_review_row_video(rows[idx], workspace_root=Path(workspace_root))
    if not path or not path.exists() or not path.is_file():
        return None
    return path


def write_wheel_review_results(out_csv: Path | str, review_rows: Sequence[Mapping[str, object]]) -> Path:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in review_rows:
        state = _normalize_wheel_state(row.get("human_state", ""))
        if state not in {"ON", "OFF", "UNCERTAIN"}:
            continue
        participant = str(row.get("participant", "")).strip()
        source_csv = Path(str(row.get("source_csv", ""))).name
        if not participant or not source_csv:
            continue
        by_key[(participant, source_csv)] = {
            "participant": participant,
            "source_csv": source_csv,
            "human_state": state,
            "human_notes": str(row.get("human_notes", row.get("notes", ""))).strip(),
        }
    out_path = Path(out_csv)
    write_csv_rows(out_path, list(by_key.values()), fieldnames=WHEEL_REVIEW_FIELDS)
    return out_path


def serve_wheel_review(
    *,
    joined_csv: Path | str = DEFAULT_REPORT_ROOT / "wheel_validation_joined.current.csv",
    out_csv: Path | str = DEFAULT_REPORT_ROOT / "wheel_validation_human_review.csv",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    joined_path = Path(joined_csv)
    out_path = Path(out_csv)
    if not joined_path.exists():
        raise FileNotFoundError(joined_path)

    class WheelReviewHandler(BaseHTTPRequestHandler):
        server_version = "AutoDriWheelReview/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"", "/"}:
                self._send_text(_wheel_review_html(), content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/rows":
                rows = prepare_wheel_review_rows(joined_path, human_results_csv=out_path)
                self._send_json({"rows": rows, "joined_csv": str(joined_path), "out_csv": str(out_path)})
                return
            if parsed.path == "/api/export":
                self._send_csv(out_path)
                return
            if parsed.path.startswith("/frame/"):
                self._send_frame(parsed.path.rsplit("/", 1)[-1])
                return
            if parsed.path.startswith("/video/"):
                self._send_video(parsed.path.rsplit("/", 1)[-1])
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
                return
            reviews = payload.get("reviews", [])
            if not isinstance(reviews, list):
                self.send_error(HTTPStatus.BAD_REQUEST, "reviews must be a list")
                return
            written = write_wheel_review_results(out_path, reviews)
            count = len(_read_csv(written)) if written.exists() else 0
            self._send_json({"saved_rows": count, "out_csv": str(written)})

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[wheel-review] {self.address_string()} - {fmt % args}")

        def _send_text(self, body: str, *, content_type: str = "text/plain; charset=utf-8") -> None:
            data = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: Mapping[str, object]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_csv(self, path: Path) -> None:
            if not path.exists():
                write_wheel_review_results(path, [])
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_frame(self, row_id: str) -> None:
            frame = generate_wheel_review_frame(joined_path, row_id)
            if not frame:
                self.send_error(HTTPStatus.NOT_FOUND, "Annotated frame is not available for this row")
                return
            data = frame.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_video(self, row_id: str) -> None:
            video = resolve_wheel_review_video(joined_path, row_id)
            if not video:
                self.send_error(HTTPStatus.NOT_FOUND, "Video is not available for this row")
                return
            size = video.stat().st_size
            start, end = _parse_range_header(self.headers.get("Range"), size)
            status = HTTPStatus.PARTIAL_CONTENT if start or end < size - 1 else HTTPStatus.OK
            content_type = mimetypes.guess_type(str(video))[0] or "application/octet-stream"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with video.open("rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    httpd = ThreadingHTTPServer((host, int(port)), WheelReviewHandler)
    print(f"Serving wheel review UI on http://{host}:{port}")
    print(f"Joined CSV: {joined_path}")
    print(f"Human review CSV: {out_path}")
    httpd.serve_forever()


STATUS_FIELDS = [
    "index",
    "run_name",
    "dataset",
    "target_participant",
    "model",
    "seed",
    "status",
    "returncode",
    "start_time",
    "end_time",
    "log_path",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoUI WIP experiment orchestration utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_lopo = sub.add_parser("prepare-lopo", help="Prepare multi-seed participant LOPO datasets and run matrix")
    _add_common_matrix_args(p_lopo, default_root=DEFAULT_MULTI_SEED_ROOT)
    p_lopo.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    p_lopo.set_defaults(func=cmd_prepare_lopo)

    p_few = sub.add_parser("prepare-fewshot", help="Prepare participant few-shot adaptation datasets and run matrix")
    _add_common_matrix_args(p_few, default_root=DEFAULT_FEWSHOT_ROOT)
    p_few.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    p_few.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    p_few.add_argument("--internal-val-ratio", type=float, default=0.2)
    p_few.set_defaults(func=cmd_prepare_fewshot)

    p_train = sub.add_parser("train-matrix", help="Run a prepared train matrix with GPU parallelism")
    p_train.add_argument("--root", required=True)
    p_train.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "4", "5", "6"])
    p_train.add_argument("--keep-going", action="store_true")
    p_train.set_defaults(func=cmd_train_matrix)

    p_pred = sub.add_parser("predict-matrix", help="Predict all successful runs in a prepared matrix")
    p_pred.add_argument("--root", required=True)
    p_pred.add_argument("--gpus", nargs="+", default=["0"])
    p_pred.add_argument("--keep-going", action="store_true")
    p_pred.set_defaults(func=cmd_predict_matrix)

    p_stats = sub.add_parser("stats-matrix", help="Compute frame/event metrics and bootstrap equivalence")
    p_stats.add_argument("--root", required=True)
    p_stats.add_argument("--n-boot", type=int, default=1000)
    p_stats.add_argument("--metric", default="primary3_macro_f1")
    p_stats.set_defaults(func=cmd_stats_matrix)

    p_report = sub.add_parser("report-matrix", help="Write paper-ready summary CSV/TeX tables")
    p_report.add_argument("--root", required=True)
    p_report.add_argument("--title", default="AutoUI AOI Experiment")
    p_report.set_defaults(func=cmd_report_matrix)

    p_few_report = sub.add_parser("report-fewshot", help="Write budget-curve summary CSV/TeX/SVG for few-shot runs")
    p_few_report.add_argument("--root", required=True)
    p_few_report.set_defaults(func=cmd_report_fewshot)

    p_deploy = sub.add_parser("deployment-matrix", help="Export/check ONNX, run ONNX parity, and benchmark latency")
    p_deploy.add_argument("--root", required=True)
    p_deploy.add_argument("--devices", nargs="+", default=["cpu"])
    p_deploy.add_argument("--keep-going", action="store_true")
    p_deploy.add_argument("--warmup", type=int, default=10)
    p_deploy.add_argument("--iterations", type=int, default=100)
    p_deploy.set_defaults(func=cmd_deployment_matrix)

    p_deploy_report = sub.add_parser("report-deployment", help="Summarize deployment parity/latency outputs")
    p_deploy_report.add_argument("--root", required=True)
    p_deploy_report.set_defaults(func=cmd_report_deployment)

    p_roi = sub.add_parser("summarize-roi-audit", help="Aggregate ROI manual audit/review results")
    p_roi.add_argument("--manifest-csv", default=str(DEFAULT_AUTOWIP_ROOT / "agent_outputs/roi_grounding/roi_grounding_review_manifest.csv"))
    p_roi.add_argument("--out-dir", default=str(DEFAULT_REPORT_ROOT))
    p_roi.add_argument("--review-results-csv", default="")
    p_roi.add_argument("--current-roi-dir", default=str(WORKSPACE_ROOT / "artifacts/manifests/current"))
    p_roi.add_argument("--manual-roi-csv", action="append", default=[])
    p_roi.set_defaults(func=cmd_summarize_roi)

    p_join = sub.add_parser("join-wheel-validation", help="Join the 40-row wheel validation manifest to videos and wheel CSV states")
    p_join.add_argument("--manifest-csv", default=str(DEFAULT_AUTOWIP_ROOT / "agent_outputs/wheel_validation/wheel_validation_manifest.csv"))
    p_join.add_argument("--out-csv", default=str(DEFAULT_REPORT_ROOT / "wheel_validation_joined.current.csv"))
    p_join.add_argument("--wheel-map", action="append", default=[])
    p_join.add_argument("--segment-manifest", action="append", default=[])
    p_join.set_defaults(func=cmd_join_wheel)

    p_wheel = sub.add_parser("summarize-wheel-validation", help="Aggregate hand-on-wheel human validation results")
    p_wheel.add_argument("--joined-csv", default=str(DEFAULT_REPORT_ROOT / "wheel_validation_joined.current.csv"))
    p_wheel.add_argument("--out-csv", default=str(DEFAULT_REPORT_ROOT / "wheel_validation_agreement.current.csv"))
    p_wheel.add_argument("--human-results-csv", default="")
    p_wheel.set_defaults(func=cmd_summarize_wheel)

    p_serve_wheel = sub.add_parser("serve-wheel-review", help="Launch a browser UI for the 40-row hand-on-wheel review")
    p_serve_wheel.add_argument("--joined-csv", default=str(DEFAULT_REPORT_ROOT / "wheel_validation_joined.current.csv"))
    p_serve_wheel.add_argument("--out-csv", default=str(DEFAULT_REPORT_ROOT / "wheel_validation_human_review.csv"))
    p_serve_wheel.add_argument("--host", default="127.0.0.1")
    p_serve_wheel.add_argument("--port", type=int, default=8765)
    p_serve_wheel.set_defaults(func=cmd_serve_wheel_review)

    return parser


def cmd_prepare_lopo(args: argparse.Namespace) -> None:
    models = _parse_model_filter(args.models)
    matrix = prepare_lopo_matrix(
        default_participant_datasets(),
        out_dir=args.root,
        targets=args.targets,
        seeds=args.seeds,
        labels=tuple(args.labels.split(",")),
        models=models,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        yolo_aug_preset=args.yolo_aug_preset,
        no_class_weights=args.no_class_weights,
    )
    print(f"Wrote {len(matrix)} LOPO training runs to {Path(args.root) / 'run_matrix.csv'}")


def cmd_prepare_fewshot(args: argparse.Namespace) -> None:
    models = _parse_model_filter(args.models)
    rows = prepare_fewshot_datasets(
        default_participant_datasets(),
        out_dir=args.root,
        targets=args.targets,
        budgets=args.budgets,
        seeds=args.seeds,
        labels=tuple(args.labels.split(",")),
        models=models,
        internal_val_ratio=args.internal_val_ratio,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        yolo_aug_preset=args.yolo_aug_preset,
        no_class_weights=args.no_class_weights,
    )
    print(f"Wrote {len(rows)} few-shot datasets and run matrix to {Path(args.root)}")


def cmd_train_matrix(args: argparse.Namespace) -> None:
    raise SystemExit(run_train_matrix(root=args.root, gpus=args.gpus, keep_going=args.keep_going))


def cmd_predict_matrix(args: argparse.Namespace) -> None:
    raise SystemExit(run_predict_matrix(root=args.root, gpus=args.gpus, keep_going=args.keep_going))


def cmd_stats_matrix(args: argparse.Namespace) -> None:
    run_stats(args.root, n_boot=args.n_boot, metric=args.metric)
    print(f"Wrote metrics and stats under {args.root}")


def cmd_report_matrix(args: argparse.Namespace) -> None:
    outputs = write_matrix_report(args.root, title=args.title)
    for key, path in outputs.items():
        print(f"{key}: {path}")


def cmd_report_fewshot(args: argparse.Namespace) -> None:
    outputs = write_fewshot_curve_report(args.root)
    for key, path in outputs.items():
        print(f"{key}: {path}")


def cmd_deployment_matrix(args: argparse.Namespace) -> None:
    raise SystemExit(
        run_deployment_matrix(
            root=args.root,
            devices=args.devices,
            keep_going=args.keep_going,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    )


def cmd_report_deployment(args: argparse.Namespace) -> None:
    outputs = write_deployment_matrix_report(args.root)
    for key, path in outputs.items():
        print(f"{key}: {path}")


def cmd_deployment_one(args: argparse.Namespace) -> None:
    run_deployment_one_from_matrix(
        root=args.root,
        index=args.index,
        run_name=args.run_name,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
    )


def cmd_summarize_roi(args: argparse.Namespace) -> None:
    manual_csvs = [Path(path) for path in args.manual_roi_csv]
    if not manual_csvs:
        default_manuals = [
            WORKSPACE_ROOT / "archive/gaze_onnx_experiments/roi_refs/p1_dual_roi_review_current/p1_gaze_rois.manual.csv",
            WORKSPACE_ROOT / "archive/gaze_onnx_experiments/roi_refs/p1_dual_roi_review_current/p1_wheel_rois.manual.csv",
        ]
        manual_csvs = [path for path in default_manuals if path.exists()]
    outputs = summarize_roi_audit(
        manifest_csv=args.manifest_csv,
        out_dir=args.out_dir,
        review_results_csv=args.review_results_csv or None,
        current_roi_dir=args.current_roi_dir or None,
        manual_roi_csvs=manual_csvs,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


def cmd_join_wheel(args: argparse.Namespace) -> None:
    out = join_wheel_validation(
        args.manifest_csv,
        args.out_csv,
        wheel_maps=[Path(path) for path in args.wheel_map] or None,
        segment_manifests=[Path(path) for path in args.segment_manifest] or None,
    )
    print(out)


def cmd_summarize_wheel(args: argparse.Namespace) -> None:
    out = summarize_wheel_validation(
        args.joined_csv,
        args.out_csv,
        human_results_csv=args.human_results_csv or None,
    )
    print(out)


def cmd_serve_wheel_review(args: argparse.Namespace) -> None:
    serve_wheel_review(
        joined_csv=args.joined_csv,
        out_csv=args.out_csv,
        host=args.host,
        port=args.port,
    )


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "_deployment-one":
        parser = _build_deployment_one_parser()
        args = parser.parse_args(argv[1:])
        cmd_deployment_one(args)
        return
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def _build_deployment_one_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=argparse.SUPPRESS)
    parser.add_argument("--root", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--visible-device", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser


def _add_common_matrix_args(parser: argparse.ArgumentParser, *, default_root: Path) -> None:
    parser.add_argument("--root", default=str(default_root))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", default=[model.model for model in DEFAULT_MODELS])
    parser.add_argument("--labels", default=",".join(PRIMARY_LABELS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--yolo-aug-preset", choices=["baseline", "robust", "genv3"], default="baseline")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--no-class-weights", dest="no_class_weights", action="store_true", default=True)
    weights.add_argument("--class-weights", dest="no_class_weights", action="store_false")


def _parse_model_filter(names: Sequence[str]) -> tuple[ExperimentModel, ...]:
    requested = set(names)
    models = tuple(model for model in DEFAULT_MODELS if model.model in requested)
    missing = requested - {model.model for model in models}
    if missing:
        raise ValueError(f"Unknown model(s): {sorted(missing)}")
    return models


def _assert_participant_datasets(participant_datasets: Mapping[str, Path | str]) -> None:
    missing = [
        f"{participant}:{Path(path) / 'split_manifest.csv'}"
        for participant, path in participant_datasets.items()
        if not (Path(path) / "split_manifest.csv").exists()
    ]
    if missing:
        raise FileNotFoundError("Missing participant split manifests: " + ", ".join(missing))


def _run_matrix_rows(
    dataset_rows: Sequence[Mapping[str, object]],
    *,
    root: Path,
    models: Sequence[ExperimentModel],
    labels: Sequence[str],
    epochs: int,
    batch: int,
    workers: int,
    yolo_aug_preset: str,
    no_class_weights: bool,
    name_prefix: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels_arg = ",".join(labels)
    for dataset_row in dataset_rows:
        dataset = str(dataset_row["dataset"])
        seed = int(dataset_row["seed"])
        data_dir = str(dataset_row["data_dir"])
        target = str(dataset_row.get("target_participant", dataset_row.get("participant", "")))
        budget_suffix = f"_budget{dataset_row['budget']}" if "budget" in dataset_row else ""
        for model in models:
            run_name = f"{name_prefix}{dataset}_{model.model}{budget_suffix}_seed{seed}".replace("__", "_")
            if model.trainer == "ultralytics":
                run_dir = root / "runs_yolo" / run_name
                train_command = (
                    "python -m autodri.cli.train_gaze_cls "
                    f"--data {data_dir} --model {model.base_model} --name {run_name} "
                    f"--project {root / 'runs_yolo'} --seed {seed} --epochs {int(epochs)} --batch {int(batch)} "
                    f"--workers {int(workers)} --aug-preset {yolo_aug_preset}"
                )
            else:
                run_dir = root / "runs_torch" / run_name
                class_weight_arg = " --no-class-weights" if no_class_weights else ""
                train_command = (
                    "python -m autodri.cli.train_aoi_backbone "
                    f"--data {data_dir} --model {model.base_model} --name {run_name} "
                    f"--project {root / 'runs_torch'} --seed {seed} --epochs {int(epochs)} --batch {int(batch)} "
                    f"--workers {int(workers)} --export-onnx --use-physical-splits --labels {labels_arg}{class_weight_arg}"
                )
            rows.append(
                {
                    "dataset": dataset,
                    "target_participant": target,
                    "budget": dataset_row.get("budget", ""),
                    "data_dir": data_dir,
                    "seed": seed,
                    "model": model.model,
                    "family": model.family,
                    "arch_group": model.arch_group,
                    "trainer": model.trainer,
                    "base_model": model.base_model,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "train_command": train_command,
                }
            )
    return rows


def _materialize_fewshot_dataset(
    source_dir: Path,
    out_dir: Path,
    *,
    participant: str,
    selected_train: Sequence[object],
    frozen_test: Sequence[object],
    train_assignment: Mapping[str, str],
    labels: Sequence[str] = PRIMARY_LABELS,
) -> dict[str, int]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_class_dirs(out_dir, splits=("train", "internal_val", "test"), labels=labels)
    rows_out: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    for sample in selected_train:
        split = train_assignment[sample.dst_rel]
        if split == "test":
            split = "internal_val"
        _link_manifest_sample(source_dir, out_dir, sample, participant=participant, split=split, rows_out=rows_out)
        counts[split] += 1
    for sample in frozen_test:
        _link_manifest_sample(source_dir, out_dir, sample, participant=participant, split="test", rows_out=rows_out)
        counts["test"] += 1

    _ensure_val_alias(out_dir)
    write_csv_rows(
        out_dir / "split_manifest.csv",
        rows_out,
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
            "participant",
            "source_split",
            "source_dataset",
        ],
    )
    return {split: counts[split] for split in ("train", "internal_val", "test")}


def _keep_labels_in_train(samples: Sequence[object], assignment: Mapping[str, str], labels: Sequence[str]) -> dict[str, str]:
    adjusted = dict(assignment)
    required = {str(label) for label in labels}
    train_labels = {sample.label for sample in samples if adjusted.get(sample.dst_rel) == "train"}
    missing = sorted((required & {sample.label for sample in samples}) - train_labels)
    if not missing:
        return adjusted

    by_group: dict[tuple[str, str, int], list[object]] = defaultdict(list)
    for sample in samples:
        by_group[sample.group_key()].append(sample)
    for label in missing:
        candidates = [
            group_rows
            for group_rows in by_group.values()
            if any(sample.label == label for sample in group_rows)
            and any(adjusted.get(sample.dst_rel) == "internal_val" for sample in group_rows)
        ]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda rows: (len(rows), rows[0].domain, rows[0].video, rows[0].timestamp))
        for sample in chosen:
            adjusted[sample.dst_rel] = "train"
    return adjusted


def _link_manifest_sample(source_dir: Path, out_dir: Path, sample: object, *, participant: str, split: str, rows_out: list[dict[str, object]]) -> None:
    src = source_dir / sample.dst_rel
    if not src.exists():
        raise FileNotFoundError(src)
    dst_rel = f"{split}/{sample.label}/{participant}__{sample.dst_rel.replace('/', '__')}"
    dst = out_dir / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())
    rows_out.append(
        {
            "split": split,
            "label": sample.label,
            "domain": sample.domain,
            "frame_id": sample.frame_id,
            "timestamp": f"{sample.timestamp:.6f}",
            "video": sample.video,
            "src_rel": sample.dst_rel,
            "dst_rel": dst_rel,
            "augmented": int(sample.augmented),
            "participant": participant,
            "source_split": sample.split,
            "source_dataset": str(source_dir),
        }
    )


def _ensure_val_alias(out_dir: Path) -> None:
    alias = out_dir / "val"
    target = out_dir / "internal_val"
    if alias.exists() or alias.is_symlink():
        if alias.is_symlink() or alias.is_file():
            alias.unlink()
        else:
            shutil.rmtree(alias)
    if target.exists():
        alias.symlink_to(target.resolve(), target_is_directory=True)


def _ensure_class_dirs(out_dir: Path, *, splits: Sequence[str], labels: Sequence[str]) -> None:
    for split in splits:
        for label in labels:
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)


def _stratified_sample(rows: Sequence[object], *, n: int, seed: int) -> list[object]:
    n = max(1, min(int(n), len(rows)))
    rng = random.Random(seed)
    by_label: dict[str, list[object]] = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)

    selected: list[object] = []
    labels = sorted(by_label)
    if n >= len(labels):
        for label in labels:
            selected.append(by_label[label].pop())

    remaining_budget = n - len(selected)
    remaining_rows = [row for label in labels for row in by_label[label]]
    if remaining_budget > 0:
        rng.shuffle(remaining_rows)
        selected.extend(remaining_rows[:remaining_budget])
    selected.sort(key=lambda row: (row.domain, row.video, row.timestamp, row.dst_rel))
    return selected


def _stratified_group_sample(rows: Sequence[object], *, n: int, seed: int) -> list[object]:
    n = max(1, min(int(n), len(rows)))
    rng = random.Random(seed)
    groups: dict[tuple[str, str, int], list[object]] = defaultdict(list)
    for row in rows:
        groups[row.group_key()].append(row)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    selected: list[object] = []
    seen_labels: set[str] = set()
    used_groups: set[tuple[str, str, int]] = set()

    for key, group_rows in sorted(group_items, key=lambda item: (item[0][0], item[0][1], item[0][2])):
        labels = {row.label for row in group_rows}
        if not seen_labels.intersection(labels):
            selected.extend(group_rows)
            seen_labels.update(labels)
            used_groups.add(key)
        if len(selected) >= n:
            break

    if len(selected) < n:
        remaining = [row for key, group_rows in group_items if key not in used_groups for row in group_rows]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    by_group: dict[tuple[str, str, int], list[object]] = defaultdict(list)
    for row in selected:
        by_group[row.group_key()].append(row)
    selected = [row for _, group_rows in sorted(by_group.items()) for row in sorted(group_rows, key=lambda row: row.dst_rel)]
    selected.sort(key=lambda row: (row.domain, row.video, row.timestamp, row.dst_rel))
    return selected[:n]


def _run_training_one(root: Path, index: int, row: Mapping[str, str], gpu: str) -> dict[str, object]:
    run_name = row["run_name"]
    run_dir = Path(row.get("run_dir") or _fallback_run_dir(root, row))
    if run_dir.exists():
        shutil.rmtree(run_dir)
    log_path = root / "logs" / f"{index:04d}_{run_name}.gpu{gpu}.log"
    cmd = _command_for_device(row["train_command"], row["trainer"], gpu)
    return _run_logged(cmd, cwd=REPO_ROOT, log_path=log_path, index=index, row=row)


def _run_predict_one(root: Path, index: int, row: Mapping[str, str], device: str) -> dict[str, object]:
    run_name = row["run_name"]
    out_csv = root / "predictions" / f"{run_name}.csv"
    cmd = _predict_command_for_row(root, row, device=device, out_csv=out_csv)
    log_path = root / "predict_logs" / f"{index:04d}_{run_name}.device{device}.log"
    return _run_logged(cmd, cwd=REPO_ROOT, log_path=log_path, index=index, row=row)


def _run_deployment_one(root: Path, index: int, row: Mapping[str, str], device: str, warmup: int, iterations: int) -> dict[str, object]:
    local_device = "cpu" if str(device) == "cpu" else "0"
    command = [
        sys.executable,
        "-m",
        "autodri.cli.autoui_experiments",
        "_deployment-one",
        "--root",
        str(root),
        "--index",
        str(index),
        "--run-name",
        row["run_name"],
        "--device",
        local_device,
        "--visible-device",
        str(device),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    log_path = root / "deployment_logs" / f"{index:04d}_{row['run_name']}.device{device}.log"
    return _run_logged(command, cwd=REPO_ROOT, log_path=log_path, index=index, row=row)


def _run_logged(cmd: Sequence[str], *, cwd: Path, log_path: Path, index: int, row: Mapping[str, str]) -> dict[str, object]:
    start = _now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"start_time={start}\n")
        log.write(f"command={shlex.join([str(x) for x in cmd])}\n\n")
        log.flush()
        env = os.environ.copy()
        if "_deployment-one" in [str(x) for x in cmd]:
            device_arg = _arg_after(cmd, "--visible-device") or _arg_after(cmd, "--device")
            env.update(_onnx_env_for_device(device_arg))
        proc = subprocess.run([str(x) for x in cmd], cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        end = _now()
        log.write(f"\nend_time={end}\nreturncode={proc.returncode}\n")
    return {
        "index": index,
        "run_name": row.get("run_name", ""),
        "dataset": row.get("dataset", ""),
        "target_participant": row.get("target_participant", ""),
        "model": row.get("model", ""),
        "seed": row.get("seed", ""),
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "start_time": start,
        "end_time": end,
        "log_path": log_path,
    }


def _command_for_device(command: str, trainer: str, gpu: str) -> list[str]:
    parts = shlex.split(command)
    device = "cpu" if gpu == "cpu" else (gpu if trainer == "ultralytics" else f"cuda:{gpu}")
    if "--device" in parts:
        parts[parts.index("--device") + 1] = device
    else:
        parts.extend(["--device", device])
    return parts


def _fallback_run_dir(root: Path, row: Mapping[str, str]) -> Path:
    return root / ("runs_yolo" if row["trainer"] == "ultralytics" else "runs_torch") / row["run_name"]


def _onnx_providers_for_device(device: str) -> list[str]:
    return ["CPUExecutionProvider"] if str(device) == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _onnx_env_for_device(device: str) -> dict[str, str]:
    if str(device) == "cpu":
        return {}
    return {"CUDA_VISIBLE_DEVICES": str(device)}


def _arg_after(cmd: Sequence[object], flag: str) -> str:
    parts = [str(x) for x in cmd]
    return parts[parts.index(flag) + 1] if flag in parts and parts.index(flag) + 1 < len(parts) else ""


def _predict_command_for_row(root: Path, row: Mapping[str, str], *, device: str, out_csv: Path) -> list[str]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(row.get("run_dir") or _fallback_run_dir(root, row))
    if row["trainer"] == "ultralytics":
        weights = run_dir / "weights" / "best.pt"
        return [
            sys.executable,
            "-m",
            "autodri.cli.aoi_equivalence",
            "predict-yolo",
            "--weights",
            str(weights),
            "--data",
            row["data_dir"],
            "--dataset-name",
            row["dataset"],
            "--model-name",
            row["model"],
            "--seed",
            row["seed"],
            "--manifest-split",
            "test",
            "--out-csv",
            str(out_csv),
            "--batch",
            "64",
            "--device",
            device,
        ]
    checkpoint = run_dir / "best.pt"
    return [
        sys.executable,
        "-m",
        "autodri.cli.aoi_equivalence",
        "predict-torch",
        "--checkpoint",
        str(checkpoint),
        "--data",
        row["data_dir"],
        "--dataset-name",
        row["dataset"],
        "--model-name",
        row["model"],
        "--seed",
        row["seed"],
        "--manifest-split",
        "test",
        "--out-csv",
        str(out_csv),
        "--batch",
        "64",
        "--device",
        "cpu" if device == "cpu" else f"cuda:{device}",
    ]


def _budget_from_row(row: Mapping[str, str]) -> int | None:
    raw = str(row.get("budget", "")).strip()
    if raw:
        return int(float(raw))
    dataset = str(row.get("dataset", ""))
    for marker in ("_b", "budget"):
        if marker in dataset:
            tail = dataset.rsplit(marker, 1)[1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            if digits:
                return int(digits)
    return None


def _mean_metric(rows: Sequence[Mapping[str, str]], key: str) -> float:
    values = [_safe_float(row.get(key, "")) for row in rows]
    return statistics.fmean(values) if values else 0.0


def _std_metric(rows: Sequence[Mapping[str, str]], key: str) -> float:
    values = [_safe_float(row.get(key, "")) for row in rows]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _write_fewshot_curve_tex(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{llrr}\n",
        "\\toprule\n",
        "Budget & Model & Runs & Primary3 Macro-F1 \\\\\n",
        "\\midrule\n",
    ]
    for row in rows:
        lines.append(
            f"{row['budget']} & {_latex_escape(str(row['model']))} & {row['runs']} & "
            f"{float(row['primary3_macro_f1_mean']):.3f} $\\pm$ {float(row['primary3_macro_f1_std']):.3f} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    path.write_text("".join(lines), encoding="utf-8")


def _write_fewshot_curve_svg(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    points_by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        points_by_model[str(row["model"])].append((float(row["budget"]), float(row["primary3_macro_f1_mean"])))
    budgets = sorted({x for pts in points_by_model.values() for x, _ in pts})
    values = [y for pts in points_by_model.values() for _, y in pts]
    width, height = 640, 360
    left, right, top, bottom = 70, 25, 30, 55
    min_x, max_x = (min(budgets), max(budgets)) if budgets else (0.0, 1.0)
    min_y, max_y = (max(0.0, min(values) - 0.05), min(1.0, max(values) + 0.05)) if values else (0.0, 1.0)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 0.1

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * (width - left - right)

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * (height - top - bottom)

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect width="100%" height="100%" fill="white"/>\n',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>\n',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>\n',
        f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="13">Label budget</text>\n',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="13">Primary3 macro-F1</text>\n',
    ]
    for budget in budgets:
        x = sx(budget)
        lines.append(f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" font-size="11">{int(budget)}</text>\n')
    for idx, (model, pts) in enumerate(sorted(points_by_model.items())):
        color = palette[idx % len(palette)]
        pts = sorted(pts)
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>\n')
        for x, y in pts:
            lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>\n')
        lines.append(f'<text x="{width-right-105}" y="{top+18+18*idx}" font-size="12" fill="{color}">{_xml_escape(model)}</text>\n')
    lines.append("</svg>\n")
    path.write_text("".join(lines), encoding="utf-8")


def _onnx_path_for_run(row: Mapping[str, str]) -> Path:
    run_dir = Path(row.get("run_dir", ""))
    if row.get("trainer") == "ultralytics":
        return run_dir / "weights" / "best.onnx"
    return run_dir / "best.onnx"


def _weights_path_for_yolo(row: Mapping[str, str]) -> Path:
    return Path(row.get("run_dir", "")) / "weights" / "best.pt"


def _checkpoint_path_for_torch(row: Mapping[str, str]) -> Path:
    return Path(row.get("run_dir", "")) / "best.pt"


def _latency_summary_for_run(root: Path, run_name: str) -> dict[str, str]:
    path = root / "deployment" / f"latency_{run_name}.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for row in _read_csv(path):
        batch = str(int(float(row.get("batch_size", "0"))))
        out[f"latency_b{batch}_p50_ms"] = row.get("latency_p50_ms", "")
        out[f"latency_b{batch}_p95_ms"] = row.get("latency_p95_ms", "")
        out[f"throughput_b{batch}_img_s"] = row.get("throughput_img_s", "")
    return out


def _parity_summary_for_run(root: Path, run_name: str) -> dict[str, str]:
    path = root / "deployment" / f"parity_{run_name}.csv"
    if not path.exists():
        return {}
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _write_deployment_tex(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    summary: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("onnx_exists", "0")) == "1":
            summary[str(row.get("model", ""))].append(row)
    lines = [
        "\\begin{tabular}{lrrr}\n",
        "\\toprule\n",
        "Model & Runs & MB & B1 p50 ms \\\\\n",
        "\\midrule\n",
    ]
    for model, model_rows in sorted(summary.items()):
        size_values = [_safe_float(row.get("model_size_mb", "")) for row in model_rows if str(row.get("model_size_mb", ""))]
        latency_values = [_safe_float(row.get("latency_b1_p50_ms", "")) for row in model_rows if str(row.get("latency_b1_p50_ms", ""))]
        lines.append(
            f"{_latex_escape(model)} & {len(model_rows)} & "
            f"{(statistics.fmean(size_values) if size_values else 0.0):.1f} & "
            f"{(statistics.fmean(latency_values) if latency_values else 0.0):.2f} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    path.write_text("".join(lines), encoding="utf-8")


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_tost_ci_summary(equivalence_csv: Path, out_csv: Path) -> None:
    if not equivalence_csv.exists():
        return
    rows = []
    for row in _read_csv(equivalence_csv):
        ci_low = _safe_float(row.get("ci_low", "nan"))
        ci_high = _safe_float(row.get("ci_high", "nan"))
        margin = abs(_safe_float(row.get("delta_margin", "0.03")))
        rows.append(
            {
                **row,
                "tost_alpha": "0.05",
                "tost_ci_equivalence_interpretation": int(ci_low >= -margin and ci_high <= margin),
                "noninferiority_ci_interpretation": int(ci_low >= -margin),
                "note": "CI decision is equivalent to two one-sided tests at alpha=0.05 for this bootstrap interval.",
            }
        )
    write_csv_rows(out_csv, rows)


def _merge_review_rows(manifest_rows: Sequence[Mapping[str, str]], review_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    by_key = {_roi_review_key(row): row for row in review_rows}
    merged: list[dict[str, str]] = []
    for row in manifest_rows:
        out = dict(row)
        review = by_key.get(_roi_review_key(row), {})
        out.update({key: value for key, value in review.items() if value != ""})
        merged.append(out)
    return merged


def _roi_review_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("participant", "")),
        str(row.get("video_or_source", row.get("video", ""))),
        str(row.get("frame_or_reference", row.get("artifact_path", ""))),
    )


def _roi_summary_row(rows: Sequence[Mapping[str, str]], manual_diff_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    valid_values = _metric_values(rows, ("human_valid", "roi_valid", "valid"))
    wrong_values = _metric_values(rows, ("wrong_panel", "human_wrong_panel", "roi_wrong_panel"))
    clipping_values = _metric_values(rows, ("clipping", "roi_clipped", "human_clipping"))
    corrections = [int(row.get("coords_changed", 0)) for row in manual_diff_rows if str(row.get("matched_current", "0")) == "1"]
    reviewed_rows = max(len(valid_values), len(wrong_values), len(clipping_values))
    return {
        "manifest_rows": len(rows),
        "reviewed_rows": reviewed_rows,
        "valid_count": _sum_or_blank(valid_values),
        "valid_rate": _rate_or_blank(valid_values),
        "wrong_panel_count": _sum_or_blank(wrong_values),
        "wrong_panel_rate": _rate_or_blank(wrong_values),
        "clipping_count": _sum_or_blank(clipping_values),
        "clipping_rate": _rate_or_blank(clipping_values),
        "manual_comparison_rows": len(corrections),
        "manual_correction_count": sum(corrections) if corrections else "",
        "manual_correction_rate": _rate(sum(corrections), len(corrections)) if corrections else "",
        "note": "" if reviewed_rows else "No human review result columns found; visual validity metrics intentionally left blank.",
    }


def _roi_by_participant_rows(rows: Sequence[Mapping[str, str]], manual_diff_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    participants = sorted({str(row.get("participant", "")) for row in rows} | {str(row.get("participant", "")) for row in manual_diff_rows})
    out = []
    for participant in participants:
        part_rows = [row for row in rows if str(row.get("participant", "")) == participant]
        part_diff = [row for row in manual_diff_rows if str(row.get("participant", "")) == participant]
        summary = _roi_summary_row(part_rows, part_diff)
        out.append({"participant": participant, **summary})
    return out


def _manual_vs_current_roi_rows(*, current_roi_dir: Path | str | None, manual_roi_csvs: Sequence[Path | str]) -> list[dict[str, object]]:
    if not current_roi_dir or not manual_roi_csvs:
        return []
    current_dir = Path(current_roi_dir)
    out: list[dict[str, object]] = []
    for manual_csv in manual_roi_csvs:
        manual_path = Path(manual_csv)
        if not manual_path.exists():
            continue
        stem = manual_path.name
        participant = stem.split("_", 1)[0]
        roi_type = "wheel" if "wheel" in stem else "gaze"
        current_path = current_dir / f"{participant}_{roi_type}_rois.current.csv"
        current_rows = _read_csv(current_path) if current_path.exists() else []
        current_by_video = {_normalize_video_key(row.get("video", "")): row for row in current_rows}
        for manual in _read_csv(manual_path):
            key = _normalize_video_key(manual.get("video", ""))
            current = current_by_video.get(key, {})
            coord_deltas = {
                field: abs(_safe_float(manual.get(field, "")) - _safe_float(current.get(field, "")))
                for field in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")
                if current
            }
            coords_changed = int(bool(coord_deltas) and any(delta > 1e-6 for delta in coord_deltas.values()))
            out.append(
                {
                    "participant": participant,
                    "roi_type": roi_type,
                    "video": manual.get("video", ""),
                    "normalized_video": key,
                    "manual_csv": str(manual_path),
                    "current_csv": str(current_path) if current_path.exists() else "",
                    "matched_current": int(bool(current)),
                    "coords_changed": coords_changed,
                    **{f"delta_{field}": f"{delta:.6f}" for field, delta in coord_deltas.items()},
                }
            )
    return out


def _load_wheel_maps(paths: Sequence[Path], *, workspace_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        participant = path.name.split("_", 1)[0] if path.name.startswith("p") else ""
        for row in _read_csv(path):
            wheel_csv = str(row.get("wheel_csv", ""))
            base = Path(wheel_csv).name
            full = _resolve_workspace_path(wheel_csv, workspace_root)
            stored = dict(row)
            stored["wheel_csv"] = str(full) if full else wheel_csv
            out[(participant, base)] = stored
            out[("", base)] = stored
    return out


def _load_segment_rows(paths: Sequence[Path]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _read_csv(path):
            key = str(row.get("segment_uid", "") or row.get("group_key", ""))
            if key:
                out[key] = dict(row)
    return out


def _select_wheel_state_row(path: Path, selector: str) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = _read_csv(path)
    if not rows:
        return {}
    text = str(selector or "last_row").strip()
    if text == "last_row":
        return rows[-1]
    if text.startswith("row:"):
        idx = max(0, min(len(rows) - 1, int(float(text.split(":", 1)[1]))))
        return rows[idx]
    try:
        target = float(text)
    except ValueError:
        return rows[-1]
    return min(rows, key=lambda row: abs(_safe_float(row.get("video_time_sec", row.get("time_sec", "0"))) - target))


def _nearest_detection_rows(
    det_rows: Sequence[Mapping[str, str]],
    *,
    target_frame: float,
    target_time: float,
) -> list[Mapping[str, str]]:
    if not det_rows:
        return []
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in det_rows:
        grouped[str(row.get("frame", ""))].append(row)

    def score(group_rows: Sequence[Mapping[str, str]]) -> float:
        row = group_rows[0]
        frame = _parse_finite_float(row.get("frame", "nan"))
        video_time = _parse_finite_float(row.get("video_time_sec", "nan"))
        if math.isfinite(target_frame) and math.isfinite(frame):
            return abs(frame - target_frame)
        if math.isfinite(target_time) and math.isfinite(video_time):
            return abs(video_time - target_time)
        return 0.0

    return list(min(grouped.values(), key=score))


def _find_current_wheel_roi_row(
    participant: str,
    *,
    video: Path | None,
    row: Mapping[str, str],
    workspace_root: Path,
) -> dict[str, str]:
    if not participant:
        return {}
    manifest = workspace_root / "artifacts" / "manifests" / "current" / f"{participant}_wheel_rois.current.csv"
    if not manifest.exists():
        return {}
    try:
        rows = _read_csv(manifest)
    except FileNotFoundError:
        return {}
    keys = {_normalize_video_key(row.get("video_path", ""))}
    if video:
        keys.add(_normalize_video_key(str(video)))
    for item in rows:
        if _normalize_video_key(item.get("video", "")) in keys:
            return item
    return {}


def _first_csv_row(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    try:
        rows = _read_csv(path)
    except FileNotFoundError:
        return {}
    return rows[0] if rows else {}


def _extract_roi_tuple(row: Mapping[str, str] | None) -> tuple[int, int, int, int] | None:
    if not row:
        return None
    vals = tuple(int(round(_safe_float(row.get(key, "nan")))) for key in ("roi_x1", "roi_y1", "roi_x2", "roi_y2"))
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals


def _format_roi_tuple(roi: tuple[int, int, int, int] | None) -> str:
    return ",".join(str(value) for value in roi) if roi else ""


def _parse_roi_string(raw: object) -> tuple[int, int, int, int] | None:
    parts = [part.strip() for part in str(raw or "").split(",")]
    if len(parts) != 4:
        return None
    try:
        vals = tuple(int(round(float(part))) for part in parts)
    except ValueError:
        return None
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals


def _suspect_single_panel_rois() -> set[tuple[int, int, int, int]]:
    return {
        (720, 0, 1440, 1080),
        (950, 300, 1650, 700),
        (960, 300, 1600, 700),
    }


def _read_video_frame(video: Path, frame_idx: int):
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


def _draw_wheel_review_frame(frame, det_rows: Sequence[Mapping[str, str]], row: Mapping[str, str]):
    out = frame.copy()
    if not det_rows:
        return out
    first = det_rows[0]
    roi = _extract_roi_tuple(first)
    roi_trusted = _truthy(row.get("roi_review_trusted", "1"))
    evidence_roi = _parse_roi_string(row.get("wheel_evidence_roi_coords", ""))
    if roi and roi_trusted:
        cv2.rectangle(out, (roi[0], roi[1]), (roi[2], roi[3]), (230, 215, 80), 2)
        cv2.putText(out, "ROI", (roi[0] + 6, max(22, roi[1] + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 215, 80), 2, cv2.LINE_AA)
    elif evidence_roi:
        cv2.rectangle(out, (evidence_roi[0], evidence_roi[1]), (evidence_roi[2], evidence_roi[3]), (80, 180, 245), 2)
        cv2.putText(
            out,
            "Wheel evidence",
            (evidence_roi[0] + 6, max(22, evidence_roi[1] + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 180, 245),
            2,
            cv2.LINE_AA,
        )
    visible_det_rows = det_rows
    if evidence_roi and not roi_trusted:
        visible_det_rows = [det for det in det_rows if _det_center_in_roi(det, evidence_roi)]
    for det in visible_det_rows:
        cls = str(det.get("class_name", det.get("class_id", ""))).lower()
        color = (70, 190, 80) if "hand" in cls or str(det.get("class_id", "")) == "0" else (40, 135, 245)
        label = "hand" if color == (70, 190, 80) else "wheel"
        x1 = int(round(_safe_float(det.get("x1", "0"))))
        y1 = int(round(_safe_float(det.get("y1", "0"))))
        x2 = int(round(_safe_float(det.get("x2", "0"))))
        y2 = int(round(_safe_float(det.get("y2", "0"))))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        text = f"{label} {_safe_float(det.get('confidence', '0')):.2f}"
        cv2.putText(out, text, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    caption = (
        f"row {row.get('participant', '')} {Path(str(row.get('source_csv', ''))).name} | "
        f"seed={_normalize_wheel_state(row.get('state', '')) or 'NA'} | "
        f"pipeline={_normalize_wheel_state(row.get('model_state', '')) or 'NA'} | "
        f"det_frame={first.get('frame', '')} video_t={first.get('video_time_sec', '')}"
    )
    if not roi_trusted:
        caption = f"ROI unverified: {row.get('roi_review_status', 'unknown')} | {caption}"
        if evidence_roi:
            caption = f"Wheel evidence candidate | {caption}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (30, 30, 30), -1)
    cv2.putText(out, caption[:180], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def _det_center_in_roi(det: Mapping[str, str], roi: tuple[int, int, int, int]) -> bool:
    x1 = _safe_float(det.get("x1", "0"))
    y1 = _safe_float(det.get("y1", "0"))
    x2 = _safe_float(det.get("x2", "0"))
    y2 = _safe_float(det.get("y2", "0"))
    if x2 <= x1 or y2 <= y1:
        return False
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def _extract_human_wheel_state(joined_row: Mapping[str, str], human_row: Mapping[str, str]) -> str:
    for row in (human_row, joined_row):
        for key in ("human_state", "human_hand_on_wheel", "human_label", "verified_state", "review_state"):
            if key in row and str(row.get(key, "")).strip():
                return _normalize_wheel_state(row.get(key, ""))
    return ""


def _wheel_by_participant_state(reviewed: Sequence[tuple[Mapping[str, str], str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for row, model, human in reviewed:
        grouped[(str(row.get("participant", "")), human)].append((model, human))
    out = []
    for (participant, human_state), vals in sorted(grouped.items()):
        correct = sum(1 for model, human in vals if model == human)
        out.append(
            {
                "participant": participant,
                "human_state": human_state,
                "reviewed_rows": len(vals),
                "agreement": correct,
                "agreement_rate": _rate(correct, len(vals)),
                "false_on": sum(1 for model, human in vals if model == "ON" and human == "OFF"),
                "false_off": sum(1 for model, human in vals if model == "OFF" and human == "ON"),
            }
        )
    return out


def _wheel_review_key(row: Mapping[str, str]) -> tuple[str, str]:
    return (str(row.get("participant", "")), Path(str(row.get("source_csv", row.get("wheel_csv", "")))).name)


def _resolve_wheel_review_row_video(row: Mapping[str, str], *, workspace_root: Path) -> Path | None:
    raw = str(row.get("video_path", "")).strip()
    if not raw:
        return None
    path = _resolve_workspace_path(raw, workspace_root)
    if path and path.exists() and path.is_file():
        return path
    return None


def _resolve_video_from_det_csv(det_csv: Path, *, workspace_root: Path) -> Path | None:
    try:
        rows = _read_csv(det_csv)
    except FileNotFoundError:
        return None
    for row in rows:
        raw = str(row.get("video_path", "")).strip()
        if not raw:
            continue
        path = _resolve_workspace_path(raw, workspace_root)
        if path and path.exists() and path.is_file():
            return path
    return None


def _normalize_wheel_state(raw: object) -> str:
    text = str(raw).strip().upper()
    if text in {"1", "TRUE", "YES", "ON", "HAND_ON_WHEEL"}:
        return "ON"
    if text in {"0", "FALSE", "NO", "OFF", "HAND_OFF_WHEEL"}:
        return "OFF"
    if "UNCERTAIN" in text or text in {"", "NAN", "NONE"}:
        return "UNCERTAIN" if text else ""
    return text


def _state_to_binary(state: str) -> str:
    if state == "ON":
        return "1"
    if state == "OFF":
        return "0"
    return ""


def _metric_values(rows: Sequence[Mapping[str, str]], keys: Sequence[str]) -> list[int]:
    values: list[int] = []
    for row in rows:
        key = next((name for name in keys if str(row.get(name, "")).strip() != ""), "")
        if key:
            values.append(int(_truthy(row.get(key, ""))))
    return values


def _truthy(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "valid", "on", "correct"}


def _sum_or_blank(values: Sequence[int]) -> int | str:
    return sum(values) if values else ""


def _rate_or_blank(values: Sequence[int]) -> str:
    return _rate(sum(values), len(values)) if values else ""


def _rate(num: int, den: int) -> str:
    return f"{(num / den):.6f}" if den else ""


def _format_counter(counter: Counter[str]) -> str:
    return ";".join(f"{key}={counter[key]}" for key in sorted(counter))


def _normalize_video_key(raw: object) -> str:
    text = str(raw).replace("\\", "/").strip()
    for marker in ("data/natural_driving_p1/", "data/natural_driving/"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:]
    return text.lstrip("/")


def _resolve_workspace_path(raw: str, workspace_root: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [workspace_root / path, REPO_ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return workspace_root / path


def _parse_range_header(header: str | None, size: int) -> tuple[int, int]:
    if size <= 0:
        return 0, 0
    if not header or not header.startswith("bytes="):
        return 0, size - 1
    spec = header.split("=", 1)[1].split(",", 1)[0].strip()
    if "-" not in spec:
        return 0, size - 1
    start_text, end_text = spec.split("-", 1)
    try:
        if start_text == "":
            suffix = max(0, int(end_text))
            return max(0, size - suffix), size - 1
        start = max(0, int(start_text))
        end = int(end_text) if end_text else size - 1
    except ValueError:
        return 0, size - 1
    end = min(size - 1, max(start, end))
    return min(start, size - 1), end


def _wheel_review_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hand-on-Wheel Review</title>
  <style>
    :root {
      color-scheme: light;
      --paper: oklch(0.975 0.006 80);
      --ink: oklch(0.21 0.012 80);
      --muted: oklch(0.48 0.014 80);
      --line: oklch(0.86 0.012 80);
      --accent: oklch(0.58 0.16 37);
      --accent-soft: oklch(0.92 0.04 37);
      --ok: oklch(0.56 0.14 148);
      --bad: oklch(0.56 0.16 24);
      --warn: oklch(0.62 0.13 82);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--paper);
    }
    .shell {
      display: grid;
      grid-template-columns: minmax(240px, 300px) minmax(360px, 1fr) minmax(260px, 330px);
      height: 100vh;
    }
    aside, main, .judge {
      min-height: 0;
      border-color: var(--line);
    }
    aside {
      display: grid;
      grid-template-rows: auto auto 1fr;
      border-right: 1px solid var(--line);
      background: oklch(0.955 0.009 80);
    }
    header {
      padding: 18px 18px 12px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.15;
      font-weight: 760;
      letter-spacing: 0;
    }
    .meta {
      margin-top: 8px;
      display: flex;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .tools {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
    }
    button, a.button {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: oklch(0.99 0.004 80);
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      display: inline-grid;
      place-items: center;
      transition: background 160ms ease-out, border-color 160ms ease-out, color 160ms ease-out;
    }
    button:hover, a.button:hover { border-color: var(--accent); }
    button.primary, a.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: oklch(0.99 0.004 80);
    }
    .rows {
      overflow: auto;
      padding: 8px;
    }
    .row {
      width: 100%;
      min-height: 58px;
      display: grid;
      grid-template-columns: 34px 1fr auto;
      align-items: center;
      gap: 8px;
      padding: 8px;
      border: 1px solid transparent;
      border-radius: 8px;
      background: transparent;
      text-align: left;
    }
    .row:hover { background: oklch(0.98 0.004 80); }
    .row.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .row .idx {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-size: 12px;
      font-weight: 800;
      background: oklch(0.91 0.01 80);
    }
    .row .title {
      font-size: 13px;
      font-weight: 760;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .row .sub {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .pill {
      min-width: 40px;
      height: 24px;
      border-radius: 999px;
      display: grid;
      place-items: center;
      padding: 0 8px;
      font-size: 11px;
      font-weight: 850;
      color: oklch(0.99 0.004 80);
      background: var(--muted);
    }
    .pill.on { background: var(--ok); }
    .pill.off { background: var(--bad); }
    .pill.uncertain { background: var(--warn); color: var(--ink); }
    .pill.roi-warn { background: var(--warn); color: var(--ink); }
    main {
      display: grid;
      grid-template-rows: auto minmax(280px, 1fr) auto;
      min-width: 0;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
    }
    .source {
      min-width: 0;
    }
    .source strong {
      display: block;
      font-size: 15px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .source span {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .statusline {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }
    .viewer {
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 18px;
      background: oklch(0.91 0.007 80);
    }
    .review-frame {
      width: 100%;
      height: auto;
      max-height: calc(100vh - 210px);
      object-fit: contain;
      background: oklch(0.14 0.01 80);
      border-radius: 8px;
      outline: 1px solid oklch(0.24 0.01 80);
    }
    .timeline {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1px;
      border-top: 1px solid var(--line);
      background: var(--line);
    }
    .metric {
      padding: 12px 16px;
      background: oklch(0.965 0.006 80);
      min-width: 0;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 760;
      text-transform: uppercase;
    }
    .metric strong {
      display: block;
      margin-top: 4px;
      font-size: 14px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .judge {
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      border-left: 1px solid var(--line);
      background: oklch(0.965 0.006 80);
    }
    .judge h2 {
      margin: 0;
      padding: 18px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      line-height: 1.2;
    }
    .choices {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }
    .choice {
      height: 48px;
      font-size: 15px;
      justify-content: start;
      padding: 0 14px;
    }
    .choice.on.active { background: var(--ok); border-color: var(--ok); color: oklch(0.99 0.004 80); }
    .choice.off.active { background: var(--bad); border-color: var(--bad); color: oklch(0.99 0.004 80); }
    .choice.uncertain.active { background: var(--warn); border-color: var(--warn); color: var(--ink); }
    .notes {
      padding: 16px 18px;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 780;
      margin-bottom: 8px;
    }
    textarea {
      width: 100%;
      min-height: 150px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      color: var(--ink);
      background: oklch(0.99 0.004 80);
      font: inherit;
      font-size: 13px;
      line-height: 1.35;
    }
    .footer {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      padding: 16px 18px;
      border-top: 1px solid var(--line);
    }
    .save-state {
      min-height: 18px;
      color: var(--muted);
      font-size: 12px;
    }
    .empty {
      color: var(--muted);
      font-weight: 700;
      text-align: center;
    }
    @media (max-width: 980px) {
      .shell {
        grid-template-columns: 1fr;
        grid-template-rows: 230px minmax(420px, 1fr) auto;
        height: auto;
        min-height: 100vh;
      }
      aside, .judge { border: 0; }
      aside { border-bottom: 1px solid var(--line); }
      .judge { border-top: 1px solid var(--line); }
      .review-frame { max-height: 58vh; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <header>
        <h1>Hand-on-Wheel Review</h1>
        <div class="meta"><span id="progress">0 / 0</span><span id="saveState">Not saved</span></div>
      </header>
      <div class="tools">
        <button id="prevBtn" type="button">Prev</button>
        <button id="nextBtn" type="button">Next</button>
      </div>
      <div id="rows" class="rows"></div>
    </aside>
    <main>
      <div class="topbar">
        <div class="source">
          <strong id="sourceTitle">Loading</strong>
          <span id="sourcePath"></span>
        </div>
        <div class="statusline">
          <span id="roiPill" class="pill roi-warn">ROI</span>
          <span id="modelPill" class="pill">PIPELINE</span>
          <span id="humanPill" class="pill uncertain">HUMAN</span>
        </div>
      </div>
      <div class="viewer">
        <img id="frameImage" class="review-frame" alt="GroundingDINO frame with hand and wheel detections">
        <div id="empty" class="empty" hidden>No GroundingDINO annotated frame for this row</div>
      </div>
      <div class="timeline">
        <div class="metric"><span>Target time</span><strong id="targetTime"></strong></div>
        <div class="metric"><span>Segment</span><strong id="segmentTime"></strong></div>
        <div class="metric"><span>Frame</span><strong id="frameValue"></strong></div>
        <div class="metric"><span>Seed label</span><strong id="expectedValue"></strong></div>
      </div>
    </main>
    <section class="judge">
      <h2>Final visible state</h2>
      <div class="choices">
        <button class="choice on" type="button" data-state="ON">ON</button>
        <button class="choice off" type="button" data-state="OFF">OFF</button>
        <button class="choice uncertain" type="button" data-state="UNCERTAIN">UNCERTAIN</button>
      </div>
      <div class="notes">
        <label for="notes">Notes</label>
        <textarea id="notes"></textarea>
      </div>
      <div class="footer">
        <button id="saveBtn" type="button" class="primary">Save CSV</button>
        <a id="downloadBtn" class="button" href="/api/export">Download CSV</a>
        <div class="save-state" id="detailState"></div>
      </div>
    </section>
  </div>
  <script>
    const state = {
      rows: [],
      current: 0,
      reviews: new Map(),
      saveTimer: null
    };
    const el = id => document.getElementById(id);
    const fmt = value => {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(2) + "s" : "";
    };
    const keyOf = row => row.participant + "::" + row.source_csv;
    const load = async () => {
      const response = await fetch("/api/rows", {cache: "no-store"});
      const payload = await response.json();
      state.rows = payload.rows || [];
      state.rows.forEach(row => {
        if (row.human_state) {
          state.reviews.set(keyOf(row), {
            participant: row.participant,
            source_csv: row.source_csv,
            human_state: row.human_state,
            human_notes: row.human_notes || ""
          });
        }
      });
      renderList();
      selectRow(0);
      updateProgress();
    };
    const renderList = () => {
      const host = el("rows");
      host.innerHTML = "";
      state.rows.forEach((row, idx) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "row";
        button.dataset.index = String(idx);
        button.innerHTML = `
          <span class="idx">${idx + 1}</span>
          <span>
            <span class="title">${escapeHtml(row.participant)} ${escapeHtml(row.segment_uid || row.source_csv)}</span>
            <span class="sub">${escapeHtml(row.source_csv)}</span>
          </span>
          <span class="pill ${pillClass(reviewFor(row).human_state)}">${escapeHtml(reviewFor(row).human_state || "")}</span>
        `;
        button.addEventListener("click", () => selectRow(idx));
        host.appendChild(button);
      });
    };
    const selectRow = idx => {
      if (!state.rows.length) return;
      state.current = Math.max(0, Math.min(state.rows.length - 1, idx));
      const row = state.rows[state.current];
      document.querySelectorAll(".row").forEach(node => node.classList.toggle("active", Number(node.dataset.index) === state.current));
      el("sourceTitle").textContent = `${row.participant} ${row.source_csv}`;
      el("sourcePath").textContent = row.video_path || "";
      el("targetTime").textContent = fmt(row.model_video_time_sec);
      el("segmentTime").textContent = `${fmt(row.segment_start_sec)} to ${fmt(row.segment_end_sec)}`;
      el("frameValue").textContent = row.model_frame || "";
      el("expectedValue").textContent = row.expected_state || "";
      setRoiPill(row);
      setPill(el("modelPill"), row.model_state ? `Pipeline stable_state: ${row.model_state}` : "Pipeline stable_state: missing");
      const review = reviewFor(row);
      setPill(el("humanPill"), review.human_state || "HUMAN");
      el("notes").value = review.human_notes || "";
      document.querySelectorAll(".choice").forEach(button => button.classList.toggle("active", button.dataset.state === review.human_state));
      const image = el("frameImage");
      const empty = el("empty");
      if (row.frame_url) {
        empty.hidden = true;
        image.hidden = false;
        image.src = row.frame_url + "?v=" + Date.now();
      } else {
        image.removeAttribute("src");
        image.hidden = true;
        empty.hidden = false;
      }
      updateProgress();
    };
    const reviewFor = row => state.reviews.get(keyOf(row)) || {
      participant: row.participant,
      source_csv: row.source_csv,
      human_state: "",
      human_notes: ""
    };
    const setReview = patch => {
      const row = state.rows[state.current];
      const review = {...reviewFor(row), ...patch};
      state.reviews.set(keyOf(row), review);
      renderList();
      selectRow(state.current);
      scheduleSave();
    };
    const updateCurrentNotes = value => {
      const row = state.rows[state.current];
      const review = {...reviewFor(row), human_notes: value};
      state.reviews.set(keyOf(row), review);
      scheduleSave();
    };
    const save = async () => {
      el("saveState").textContent = "Saving";
      const reviews = Array.from(state.reviews.values()).filter(row => row.human_state);
      const response = await fetch("/api/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({reviews})
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      el("saveState").textContent = "Saved";
      el("detailState").textContent = `${payload.saved_rows} rows saved to ${payload.out_csv}`;
      updateProgress();
    };
    const scheduleSave = () => {
      clearTimeout(state.saveTimer);
      el("saveState").textContent = "Pending";
      state.saveTimer = setTimeout(() => save().catch(showError), 450);
    };
    const updateProgress = () => {
      const done = Array.from(state.reviews.values()).filter(row => row.human_state).length;
      el("progress").textContent = `${done} / ${state.rows.length}`;
    };
    const setPill = (node, text) => {
      node.textContent = text || "";
      node.className = "pill " + pillClass(text);
    };
    const setRoiPill = row => {
      const node = el("roiPill");
      const trusted = Number(row.roi_review_trusted) === 1;
      const coords = trusted ? row.roi_coords : (row.wheel_evidence_roi_coords || row.roi_coords);
      const coordText = coords ? ` ${coords}` : "";
      node.textContent = trusted ? `ROI verified${coordText}` : `Wheel evidence candidate${coordText}`;
      node.title = [
        `roi=${row.roi_review_source || "unknown"} ${row.roi_review_status || ""} ${row.roi_review_note || ""}`,
        `evidence=${row.wheel_evidence_roi_source || "unknown"} ${row.wheel_evidence_roi_status || ""} ${row.wheel_evidence_roi_note || ""}`
      ].join(" | ").trim();
      node.className = "pill " + (trusted ? "on" : "roi-warn");
    };
    const pillClass = text => String(text || "").toLowerCase();
    const escapeHtml = text => String(text || "").replace(/[&<>"']/g, ch => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    }[ch]));
    const showError = err => {
      el("saveState").textContent = "Error";
      el("detailState").textContent = String(err.message || err);
    };
    document.querySelectorAll(".choice").forEach(button => {
      button.addEventListener("click", () => setReview({human_state: button.dataset.state, human_notes: el("notes").value}));
    });
    el("notes").addEventListener("input", () => updateCurrentNotes(el("notes").value));
    el("prevBtn").addEventListener("click", () => selectRow(state.current - 1));
    el("nextBtn").addEventListener("click", () => selectRow(state.current + 1));
    el("saveBtn").addEventListener("click", () => save().catch(showError));
    window.addEventListener("keydown", event => {
      if (event.target && ["TEXTAREA", "INPUT"].includes(event.target.tagName)) return;
      if (event.key === "ArrowLeft") selectRow(state.current - 1);
      if (event.key === "ArrowRight") selectRow(state.current + 1);
      if (event.key === "1") setReview({human_state: "ON", human_notes: el("notes").value});
      if (event.key === "2") setReview({human_state: "OFF", human_notes: el("notes").value});
      if (event.key === "3") setReview({human_state: "UNCERTAIN", human_notes: el("notes").value});
    });
    load().catch(showError);
  </script>
</body>
</html>
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _append_csv(path: Path, row: Mapping[str, object], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _load_success_names(status_path: Path) -> set[str]:
    if not status_path.exists():
        return set()
    return {row["run_name"] for row in _read_csv(status_path) if row.get("status") == "success"}


def _run_checked(cmd: Sequence[str], *, timeout: int) -> None:
    proc = subprocess.run([str(x) for x in cmd], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)


def _write_command_script(path: Path, commands: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _safe_float(raw: object) -> float:
    try:
        value = float(str(raw).strip())
        return value if math.isfinite(value) else 0.0
    except Exception:
        return 0.0


def _parse_finite_float(raw: object) -> float:
    try:
        value = float(str(raw).strip())
    except Exception:
        return math.nan
    return value if math.isfinite(value) else math.nan


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


__all__ = [
    "DEFAULT_AUTOWIP_ROOT",
    "DEFAULT_BUDGETS",
    "DEFAULT_FEWSHOT_ROOT",
    "DEFAULT_MODELS",
    "DEFAULT_MULTI_SEED_ROOT",
    "DEFAULT_SEEDS",
    "DEFAULT_TARGETS",
    "ExperimentModel",
    "build_parser",
    "default_participant_datasets",
    "join_wheel_validation",
    "main",
    "prepare_fewshot_datasets",
    "prepare_lopo_matrix",
    "prepare_wheel_review_rows",
    "resolve_wheel_review_video",
    "run_predict_matrix",
    "run_stats",
    "run_train_matrix",
    "run_deployment_matrix",
    "serve_wheel_review",
    "summarize_roi_audit",
    "summarize_wheel_validation",
    "write_wheel_review_results",
    "write_deployment_matrix_report",
    "write_fewshot_curve_report",
    "write_matrix_report",
]
