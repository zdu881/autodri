from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from autodri.aoi.equivalence import (
    DEFAULT_DATASETS,
    DEFAULT_LABELS,
    DEFAULT_SEEDS,
    PredictionRow,
    assign_internal_validation,
    compute_top1_parity,
    confusion_matrix,
    default_model_specs,
    generate_run_matrix,
    holm_adjust,
    load_split_manifest,
    mcnemar_exact,
    metrics_by_run,
    paired_bootstrap_delta,
    read_predictions_csv,
    select_family_best,
    summarize_latency,
    validate_split_integrity,
    write_csv_rows,
)
from autodri.common.paths import resolve_workspace_or_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, evaluate, and summarize AOI classifier equivalence experiments."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("make-plan", help="Create run matrix, split assignments, and command templates")
    p_plan.add_argument("--dataset", action="append", default=[], help="Dataset item as name=path")
    p_plan.add_argument("--out-dir", default="artifacts/aoi_equivalence_plan")
    p_plan.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p_plan.add_argument("--internal-val-ratio", type=float, default=0.2)
    p_plan.add_argument("--strict-datasets", action="store_true", help="Fail if a default dataset is missing")
    p_plan.set_defaults(func=cmd_make_plan)

    p_predict_torch = sub.add_parser("predict-torch", help="Run a trained torch/timm checkpoint on frozen manifest rows")
    p_predict_torch.add_argument("--checkpoint", required=True)
    p_predict_torch.add_argument("--data", required=True)
    p_predict_torch.add_argument("--dataset-name", required=True)
    p_predict_torch.add_argument("--out-csv", required=True)
    p_predict_torch.add_argument("--manifest-split", default="val")
    p_predict_torch.add_argument("--model-name", default="")
    p_predict_torch.add_argument("--seed", type=int, default=0)
    p_predict_torch.add_argument("--batch", type=int, default=64)
    p_predict_torch.add_argument("--device", default="auto")
    p_predict_torch.set_defaults(func=cmd_predict_torch)

    p_predict_yolo = sub.add_parser("predict-yolo", help="Run a YOLOv8-cls checkpoint on frozen manifest rows")
    p_predict_yolo.add_argument("--weights", required=True)
    p_predict_yolo.add_argument("--data", required=True)
    p_predict_yolo.add_argument("--dataset-name", required=True)
    p_predict_yolo.add_argument("--model-name", required=True)
    p_predict_yolo.add_argument("--out-csv", required=True)
    p_predict_yolo.add_argument("--manifest-split", default="val")
    p_predict_yolo.add_argument("--seed", type=int, default=0)
    p_predict_yolo.add_argument("--batch", type=int, default=64)
    p_predict_yolo.add_argument("--imgsz", type=int, default=224)
    p_predict_yolo.add_argument("--device", default="cpu")
    p_predict_yolo.set_defaults(func=cmd_predict_yolo)

    p_predict_onnx = sub.add_parser("predict-onnx", help="Run an exported AOI ONNX model on frozen manifest rows")
    p_predict_onnx.add_argument("--onnx", required=True)
    p_predict_onnx.add_argument("--data", required=True)
    p_predict_onnx.add_argument("--dataset-name", required=True)
    p_predict_onnx.add_argument("--model-name", required=True)
    p_predict_onnx.add_argument("--out-csv", required=True)
    p_predict_onnx.add_argument("--manifest-split", default="val")
    p_predict_onnx.add_argument("--seed", type=int, default=0)
    p_predict_onnx.add_argument("--imgsz", type=int, default=224)
    p_predict_onnx.add_argument("--batch", type=int, default=64)
    p_predict_onnx.add_argument("--labels-json", default="")
    p_predict_onnx.add_argument("--providers", nargs="+", default=[])
    p_predict_onnx.set_defaults(func=cmd_predict_onnx)

    p_metrics = sub.add_parser("metrics", help="Compute frame/event metrics from prediction CSV files")
    p_metrics.add_argument("--predictions", nargs="+", required=True)
    p_metrics.add_argument("--out-dir", required=True)
    p_metrics.add_argument("--event-window-sec", type=float, default=30.0)
    p_metrics.set_defaults(func=cmd_metrics)

    p_equiv = sub.add_parser("equivalence", help="Compare non-YOLO models against YOLO family-best")
    p_equiv.add_argument("--predictions", nargs="+", required=True)
    p_equiv.add_argument("--out-dir", required=True)
    p_equiv.add_argument("--metric", default="primary3_macro_f1")
    p_equiv.add_argument("--delta-margin", type=float, default=0.03)
    p_equiv.add_argument("--n-boot", type=int, default=1000)
    p_equiv.add_argument("--seed", type=int, default=42)
    p_equiv.set_defaults(func=cmd_equivalence)

    p_parity = sub.add_parser("parity", help="Compute top1 parity between two prediction CSV files")
    p_parity.add_argument("--reference-predictions", required=True)
    p_parity.add_argument("--candidate-predictions", required=True)
    p_parity.add_argument("--out-csv", required=True)
    p_parity.set_defaults(func=cmd_parity)

    p_bench = sub.add_parser("benchmark-onnx", help="Benchmark one exported ONNX model with random input")
    p_bench.add_argument("--onnx", required=True)
    p_bench.add_argument("--model-name", required=True)
    p_bench.add_argument("--out-csv", required=True)
    p_bench.add_argument("--input-shape", default="1,3,224,224")
    p_bench.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    p_bench.add_argument("--warmup", type=int, default=10)
    p_bench.add_argument("--iterations", type=int, default=100)
    p_bench.add_argument("--providers", nargs="+", default=[])
    p_bench.set_defaults(func=cmd_benchmark_onnx)

    return parser


def cmd_make_plan(args: argparse.Namespace) -> None:
    datasets = _parse_datasets(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_specs = generate_run_matrix(sorted(datasets), seeds=args.seeds, specs=default_model_specs())

    matrix_rows = []
    for run in run_specs:
        data_dir = datasets[run.dataset]
        run_name = f"{run.dataset}_{run.model}_seed{run.seed}"
        if run.trainer == "ultralytics":
            train_cmd = (
                "python -m autodri.cli.train_gaze_cls "
                f"--data {data_dir} --model {run.base_model} --name {run_name} --project {out_dir / 'runs_yolo'} "
                f"--seed {run.seed}"
            )
        else:
            train_cmd = (
                "python -m autodri.cli.train_aoi_backbone "
                f"--data {data_dir} --model {run.base_model} --name {run_name} --project {out_dir / 'runs_torch'} "
                f"--seed {run.seed} --export-onnx"
            )
        matrix_rows.append(
            {
                "dataset": run.dataset,
                "data_dir": data_dir,
                "seed": run.seed,
                "model": run.model,
                "family": run.family,
                "arch_group": run.arch_group,
                "trainer": run.trainer,
                "base_model": run.base_model,
                "run_name": run_name,
                "train_command": train_cmd,
            }
        )

    write_csv_rows(out_dir / "run_matrix.csv", matrix_rows)
    _write_commands(out_dir / "commands.sh", [str(row["train_command"]) for row in matrix_rows])

    status_rows = []
    integrity_rows = []
    split_dir = out_dir / "split_assignments"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, raw_path in sorted(datasets.items()):
        data_dir = resolve_workspace_or_repo_path(raw_path)
        manifest = data_dir / "split_manifest.csv"
        exists = manifest.exists()
        if args.strict_datasets and not exists:
            raise FileNotFoundError(f"Missing split manifest for {name}: {manifest}")
        status_rows.append({"dataset": name, "data_dir": str(data_dir), "split_manifest": str(manifest), "exists": int(exists)})
        if not exists:
            continue
        samples = load_split_manifest(manifest)
        assignment = assign_internal_validation(samples, val_ratio=args.internal_val_ratio, seed=42)
        report = validate_split_integrity(samples, assignment)
        integrity_rows.append({"dataset": name, **report})
        rows = [{"dst_rel": sample.dst_rel, "original_split": sample.split, "assigned_split": assignment[sample.dst_rel]} for sample in samples]
        write_csv_rows(split_dir / f"{name}.csv", rows, fieldnames=["dst_rel", "original_split", "assigned_split"])

    write_csv_rows(out_dir / "dataset_status.csv", status_rows)
    write_csv_rows(out_dir / "integrity_report.csv", integrity_rows)
    _write_engineering_rubric(out_dir / "engineering_rubric.csv")
    print(f"Wrote AOI equivalence plan to {out_dir}")


def cmd_metrics(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    rows = _load_prediction_files(args.predictions)
    metrics = metrics_by_run(rows, event_window_sec=args.event_window_sec)
    write_csv_rows(out_dir / "metrics.csv", metrics)

    grouped: dict[tuple[str, str, int, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.dataset, row.split, row.seed, row.model)].append(row)
    cm_dir = out_dir / "confusion_matrices"
    for (dataset, split, seed, model), run_rows in grouped.items():
        filename = f"{dataset}__{split}__seed{seed}__{model}.csv".replace("/", "_")
        write_csv_rows(cm_dir / filename, confusion_matrix(run_rows), fieldnames=["label", *DEFAULT_LABELS])
    print(f"Wrote metrics to {out_dir / 'metrics.csv'}")


def cmd_predict_torch(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from autodri.workflows.train_aoi_backbone import AoiManifestDataset, _collate_batch, _resolve_device, build_model

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    labels = list(checkpoint.get("labels", DEFAULT_LABELS))
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    model_name = str(checkpoint["model_name"])
    device = _resolve_device(args.device)
    model = build_model(model_name, num_classes=len(labels), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    data_dir = Path(args.data)
    samples = [sample for sample in load_split_manifest(data_dir / "split_manifest.csv") if sample.split == args.manifest_split]
    dataset = AoiManifestDataset(data_dir, samples, class_to_idx, imgsz=int(checkpoint.get("imgsz", 224)), train=False)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=_collate_batch)

    out_rows = []
    with torch.no_grad():
        for images, _, batch_samples in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().tolist()
            for sample, pred_idx in zip(batch_samples, preds):
                out_rows.append(
                    _prediction_csv_row(
                        args.dataset_name,
                        args.manifest_split,
                        args.model_name or model_name,
                        args.seed,
                        sample,
                        labels[int(pred_idx)],
                    )
                )
    write_csv_rows(Path(args.out_csv), out_rows, fieldnames=_prediction_fieldnames())
    print(f"Wrote torch predictions to {args.out_csv}")


def cmd_predict_yolo(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    data_dir = Path(args.data)
    samples = [sample for sample in load_split_manifest(data_dir / "split_manifest.csv") if sample.split == args.manifest_split]
    image_paths = [str(data_dir / sample.dst_rel) for sample in samples]
    model = YOLO(args.weights)
    outputs = model.predict(source=image_paths, imgsz=args.imgsz, batch=args.batch, device=args.device, verbose=False)
    out_rows = []
    for sample, result in zip(samples, outputs):
        pred_idx = int(result.probs.top1)
        pred_name = str(result.names[pred_idx])
        out_rows.append(_prediction_csv_row(args.dataset_name, args.manifest_split, args.model_name, args.seed, sample, pred_name))
    write_csv_rows(Path(args.out_csv), out_rows, fieldnames=_prediction_fieldnames())
    print(f"Wrote YOLO predictions to {args.out_csv}")


def cmd_predict_onnx(args: argparse.Namespace) -> None:
    import json

    import numpy as np
    import onnxruntime as ort
    labels = list(DEFAULT_LABELS)
    if args.labels_json:
        labels = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
    data_dir = Path(args.data)
    samples = [sample for sample in load_split_manifest(data_dir / "split_manifest.csv") if sample.split == args.manifest_split]
    providers = args.providers or ort.get_available_providers()
    session = ort.InferenceSession(str(args.onnx), providers=providers)
    input_name = session.get_inputs()[0].name

    out_rows = []
    for start in range(0, len(samples), args.batch):
        batch_samples = samples[start : start + args.batch]
        batch = np.stack([_load_normalized_image(data_dir / sample.dst_rel, args.imgsz) for sample in batch_samples], axis=0)
        logits = session.run(None, {input_name: batch})[0]
        preds = np.asarray(logits).argmax(axis=1).tolist()
        for sample, pred_idx in zip(batch_samples, preds):
            out_rows.append(_prediction_csv_row(args.dataset_name, args.manifest_split, args.model_name, args.seed, sample, labels[int(pred_idx)]))
    write_csv_rows(Path(args.out_csv), out_rows, fieldnames=_prediction_fieldnames())
    print(f"Wrote ONNX predictions to {args.out_csv}")


def cmd_equivalence(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    rows = _load_prediction_files(args.predictions)
    yolo_models = [spec.name for spec in default_model_specs() if spec.family == "yolo"]
    yolo_best = select_family_best(rows, family_models=yolo_models, metric_name=args.metric)
    by_run: dict[tuple[str, str, str, int], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        by_run[(row.dataset, row.split, row.model, row.seed)].append(row)

    result_rows = []
    for (dataset, split), baseline_model in sorted(yolo_best.items()):
        seeds = sorted({seed for ds, sp, _, seed in by_run if ds == dataset and sp == split})
        for seed in seeds:
            baseline = by_run.get((dataset, split, baseline_model, seed), [])
            if not baseline:
                continue
            candidates = sorted(
                {
                    model
                    for ds, sp, model, sd in by_run
                    if ds == dataset and sp == split and sd == seed and model not in yolo_models
                }
            )
            for model in candidates:
                candidate = by_run[(dataset, split, model, seed)]
                result = paired_bootstrap_delta(
                    candidate,
                    baseline,
                    metric_name=args.metric,
                    delta_margin=args.delta_margin,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                mcnemar = mcnemar_exact(candidate, baseline)
                result_rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "seed": seed,
                        "candidate_model": model,
                        "baseline_model": baseline_model,
                        "metric": args.metric,
                        "observed_delta": f"{result.observed_delta:.6f}",
                        "ci_low": f"{result.ci_low:.6f}",
                        "ci_high": f"{result.ci_high:.6f}",
                        "delta_margin": f"{args.delta_margin:.6f}",
                        "noninferior": int(result.noninferior),
                        "equivalent": int(result.equivalent),
                        "mcnemar_candidate_only_correct": int(mcnemar["candidate_only_correct"]),
                        "mcnemar_baseline_only_correct": int(mcnemar["baseline_only_correct"]),
                        "mcnemar_p": f"{mcnemar['p_value']:.6f}",
                    }
                )

    if result_rows:
        adjusted = holm_adjust([float(row["mcnemar_p"]) for row in result_rows])
        for row, p_value in zip(result_rows, adjusted):
            row["mcnemar_holm_p"] = f"{p_value:.6f}"
    write_csv_rows(out_dir / "equivalence_results.csv", result_rows)
    _write_conclusion(out_dir / "conclusion.md", result_rows, metric=args.metric, margin=args.delta_margin)
    print(f"Wrote equivalence results to {out_dir / 'equivalence_results.csv'}")


def cmd_parity(args: argparse.Namespace) -> None:
    reference = read_predictions_csv(Path(args.reference_predictions))
    candidate = read_predictions_csv(Path(args.candidate_predictions))
    parity = compute_top1_parity(reference, candidate)
    rows = [{key: f"{value:.6f}" for key, value in parity.items()}]
    write_csv_rows(Path(args.out_csv), rows)
    print(f"Wrote top1 parity to {args.out_csv}")


def cmd_benchmark_onnx(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    onnx_path = Path(args.onnx)
    base_shape = [int(x) for x in str(args.input_shape).split(",")]
    if len(base_shape) != 4:
        raise ValueError("--input-shape must be N,C,H,W")
    providers = args.providers or ort.get_available_providers()
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    rows = []
    for batch in args.batch_sizes:
        shape = list(base_shape)
        shape[0] = int(batch)
        sample = np.random.default_rng(0).random(shape, dtype=np.float32)
        for _ in range(args.warmup):
            session.run(None, {input_name: sample})
        durations = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            session.run(None, {input_name: sample})
            durations.append(time.perf_counter() - start)
        summary = summarize_latency(durations, batch_size=batch)
        rows.append(
            {
                "model": args.model_name,
                "onnx": str(onnx_path),
                "model_size_mb": f"{onnx_path.stat().st_size / (1024 * 1024):.3f}",
                "providers": ",".join(providers),
                **{key: f"{value:.6f}" for key, value in summary.items()},
            }
        )
    write_csv_rows(Path(args.out_csv), rows)
    print(f"Wrote ONNX benchmark to {args.out_csv}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def _parse_datasets(items: Sequence[str]) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_DATASETS)
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --dataset item {item!r}; expected name=path")
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def _load_prediction_files(paths: Sequence[str]) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for path in paths:
        rows.extend(read_predictions_csv(Path(path)))
    if not rows:
        raise ValueError("No prediction rows loaded")
    return rows


def _prediction_fieldnames() -> list[str]:
    return ["dataset", "split", "model", "seed", "image_path", "label", "pred", "domain", "video", "timestamp"]


def _prediction_csv_row(dataset: str, split: str, model: str, seed: int, sample: object, pred: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "split": split,
        "model": model,
        "seed": seed,
        "image_path": sample.dst_rel,
        "label": sample.label,
        "pred": pred,
        "domain": sample.domain,
        "video": sample.video,
        "timestamp": f"{sample.timestamp:.6f}",
    }


def _load_normalized_image(path: Path, imgsz: int):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        arr = np.asarray(
            image.convert("RGB").resize((imgsz, imgsz), resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def _write_commands(path: Path, commands: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n"
    path.write_text(text, encoding="utf-8")


def _write_engineering_rubric(path: Path) -> None:
    rows = [
        {"criterion": "framework_unification", "preferred": "yolo", "measurement": "reuse of Ultralytics train/export/predict path"},
        {"criterion": "onnx_export", "preferred": "lower_failure_rate", "measurement": "export success and PyTorch-ONNX top1 parity"},
        {"criterion": "latency", "preferred": "lower", "measurement": "ONNX Runtime p50/p95 at batch=1 and batch=32"},
        {"criterion": "model_size", "preferred": "lower", "measurement": "ONNX file size in MB"},
        {"criterion": "custom_code", "preferred": "lower", "measurement": "runtime-specific preprocessing/head code required"},
    ]
    write_csv_rows(path, rows)


def _write_conclusion(path: Path, rows: Sequence[dict[str, object]], *, metric: str, margin: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    noninferior_splits = sum(
        1 for dataset_rows in by_dataset.values() if any(str(row.get("noninferior", "0")) == "1" for row in dataset_rows)
    )
    total_splits = len(by_dataset)
    decision = "inconclusive"
    if total_splits and noninferior_splits >= max(1, math.ceil(total_splits * 2 / 3)):
        decision = "supports_non_yolo_noninferiority"
    text = (
        "# AOI Equivalence Conclusion\n\n"
        f"- Metric: `{metric}`\n"
        f"- Non-inferiority margin: `{margin:.3f}`\n"
        f"- Splits with at least one non-YOLO non-inferior candidate: `{noninferior_splits}/{total_splits}`\n"
        f"- Decision: `{decision}`\n\n"
        "The primary statistical claim applies to `Forward`, `In-Car`, and `Non-Forward`. "
        "`Other` remains a secondary class because the current test sets have limited support.\n"
    )
    path.write_text(text, encoding="utf-8")


__all__ = ["build_parser", "main"]
