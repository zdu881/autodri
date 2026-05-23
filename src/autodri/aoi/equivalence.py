from __future__ import annotations

import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_LABELS = ("Forward", "In-Car", "Non-Forward", "Other")
PRIMARY_LABELS = ("Forward", "In-Car", "Non-Forward")
DEFAULT_DATASETS = {
    "stratified": "gaze_onnx/experiments/cls_dataset_two_domain_stratified_run1",
    "holdout_car1": "gaze_onnx/experiments/cls_dataset_two_domain_holdout_car1_genv3",
    "holdout_car2": "gaze_onnx/experiments/cls_dataset_two_domain_holdout_car2_genv3",
}
DEFAULT_SEEDS = (13, 29, 43, 71, 101)


@dataclass(frozen=True)
class ManifestSample:
    split: str
    label: str
    domain: str
    frame_id: int
    timestamp: float
    video: str
    src_rel: str
    dst_rel: str
    augmented: bool

    def group_key(self, window_sec: float = 30.0) -> tuple[str, str, int]:
        window = int(self.timestamp // window_sec) if window_sec > 0 else 0
        return (self.domain, self.video, window)


@dataclass(frozen=True)
class PredictionRow:
    dataset: str
    split: str
    model: str
    seed: int
    image_path: str
    label: str
    pred: str
    domain: str
    video: str
    timestamp: float

    def group_key(self, window_sec: float = 30.0) -> tuple[str, str, int]:
        window = int(self.timestamp // window_sec) if window_sec > 0 else 0
        return (self.domain, self.video, window)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    arch_group: str
    trainer: str
    base_model: str


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    seed: int
    model: str
    family: str
    arch_group: str
    trainer: str
    base_model: str


@dataclass(frozen=True)
class BootstrapResult:
    candidate_model: str
    baseline_model: str
    metric_name: str
    observed_delta: float
    ci_low: float
    ci_high: float
    delta_margin: float
    n_boot: int
    noninferior: bool
    equivalent: bool


def default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("yolov8n-cls", "yolo", "yolo", "ultralytics", "yolov8n-cls.pt"),
        ModelSpec("yolov8s-cls", "yolo", "yolo", "ultralytics", "yolov8s-cls.pt"),
        ModelSpec("yolov8m-cls", "yolo", "yolo", "ultralytics", "yolov8m-cls.pt"),
        ModelSpec("resnet50", "convnet", "resnet", "torchvision_timm", "resnet50"),
        ModelSpec("efficientnet_b0", "convnet", "efficientnet", "torchvision_timm", "efficientnet_b0"),
        ModelSpec("efficientnet_b3", "convnet", "efficientnet", "torchvision_timm", "efficientnet_b3"),
        ModelSpec("convnext_tiny", "convnet", "convnext", "torchvision_timm", "convnext_tiny"),
        ModelSpec("deit_tiny", "vit", "light_vit", "torchvision_timm", "deit_tiny_patch16_224"),
    ]


def generate_run_matrix(
    datasets: Sequence[str],
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    specs: Sequence[ModelSpec] | None = None,
) -> list[RunSpec]:
    model_specs = list(specs or default_model_specs())
    return [
        RunSpec(
            dataset=str(dataset),
            seed=int(seed),
            model=spec.name,
            family=spec.family,
            arch_group=spec.arch_group,
            trainer=spec.trainer,
            base_model=spec.base_model,
        )
        for dataset in datasets
        for seed in seeds
        for spec in model_specs
    ]


def load_split_manifest(path: Path) -> list[ManifestSample]:
    rows: list[ManifestSample] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "label", "domain", "frame_id", "timestamp", "video", "dst_rel", "augmented"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                ManifestSample(
                    split=str(row.get("split", "")).strip(),
                    label=str(row.get("label", "")).strip(),
                    domain=str(row.get("domain", "")).strip(),
                    frame_id=_safe_int(row.get("frame_id", "")),
                    timestamp=_safe_float(row.get("timestamp", "")),
                    video=str(row.get("video", "")).strip(),
                    src_rel=str(row.get("src_rel", "")).strip(),
                    dst_rel=str(row.get("dst_rel", "")).strip(),
                    augmented=str(row.get("augmented", "0")).strip() not in {"", "0", "false", "False"},
                )
            )
    return rows


def assign_internal_validation(
    samples: Sequence[ManifestSample],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    group_window_sec: float = 30.0,
) -> dict[str, str]:
    rng = random.Random(seed)
    val_ratio = max(0.01, min(0.5, float(val_ratio)))
    assignment: dict[str, str] = {}
    groups: dict[tuple[str, str, int], list[ManifestSample]] = defaultdict(list)

    for sample in samples:
        if sample.split == "val":
            assignment[sample.dst_rel] = "test"
        elif sample.split == "train":
            groups[sample.group_key(group_window_sec)].append(sample)
            assignment[sample.dst_rel] = "train"
        else:
            assignment[sample.dst_rel] = "train"

    eligible: list[tuple[str, str, int]] = []
    by_label: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for key, rows in groups.items():
        if any(row.augmented for row in rows):
            continue
        labels = sorted({row.label for row in rows})
        eligible.append(key)
        by_label[labels[0] if labels else ""].append(key)

    selected: set[tuple[str, str, int]] = set()
    for keys in by_label.values():
        keys = list(keys)
        if len(keys) < 2:
            continue
        rng.shuffle(keys)
        n_val = max(1, int(round(len(keys) * val_ratio)))
        n_val = min(n_val, len(keys) - 1)
        selected.update(keys[:n_val])

    if not selected and eligible:
        selected.add(rng.choice(eligible))

    for key in selected:
        for row in groups[key]:
            assignment[row.dst_rel] = "internal_val"
    return assignment


def validate_split_integrity(
    samples: Sequence[ManifestSample],
    assignment: Mapping[str, str],
    *,
    group_window_sec: float = 30.0,
) -> dict[str, int]:
    by_group: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    augmented_not_train = 0
    frozen_val_not_test = 0
    missing_assignment = 0

    for sample in samples:
        assigned = assignment.get(sample.dst_rel)
        if assigned is None:
            missing_assignment += 1
            continue
        by_group[sample.group_key(group_window_sec)].add(assigned)
        if sample.augmented and assigned != "train":
            augmented_not_train += 1
        if sample.split == "val" and assigned != "test":
            frozen_val_not_test += 1

    group_leaks = sum(1 for splits in by_group.values() if len(splits) > 1)
    return {
        "sample_count": len(samples),
        "group_count": len(by_group),
        "group_leak_count": group_leaks,
        "augmented_not_train_count": augmented_not_train,
        "frozen_val_not_test_count": frozen_val_not_test,
        "missing_assignment_count": missing_assignment,
    }


def compute_frame_metrics(
    rows: Sequence[PredictionRow],
    *,
    labels: Sequence[str] = DEFAULT_LABELS,
    primary_labels: Sequence[str] = PRIMARY_LABELS,
) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(_metrics_for_labels(rows, labels=primary_labels, prefix="primary3"))
    out.update(_metrics_for_labels(rows, labels=labels, prefix="all4"))
    return out


def compute_event_accuracy(
    rows: Sequence[PredictionRow],
    *,
    window_sec: float = 30.0,
    labels: Sequence[str] | None = None,
) -> tuple[float, int]:
    label_set = set(labels) if labels is not None else None
    grouped: dict[tuple[str, str, int], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        if label_set is not None and row.label not in label_set:
            continue
        grouped[row.group_key(window_sec)].append(row)

    total = 0
    correct = 0
    for group_rows in grouped.values():
        gt = majority_vote([row.label for row in group_rows])
        pred = majority_vote([row.pred for row in group_rows])
        total += 1
        correct += int(gt == pred)
    return (correct / total if total else 0.0, total)


def paired_bootstrap_delta(
    candidate_rows: Sequence[PredictionRow],
    baseline_rows: Sequence[PredictionRow],
    *,
    metric_name: str,
    delta_margin: float = 0.03,
    n_boot: int = 1000,
    seed: int = 42,
    window_sec: float = 30.0,
) -> BootstrapResult:
    candidate_by_image = {row.image_path: row for row in candidate_rows}
    baseline_by_image = {row.image_path: row for row in baseline_rows}
    image_ids = sorted(set(candidate_by_image) & set(baseline_by_image))
    if not image_ids:
        raise ValueError("No overlapping image_path values between candidate and baseline predictions")

    aligned_candidate = [candidate_by_image[k] for k in image_ids]
    aligned_baseline = [baseline_by_image[k] for k in image_ids]
    observed = _metric_value(aligned_candidate, metric_name) - _metric_value(aligned_baseline, metric_name)

    block_to_images: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for image_id in image_ids:
        block_to_images[candidate_by_image[image_id].group_key(window_sec)].append(image_id)
    blocks = sorted(block_to_images)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(int(n_boot)):
        picked = [rng.choice(blocks) for _ in blocks]
        boot_images = [image_id for block in picked for image_id in block_to_images[block]]
        cand = [candidate_by_image[image_id] for image_id in boot_images]
        base = [baseline_by_image[image_id] for image_id in boot_images]
        deltas.append(_metric_value(cand, metric_name) - _metric_value(base, metric_name))

    ci_low = percentile(deltas, 2.5)
    ci_high = percentile(deltas, 97.5)
    return BootstrapResult(
        candidate_model=aligned_candidate[0].model,
        baseline_model=aligned_baseline[0].model,
        metric_name=metric_name,
        observed_delta=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        delta_margin=float(delta_margin),
        n_boot=int(n_boot),
        noninferior=ci_low >= -float(delta_margin),
        equivalent=ci_low >= -float(delta_margin) and ci_high <= float(delta_margin),
    )


def summarize_latency(durations_sec: Sequence[float], *, batch_size: int) -> dict[str, float]:
    if not durations_sec:
        raise ValueError("durations_sec must not be empty")
    durations_ms = [float(x) * 1000.0 for x in durations_sec]
    mean_sec = statistics.fmean(float(x) for x in durations_sec)
    return {
        "batch_size": float(batch_size),
        "latency_p50_ms": percentile(durations_ms, 50.0),
        "latency_p95_ms": percentile(durations_ms, 95.0),
        "latency_mean_ms": statistics.fmean(durations_ms),
        "throughput_img_s": float(batch_size) / mean_sec if mean_sec > 0 else 0.0,
    }


def compute_top1_parity(reference_rows: Sequence[PredictionRow], candidate_rows: Sequence[PredictionRow]) -> dict[str, float]:
    reference_by_image = {row.image_path: row for row in reference_rows}
    candidate_by_image = {row.image_path: row for row in candidate_rows}
    image_ids = sorted(set(reference_by_image) & set(candidate_by_image))
    if not image_ids:
        raise ValueError("No overlapping image_path values for parity calculation")
    matches = sum(1 for image_id in image_ids if reference_by_image[image_id].pred == candidate_by_image[image_id].pred)
    return {
        "aligned_total": float(len(image_ids)),
        "top1_matches": float(matches),
        "top1_parity": matches / len(image_ids),
    }


def mcnemar_exact(candidate_rows: Sequence[PredictionRow], baseline_rows: Sequence[PredictionRow]) -> dict[str, float]:
    candidate_by_image = {row.image_path: row for row in candidate_rows}
    baseline_by_image = {row.image_path: row for row in baseline_rows}
    image_ids = sorted(set(candidate_by_image) & set(baseline_by_image))
    if not image_ids:
        raise ValueError("No overlapping image_path values for McNemar test")
    candidate_only = 0
    baseline_only = 0
    for image_id in image_ids:
        cand = candidate_by_image[image_id]
        base = baseline_by_image[image_id]
        cand_correct = cand.pred == cand.label
        base_correct = base.pred == base.label
        if cand_correct and not base_correct:
            candidate_only += 1
        elif base_correct and not cand_correct:
            baseline_only += 1
    discordant = candidate_only + baseline_only
    p_value = 1.0
    if discordant:
        low_tail = sum(math.comb(discordant, k) for k in range(0, min(candidate_only, baseline_only) + 1))
        p_value = min(1.0, 2.0 * low_tail * (0.5 ** discordant))
    return {
        "aligned_total": float(len(image_ids)),
        "candidate_only_correct": float(candidate_only),
        "baseline_only_correct": float(baseline_only),
        "discordant_total": float(discordant),
        "p_value": float(p_value),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(indexed)
    running_max = 0.0
    m = len(indexed)
    for rank, (idx, p_value) in enumerate(indexed):
        raw = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, raw)
        adjusted[idx] = running_max
    return adjusted


def read_predictions_csv(path: Path) -> list[PredictionRow]:
    out: list[PredictionRow] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "split", "model", "seed", "image_path", "label", "pred", "domain", "video", "timestamp"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")
        for row in reader:
            out.append(
                PredictionRow(
                    dataset=str(row["dataset"]),
                    split=str(row["split"]),
                    model=str(row["model"]),
                    seed=_safe_int(row["seed"]),
                    image_path=str(row["image_path"]),
                    label=str(row["label"]),
                    pred=str(row["pred"]),
                    domain=str(row["domain"]),
                    video=str(row["video"]),
                    timestamp=_safe_float(row["timestamp"]),
                )
            )
    return out


def metrics_by_run(rows: Sequence[PredictionRow], *, event_window_sec: float = 30.0) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, int, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.dataset, row.split, row.seed, row.model)].append(row)

    out: list[dict[str, str]] = []
    for (dataset, split, seed, model), run_rows in sorted(grouped.items()):
        metrics = compute_frame_metrics(run_rows)
        event_acc, event_total = compute_event_accuracy(run_rows, window_sec=event_window_sec, labels=PRIMARY_LABELS)
        metrics["primary3_event_acc"] = event_acc
        metrics["primary3_event_total"] = float(event_total)
        row = {
            "dataset": dataset,
            "split": split,
            "seed": str(seed),
            "model": model,
            "n": str(len(run_rows)),
        }
        row.update({key: f"{value:.6f}" for key, value in sorted(metrics.items())})
        out.append(row)
    return out


def confusion_matrix(rows: Sequence[PredictionRow], *, labels: Sequence[str] = DEFAULT_LABELS) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for gt in labels:
        row = {"label": gt}
        for pred in labels:
            row[pred] = str(sum(1 for item in rows if item.label == gt and item.pred == pred))
        out.append(row)
    return out


def select_family_best(
    rows: Sequence[PredictionRow],
    *,
    family_models: Iterable[str],
    metric_name: str = "primary3_macro_f1",
) -> dict[tuple[str, str], str]:
    family_set = set(family_models)
    grouped: dict[tuple[str, str, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        if row.model in family_set:
            grouped[(row.dataset, row.split, row.model)].append(row)

    best: dict[tuple[str, str], tuple[str, float]] = {}
    for (dataset, split, model), run_rows in grouped.items():
        value = _metric_value(run_rows, metric_name)
        key = (dataset, split)
        if key not in best or value > best[key][1] or (value == best[key][1] and model < best[key][0]):
            best[key] = (model, value)
    return {key: model for key, (model, _) in best.items()}


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(str(key))
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def majority_vote(values: Sequence[str]) -> str:
    counts = Counter(values)
    max_count = max(counts.values())
    return sorted(key for key, count in counts.items() if count == max_count)[0]


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (float(pct) / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _metrics_for_labels(rows: Sequence[PredictionRow], *, labels: Sequence[str], prefix: str) -> dict[str, float]:
    label_set = set(labels)
    eval_rows = [row for row in rows if row.label in label_set]
    correct = sum(1 for row in eval_rows if row.label == row.pred)
    total = len(eval_rows)
    recalls: list[float] = []
    f1s: list[float] = []
    out: dict[str, float] = {f"{prefix}_accuracy": correct / total if total else 0.0}
    for label in labels:
        tp = sum(1 for row in eval_rows if row.label == label and row.pred == label)
        fp = sum(1 for row in eval_rows if row.label != label and row.pred == label)
        fn = sum(1 for row in eval_rows if row.label == label and row.pred != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[f"{prefix}_precision_{label}"] = precision
        out[f"{prefix}_recall_{label}"] = recall
        out[f"{prefix}_f1_{label}"] = f1
        out[f"{prefix}_support_{label}"] = float(support)
        if support:
            recalls.append(recall)
            f1s.append(f1)
    out[f"{prefix}_balanced_accuracy"] = statistics.fmean(recalls) if recalls else 0.0
    out[f"{prefix}_macro_f1"] = statistics.fmean(f1s) if f1s else 0.0
    return out


def _metric_value(rows: Sequence[PredictionRow], metric_name: str) -> float:
    if metric_name == "primary3_event_acc":
        return compute_event_accuracy(rows, labels=PRIMARY_LABELS)[0]
    metrics = compute_frame_metrics(rows)
    if metric_name not in metrics:
        raise KeyError(f"Unknown metric_name: {metric_name}")
    return metrics[metric_name]


def _safe_int(raw: object) -> int:
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return 0


def _safe_float(raw: object) -> float:
    try:
        return float(str(raw).strip())
    except Exception:
        return 0.0
