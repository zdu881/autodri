from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from autodri.aoi.equivalence import (
    DEFAULT_LABELS,
    PredictionRow,
    assign_internal_validation,
    compute_frame_metrics,
    load_split_manifest,
    validate_split_integrity,
    write_csv_rows,
)


TORCHVISION_MODELS = {"resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny"}
TIMM_MODELS = {"deit_tiny", "deit_tiny_patch16_224"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train non-YOLO AOI image classifiers with an internal validation split."
    )
    parser.add_argument("--data", required=True, help="Dataset directory with split_manifest.csv and image folders")
    parser.add_argument(
        "--model",
        choices=sorted(TORCHVISION_MODELS | TIMM_MODELS),
        default="resnet50",
        help="Backbone to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="artifacts/aoi_backbone_runs")
    parser.add_argument("--name", default="aoi_backbone")
    parser.add_argument("--internal-val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--use-physical-splits",
        action="store_true",
        help="Use manifest train/val/test directly, treating val as internal validation and test as frozen holdout.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated labels/classes to train, in output-index order.",
    )
    parser.add_argument("--export-onnx", action="store_true")
    return parser


def train(args: argparse.Namespace) -> Path:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    data_dir = Path(args.data)
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_split_manifest(data_dir / "split_manifest.csv")
    assignment = assign_training_splits(
        samples,
        val_ratio=args.internal_val_ratio,
        seed=args.seed,
        use_physical_splits=args.use_physical_splits,
    )
    integrity = validate_split_integrity(samples, assignment)
    if has_split_integrity_errors(integrity, use_physical_splits=args.use_physical_splits):
        raise ValueError(f"Split integrity failed: {integrity}")

    labels = resolve_training_labels(str(args.labels).split(","))
    train_rows = [sample for sample in samples if assignment[sample.dst_rel] == "train" and sample.label in labels]
    val_rows = [sample for sample in samples if assignment[sample.dst_rel] == "internal_val" and sample.label in labels]
    if not train_rows or not val_rows:
        raise ValueError("Internal train/validation split is empty; inspect split_manifest.csv")

    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    train_ds = AoiManifestDataset(data_dir, train_rows, class_to_idx, imgsz=args.imgsz, train=True)
    val_ds = AoiManifestDataset(data_dir, val_rows, class_to_idx, imgsz=args.imgsz, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=_collate_batch,
        drop_last=should_drop_last_batch(len(train_ds), args.batch),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=_collate_batch)

    model = build_model(args.model, num_classes=len(labels), pretrained=not args.no_pretrained).to(device)
    weight = None if args.no_class_weights else _class_weights(train_rows, class_to_idx, device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_metric = -1.0
    best_path = out_dir / "best.pt"
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = _evaluate(model, val_loader, device)
        scheduler.step()
        metric = metrics["primary3_macro_f1"]
        history_row = {"epoch": epoch, "train_loss": f"{train_loss:.6f}", **{k: f"{v:.6f}" for k, v in metrics.items()}}
        history_rows.append(history_row)
        if metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "labels": list(labels),
                    "imgsz": args.imgsz,
                    "class_to_idx": class_to_idx,
                    "args": vars(args),
                },
                best_path,
            )

    write_csv_rows(out_dir / "history.csv", history_rows)
    write_csv_rows(
        out_dir / "split_assignment.csv",
        [{"dst_rel": sample.dst_rel, "original_split": sample.split, "assigned_split": assignment[sample.dst_rel]} for sample in samples],
        fieldnames=["dst_rel", "original_split", "assigned_split"],
    )
    write_csv_rows(out_dir / "integrity_report.csv", [integrity])
    (out_dir / "labels.json").write_text(json.dumps(list(labels), indent=2), encoding="utf-8")
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    if args.export_onnx:
        export_onnx(best_path, out_dir / "best.onnx", device=device)
    print(f"Best checkpoint: {best_path}")
    return best_path


def build_model(model_name: str, *, num_classes: int, pretrained: bool = True):
    from torch import nn
    from torchvision import models

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name in TIMM_MODELS:
        try:
            import timm
        except ImportError as exc:
            raise SystemExit("timm is required for ViT/DeiT models. Install with: pip install timm") from exc
        return timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def export_onnx(checkpoint_path: Path, out_path: Path, *, device: str = "cpu") -> Path:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(str(checkpoint["model_name"]), num_classes=len(checkpoint["labels"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    imgsz = int(checkpoint.get("imgsz", 224))
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return out_path


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)


def should_drop_last_batch(dataset_size: int, batch_size: int) -> bool:
    if dataset_size <= batch_size or batch_size < 2:
        return False
    return dataset_size % batch_size == 1


def assign_training_splits(
    samples: Sequence[object],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    use_physical_splits: bool = False,
) -> dict[str, str]:
    if not use_physical_splits:
        return assign_internal_validation(samples, val_ratio=val_ratio, seed=seed)

    assignment: dict[str, str] = {}
    for sample in samples:
        split = str(sample.split)
        if split == "train":
            assigned = "train"
        elif split in {"val", "internal_val"}:
            assigned = "internal_val"
        elif split == "test":
            assigned = "test"
        else:
            assigned = "train"
        assignment[sample.dst_rel] = assigned
    return assignment


def resolve_training_labels(raw_labels: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(label.strip() for label in raw_labels if label.strip())
    if not labels:
        raise ValueError("At least one training label is required")
    unknown = [label for label in labels if label not in DEFAULT_LABELS]
    if unknown:
        raise ValueError(f"Unknown training labels: {unknown}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate training labels are not allowed: {labels}")
    return labels


def has_split_integrity_errors(integrity: dict[str, int], *, use_physical_splits: bool = False) -> bool:
    frozen_val_error = 0 if use_physical_splits else int(integrity.get("frozen_val_not_test_count", 0))
    return bool(
        int(integrity.get("group_leak_count", 0))
        or int(integrity.get("augmented_not_train_count", 0))
        or frozen_val_error
        or int(integrity.get("missing_assignment_count", 0))
    )


class AoiManifestDataset:
    def __init__(self, data_dir: Path, rows: Sequence[object], class_to_idx: dict[str, int], *, imgsz: int, train: bool):
        from torchvision import transforms

        self.data_dir = Path(data_dir)
        self.rows = list(rows)
        self.class_to_idx = class_to_idx
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((imgsz, imgsz)),
                    transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.25, 0.05)], p=0.8),
                    transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.85, 1.15)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    transforms.RandomErasing(p=0.15),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((imgsz, imgsz)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        from PIL import Image

        row = self.rows[idx]
        path = self.data_dir / row.dst_rel
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_idx[row.label], row


def _collate_batch(batch):
    import torch

    images, targets, samples = zip(*batch)
    return torch.stack(list(images), dim=0), torch.tensor(targets, dtype=torch.long), list(samples)


def _train_one_epoch(model, loader, criterion, optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(images.shape[0])
        total += int(images.shape[0])
    return total_loss / total if total else 0.0


def _evaluate(model, loader, device: str) -> dict[str, float]:
    import torch

    model.eval()
    pred_rows: list[PredictionRow] = []
    with torch.no_grad():
        for images, targets, samples in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu().tolist()
            for sample, pred_idx in zip(samples, preds):
                pred_rows.append(
                    PredictionRow(
                        dataset="internal",
                        split="internal_val",
                        model="candidate",
                        seed=0,
                        image_path=sample.dst_rel,
                        label=sample.label,
                        pred=DEFAULT_LABELS[int(pred_idx)],
                        domain=sample.domain,
                        video=sample.video,
                        timestamp=sample.timestamp,
                    )
                )
    return compute_frame_metrics(pred_rows)


def _class_weights(rows: Sequence[object], class_to_idx: dict[str, int], device: str):
    import torch

    counts = {label: 0 for label in class_to_idx}
    for row in rows:
        counts[row.label] += 1
    total = sum(counts.values())
    weights = []
    for label in class_to_idx:
        count = counts[label]
        weights.append((total / (len(class_to_idx) * count)) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


__all__ = [
    "assign_training_splits",
    "build_parser",
    "build_model",
    "export_onnx",
    "has_split_integrity_errors",
    "main",
    "resolve_training_labels",
    "should_drop_last_batch",
    "train",
]
