#!/usr/bin/env python3
"""Detailed per-class evaluation of trained YOLOv8-cls model."""

from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

WEIGHTS = "runs/classify/gaze_onnx/experiments/runs_cls/gaze_v1/weights/best.pt"
VAL_DIR = "gaze_onnx/experiments/cls_dataset/val"


def main():
    model = YOLO(WEIGHTS)
    val_dir = Path(VAL_DIR)
    classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    print("Classes:", classes)
    print("Model names:", model.names)

    # Per-class evaluation
    cm = defaultdict(lambda: defaultdict(int))
    total = 0
    correct = 0
    errors = []

    for cls_name in classes:
        cls_dir = val_dir / cls_name
        imgs = sorted(list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")))
        for img_path in imgs:
            results = model(str(img_path), verbose=False)
            pred_idx = results[0].probs.top1
            pred_name = results[0].names[pred_idx]
            conf = results[0].probs.top1conf.item()
            cm[cls_name][pred_name] += 1
            total += 1
            if pred_name == cls_name:
                correct += 1
            else:
                errors.append((cls_name, pred_name, conf, img_path.name))

    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.1%}")

    print(f"\n=== Confusion Matrix (row=GT, col=Predicted) ===")
    header = f"{'':>14s}"
    for c in classes:
        header += f"{c:>14s}"
    print(header)
    for gt in classes:
        row = f"{gt:>14s}"
        row_total = sum(cm[gt].values())
        for p in classes:
            row += f"{cm[gt][p]:>14d}"
        acc = cm[gt][gt] / row_total if row_total > 0 else 0
        row += f"  | recall={acc:.1%} (n={row_total})"
        print(row)

    print(f"\n=== Per-class Precision ===")
    for c in classes:
        col_total = sum(cm[gt][c] for gt in classes)
        if col_total > 0:
            prec = cm[c][c] / col_total
            print(f"  {c}: {prec:.1%} ({cm[c][c]}/{col_total})")
        else:
            print(f"  {c}: N/A (no predictions)")

    if errors:
        print(f"\n=== Misclassified samples ({len(errors)}) ===")
        for gt, pred, conf, fname in errors:
            print(f"  {fname}: GT={gt}, Pred={pred} (conf={conf:.3f})")


if __name__ == "__main__":
    main()
