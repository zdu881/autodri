#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train YOLOv8-cls for driver gaze classification.

Usage:
  conda run -n adri python gaze_onnx/experiments/train_gaze_cls.py \
    --data gaze_onnx/experiments/cls_dataset \
    --epochs 50 \
    --imgsz 224 \
    --batch 32 \
    --model yolov8n-cls.pt

After training, export to ONNX:
  conda run -n adri python gaze_onnx/experiments/train_gaze_cls.py \
    --data gaze_onnx/experiments/cls_dataset \
    --mode export \
    --weights runs/classify/train/weights/best.pt
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Dataset dir (with train/val subdirs)")
    p.add_argument("--mode", choices=["train", "eval", "export"], default="train")
    p.add_argument("--model", default="yolov8n-cls.pt", help="Base model for training")
    p.add_argument("--weights", default=None, help="Trained weights for eval/export")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", default="0", help="CUDA device, e.g. 0 or cpu")
    p.add_argument("--project", default="gaze_onnx/experiments/runs_cls")
    p.add_argument("--name", default="gaze_v1")
    p.add_argument(
        "--aug-preset",
        choices=["baseline", "robust"],
        default="robust",
        help="Augmentation preset. robust is recommended for cross-domain generalization.",
    )
    p.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout (0~1)")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def train(args):
    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        workers=args.workers,
        dropout=max(0.0, min(0.9, args.dropout)),
        cos_lr=True,
    )

    if args.aug_preset == "robust":
        # Stronger photometric/geometry augmentations for lighting + domain robustness.
        train_kwargs.update(
            dict(
                hsv_h=0.03,
                hsv_s=0.70,
                hsv_v=0.70,
                degrees=20.0,
                translate=0.15,
                scale=0.40,
                flipud=0.0,
                fliplr=0.50,
                erasing=0.35,
                auto_augment="randaugment",
                mixup=0.20,
                cutmix=0.20,
            )
        )
    else:
        train_kwargs.update(
            dict(
                hsv_h=0.015,
                hsv_s=0.40,
                hsv_v=0.40,
                degrees=15.0,
                translate=0.10,
                scale=0.30,
                flipud=0.0,
                fliplr=0.50,
                erasing=0.10,
                mixup=0.0,
                cutmix=0.0,
            )
        )

    results = model.train(
        **train_kwargs
    )
    print("\n=== Training complete ===")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    return results


def evaluate(args):
    from ultralytics import YOLO

    if not args.weights:
        raise SystemExit("--weights required for eval mode")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="val",
    )
    print("\n=== Evaluation ===")
    print(f"Top-1 accuracy: {metrics.top1:.4f}")
    print(f"Top-5 accuracy: {metrics.top5:.4f}")
    return metrics


def export_onnx(args):
    from ultralytics import YOLO

    if not args.weights:
        raise SystemExit("--weights required for export mode")

    model = YOLO(args.weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=False,
        simplify=True,
    )
    print(f"\nExported ONNX: {onnx_path}")
    return onnx_path


def main():
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    elif args.mode == "export":
        export_onnx(args)


if __name__ == "__main__":
    main()
