# AOI Model Equivalence Experiment

This workflow tests whether YOLOv8-cls has an irreplaceable accuracy advantage for driver AOI classification, or whether its advantage is mainly engineering convenience.

## Claim

The primary statistical claim is evaluated only on `Forward`, `In-Car`, and `Non-Forward`. `Other` remains in training and secondary reports, but it is not part of the main equivalence decision because the current test sets have limited `Other` support.

The default non-inferiority/equivalence margin is `0.03` on `primary3_macro_f1`.

## Setup

Install the optional experiment dependencies:

```bash
pip install -e .[experiments]
```

Create the pre-registered run matrix, split assignments, integrity checks, command templates, and engineering rubric:

```bash
python -m autodri.cli.aoi_equivalence make-plan \
  --out-dir "$AUTODRI_WORKSPACE/artifacts/aoi_equivalence"
```

The default datasets are:

- `gaze_onnx/experiments/cls_dataset_two_domain_stratified_run1`
- `gaze_onnx/experiments/cls_dataset_two_domain_holdout_car1_genv3`
- `gaze_onnx/experiments/cls_dataset_two_domain_holdout_car2_genv3`

## Training

YOLOv8-cls runs use the existing Ultralytics workflow:

```bash
python -m autodri.cli.train_gaze_cls \
  --data gaze_onnx/experiments/cls_dataset_two_domain_holdout_car1_genv3 \
  --model yolov8s-cls.pt \
  --name holdout_car1_yolov8s_seed13 \
  --seed 13
```

Non-YOLO backbones use the internal validation split from the original `train` rows and leave the original `val` rows frozen for test-only evaluation:

```bash
python -m autodri.cli.train_aoi_backbone \
  --data gaze_onnx/experiments/cls_dataset_two_domain_holdout_car1_genv3 \
  --model convnext_tiny \
  --name holdout_car1_convnext_tiny_seed13 \
  --seed 13 \
  --export-onnx
```

Supported non-YOLO models are `resnet50`, `efficientnet_b0`, `efficientnet_b3`, `convnext_tiny`, and `deit_tiny`.

## Evaluation Inputs

Export frozen-test predictions from YOLO:

```bash
python -m autodri.cli.aoi_equivalence predict-yolo \
  --weights "$AUTODRI_WORKSPACE/artifacts/aoi_equivalence/runs_yolo/holdout_car1_yolov8s_seed13/weights/best.pt" \
  --data gaze_onnx/experiments/cls_dataset_two_domain_holdout_car1_genv3 \
  --dataset-name holdout_car1 \
  --model-name yolov8s-cls \
  --seed 13 \
  --out-csv predictions/holdout_car1_yolov8s_seed13.csv
```

Export frozen-test predictions from non-YOLO backbones:

```bash
python -m autodri.cli.aoi_equivalence predict-torch \
  --checkpoint "$AUTODRI_WORKSPACE/artifacts/aoi_equivalence/runs_torch/holdout_car1_convnext_tiny_seed13/best.pt" \
  --data gaze_onnx/experiments/cls_dataset_two_domain_holdout_car1_genv3 \
  --dataset-name holdout_car1 \
  --model-name convnext_tiny \
  --seed 13 \
  --out-csv predictions/holdout_car1_convnext_tiny_seed13.csv
```

Export predictions from the ONNX version of the same non-YOLO model:

```bash
python -m autodri.cli.aoi_equivalence predict-onnx \
  --onnx "$AUTODRI_WORKSPACE/artifacts/aoi_equivalence/runs_torch/holdout_car1_convnext_tiny_seed13/best.onnx" \
  --labels-json "$AUTODRI_WORKSPACE/artifacts/aoi_equivalence/runs_torch/holdout_car1_convnext_tiny_seed13/labels.json" \
  --data gaze_onnx/experiments/cls_dataset_two_domain_holdout_car1_genv3 \
  --dataset-name holdout_car1 \
  --model-name convnext_tiny_onnx \
  --seed 13 \
  --out-csv predictions/holdout_car1_convnext_tiny_seed13_onnx.csv
```

Check PyTorch-ONNX top1 parity:

```bash
python -m autodri.cli.aoi_equivalence parity \
  --reference-predictions predictions/holdout_car1_convnext_tiny_seed13.csv \
  --candidate-predictions predictions/holdout_car1_convnext_tiny_seed13_onnx.csv \
  --out-csv "$AUTODRI_WORKSPACE/artifacts/aoi_equivalence/bench/convnext_tiny_parity.csv"
```

The statistics commands expect one or more prediction CSV files with these columns:

```text
dataset,split,model,seed,image_path,label,pred,domain,video,timestamp
```

All models for a given dataset/split/seed must use the same `image_path` list so paired bootstrap comparisons are valid.

## Metrics And Equivalence

Compute metrics and confusion matrices:

```bash
python -m autodri.cli.aoi_equivalence metrics \
  --predictions predictions/*.csv \
  --out-dir "$AUTODRI_WORKSPACE/artifacts/aoi_equivalence/eval"
```

Compare every non-YOLO model against the YOLO family-best model for each dataset/split/seed:

```bash
python -m autodri.cli.aoi_equivalence equivalence \
  --predictions predictions/*.csv \
  --out-dir "$AUTODRI_WORKSPACE/artifacts/aoi_equivalence/stats" \
  --metric primary3_macro_f1 \
  --delta-margin 0.03 \
  --n-boot 1000
```

The generated `conclusion.md` reports whether at least two thirds of core splits have a non-YOLO candidate whose lower confidence bound is above `-0.03`.
The `equivalence_results.csv` table also includes McNemar discordant counts, raw McNemar `p` values, and Holm-adjusted `p` values as auxiliary significance checks.

## ONNX Engineering Benchmark

Benchmark exported ONNX models with the same input shape and batch sizes:

```bash
python -m autodri.cli.aoi_equivalence benchmark-onnx \
  --onnx "$AUTODRI_WORKSPACE/artifacts/aoi_equivalence/runs_torch/holdout_car1_convnext_tiny_seed13/best.onnx" \
  --model-name convnext_tiny \
  --out-csv "$AUTODRI_WORKSPACE/artifacts/aoi_equivalence/bench/convnext_tiny.csv" \
  --batch-sizes 1 32
```

Report `p50/p95` latency, throughput, model size, ONNX export success, and PyTorch-ONNX top1 parity separately from accuracy. The intended interpretation is that YOLOv8-cls wins engineering convenience only if its deployment metrics are better while accuracy confidence intervals overlap or non-YOLO candidates are non-inferior.
