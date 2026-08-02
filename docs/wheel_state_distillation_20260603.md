# Wheel State Distillation Check (2026-06-03)

This report records the GroundingDINO-to-student state distillation check used
for `autoui.tex`.

## Source Teacher Outputs

- Teacher state CSVs:
  - `${AUTODRI_WORKSPACE}/data/natural_driving_p1/analysis/tmp_wheel_state_gd_60s.csv`
  - `${AUTODRI_WORKSPACE}/data/natural_driving_p1/analysis/tmp_wheel_state_gd_140s.csv`
- Teacher detection metadata CSVs:
  - `${AUTODRI_WORKSPACE}/data/natural_driving_p1/analysis/tmp_wheel_det_gd_60s.csv`
  - `${AUTODRI_WORKSPACE}/data/natural_driving_p1/analysis/tmp_wheel_det_gd_140s.csv`
- Teacher probe wall time from the previous runtime check:
  - 60 s probe: 53.698 s
  - 140 s probe: 119.054 s
  - combined: 172.752 s

## Dataset

Command:

```bash
export AUTODRI_WORKSPACE=/path/to/autodri_workspace
python scripts/wheel_state_distill_experiment.py build \
  --state-csv "${AUTODRI_WORKSPACE}/data/natural_driving_p1/analysis/tmp_wheel_state_gd_60s.csv" \
  --det-csv "${AUTODRI_WORKSPACE}/data/natural_driving_p1/analysis/tmp_wheel_det_gd_60s.csv" \
  --state-csv "${AUTODRI_WORKSPACE}/data/natural_driving_p1/analysis/tmp_wheel_state_gd_140s.csv" \
  --det-csv "${AUTODRI_WORKSPACE}/data/natural_driving_p1/analysis/tmp_wheel_det_gd_140s.csv" \
  --out-dir "${AUTODRI_WORKSPACE}/artifacts/wheel_state_distill_20260603/dataset_224_hash" \
  --workspace-root "${AUTODRI_WORKSPACE}" \
  --imgsz 224 \
  --sample-stride 1 \
  --val-ratio 0.2 \
  --split-mode hash \
  --seed 3407
```

Output:

- 5,000 ROI crops at 224 px.
- Train: 3,965 images (`OFF` 2,025, `ON` 58, `UNCERTAIN` 1,882).
- Validation: 1,035 images (`OFF` 540, `ON` 12, `UNCERTAIN` 483).
- Manifest/image consistency check: 5,000 manifest rows and 5,000 existing images.

## Student Training

Command:

```bash
yolo classify train \
  model=yolov8n-cls.pt \
  data="${AUTODRI_WORKSPACE}/artifacts/wheel_state_distill_20260603/dataset_224_hash" \
  epochs=30 imgsz=224 batch=64 device=6 workers=8 \
  project="${AUTODRI_WORKSPACE}/artifacts/wheel_state_distill_20260603/yolo_runs" \
  name=yolov8n_cls_state_hash_e30_codex_verify exist_ok=True seed=3407 deterministic=True \
  patience=10 auto_augment=none erasing=0.0 fliplr=0.0 flipud=0.0 \
  hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 translate=0.0 scale=0.0
```

Output:

- Student: YOLOv8n-cls, 1.44M parameters.
- Best checkpoint: epoch 15, `top1_acc=0.988` by Ultralytics validation.
- Saved checkpoint: `${AUTODRI_WORKSPACE}/artifacts/wheel_state_distill_20260603/yolo_runs/yolov8n_cls_state_hash_e30_codex_verify/weights/best.pt`
- Checkpoint size: 2.9 MB.

## Independent Prediction Check

Command:

```bash
python scripts/wheel_state_distill_experiment.py predict \
  --model "${AUTODRI_WORKSPACE}/artifacts/wheel_state_distill_20260603/yolo_runs/yolov8n_cls_state_hash_e30_codex_verify/weights/best.pt" \
  --manifest "${AUTODRI_WORKSPACE}/artifacts/wheel_state_distill_20260603/dataset_224_hash/manifest.csv" \
  --out-csv "${AUTODRI_WORKSPACE}/artifacts/wheel_state_distill_20260603/yolov8n_cls_state_hash_codex_verify_val_predictions.csv" \
  --split val \
  --imgsz 224 \
  --batch 128 \
  --device 6
```

Output:

- Validation agreement: 1,022 / 1,035 = 0.9874396135.
- Confusion:
  - `OFF->OFF`: 531
  - `OFF->ON`: 2
  - `OFF->UNCERTAIN`: 7
  - `ON->ON`: 12
  - `UNCERTAIN->OFF`: 4
  - `UNCERTAIN->UNCERTAIN`: 479
- Timed command wall time, including model load: 3.60 s for the 1,035 validation crops.

## Loaded-Model Throughput

After loading the model and warming up with 128 crops, prediction on all 5,000
ROI crops took 6.341544 s:

- 788.45 images/s
- 1.2683 ms/image
- 27.24x faster than the combined 172.752 s GroundingDINO teacher probe time

This check demonstrates that the slow GroundingDINO branch can be amortized into
a compact state student for the sampled p1 probes. It is not a cross-participant
production replacement test.
