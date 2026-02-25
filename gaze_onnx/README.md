# Gaze State Classification (ONNX)

This folder now contains two inference pipelines:

- `gaze_state_onnx.py`: SCRFD + L2CS (angle + rule based)
- `gaze_state_cls.py`: SCRFD + YOLOv8-cls (direct class prediction, recommended)

`gaze_state_cls.py` supports 4 final states:
- `Forward`
- `In-Car`
- `Non-Forward`
- `Other` (driver absent / not confidently present)

## Install

Python 3.x

```bash
pip install opencv-python onnxruntime numpy
```

## Run (rule-based baseline)

```bash
python gaze_state_onnx.py \
  --video "../6月1日.mp4" \
  --scrfd "../models/scrfd_person_2.5g.onnx" \
  --l2cs "../models/L2CSNet_gaze360.onnx" \
  --roi 950 300 1650 690 \
  --out-video output_gaze.mp4 \
  --csv output_gaze.csv

# If you want a faster run without drawing annotations onto the output video:
#   add: --no-overlay
```

Notes:
- The script supports two common L2CS ONNX output formats:
  - Direct regression outputs (pitch/yaw in degrees)
  - Classification logits over bins (computes expected angle)
- If no face is detected, it logs `No Face` for that frame.

## Run (recommended 4-class pipeline)

```bash
python gaze_state_cls.py \
  --video "../6月1日.mp4" \
  --scrfd "../models/scrfd_person_2.5g.onnx" \
  --cls-model "../models/gaze_cls_yolov8n.onnx" \
  --roi 950 300 1650 690 \
  --out-video output_gaze_cls.mp4 \
  --csv output_gaze_cls.csv \
  --presence-min-face-score 0.45 \
  --presence-min-face-ratio 0.00 \
  --other-enter-frames 8 \
  --other-exit-frames 3
```

Useful options:
- `--class-bias BIAS_F BIAS_IC BIAS_NF`
- `--cls-threshold 0.45`
- `--write-when-other skip|other`

CSV now includes both `Base_Class` (3-way gaze class) and final `Gaze_Class` (4-way with `Other`).

## Tune Post-process Without Retraining

If you already have manual labels, tune inference parameters directly:

```bash
python experiments/tune_cls_postprocess.py \
  --pred-csv output/output_gaze_cls_full.csv \
  --labels experiments/samples_smooth4_full_500/labels.csv \
  --out-json output/tune_cls_postprocess.best.json
```

Then apply tuned params in `gaze_state_cls.py` (`--class-bias`, `--cls-threshold`,
`--presence-min-face-score`, `--presence-min-face-ratio`).

## Model download

- SCRFD (downloadable): use the InsightFace release asset `scrfd_person_2.5g.onnx`.
- L2CS ONNX: many sources provide PyTorch `.pkl` weights (Google Drive) or ONNX via Baidu Pan; depending on your network, you may need to download it manually and place it under `../models/`.
