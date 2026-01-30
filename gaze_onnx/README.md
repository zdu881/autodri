# Gaze State Classification (ONNX)

This folder contains a standalone script to classify driver gaze into 3 states using ONNX models:
- Face detection: SCRFD (`scrfd_person_2.5g.onnx`)
- Gaze estimation: L2CS-Net (`L2CSNet_gaze360.onnx`)

## Install

Python 3.x

```bash
pip install opencv-python onnxruntime numpy
```

## Run

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

## Model download

- SCRFD (downloadable): use the InsightFace release asset `scrfd_person_2.5g.onnx`.
- L2CS ONNX: many sources provide PyTorch `.pkl` weights (Google Drive) or ONNX via Baidu Pan; depending on your network, you may need to download it manually and place it under `../models/`.
