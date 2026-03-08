# driver_monitor

Driver-side monitoring scripts:

- `gaze_tracking.py`: MediaPipe eye/iris based gaze direction overlay
- `hand_on_wheel.py`: GroundingDINO based hand-on-wheel decision

## Install (VCS)

Install dependencies (including GroundingDINO from GitHub VCS, pinned commit):

```bash
python -m pip install -r driver_monitor/requirements.txt
```

If you need to refresh/rebuild the VCS package:

```bash
python -m pip install --upgrade --force-reinstall -r driver_monitor/requirements.txt
```

`driver_monitor/requirements.txt` currently pins:
- `groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git@856dde20aee659246248e20734ef9ba5214f5e44`

To switch version, edit that line to another commit/tag and reinstall.

Verify import path:

```bash
python - <<'PY'
import groundingdino
from pathlib import Path
print(Path(groundingdino.__file__).resolve())
PY
```

Model weights are still required separately. Recommended local path:
- `models/groundingdino_swint_ogc.pth`

## Recommended output directory

Use `driver_monitor/output/` for generated videos and ROI helper images.

## Run: gaze tracking

```bash
python driver_monitor/gaze_tracking.py \
  --video /path/to/input.mp4 \
  --output driver_monitor/output/gaze_output.mp4
```

## Run: hand on wheel

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --output driver_monitor/output/hand_on_wheel.mp4 \
  --weights models/groundingdino_swint_ogc.pth \
  --select-roi
```

Use YOLO backend (after training a small detector):

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --detector yolo \
  --yolo-model /path/to/best.pt \
  --yolo-hand-class-ids 0 \
  --yolo-wheel-class-ids 1 \
  --roi 950 300 1650 690 \
  --output driver_monitor/output/hand_on_wheel_yolo.mp4 \
  --state-csv driver_monitor/output/hand_on_wheel_yolo_states.csv
```

Optional:

```bash
# Fixed ROI and helper image directory
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --roi 950 300 1650 690 \
  --artifacts-dir driver_monitor/output
```

If you want to force a custom config path (normally auto-detected from installed package):

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --config /path/to/GroundingDINO_SwinT_OGC.py \
  --weights /path/to/groundingdino_swint_ogc.pth
```

Add temporal voting (e.g. 30s majority window) and export per-frame state:

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --output driver_monitor/output/hand_on_wheel_30s.mp4 \
  --roi 950 300 1650 690 \
  --iou-on-threshold 0.08 \
  --iou-off-threshold 0.03 \
  --decision-window-sec 30 \
  --state-csv driver_monitor/output/hand_on_wheel_30s_states.csv
```

Export sampled-frame detection boxes (for YOLO dataset building):

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --roi 950 300 1650 690 \
  --weights models/groundingdino_swint_ogc.pth \
  --no-video \
  --state-csv driver_monitor/output/hand_on_wheel_states.csv \
  --det-csv driver_monitor/output/hand_on_wheel_dets.csv
```

Analyze state stability metrics for poster reporting:

```bash
python driver_monitor/analyze_state_csv.py \
  --csv driver_monitor/output/hand_on_wheel_30s_states.csv \
  --sweep-windows 0,3,5,30 \
  --sweep-out-csv driver_monitor/output/window_sweep.csv
```

## Notes

- ROI helper/preview images are saved under `--artifacts-dir`.
- `--sample-fps` can reduce compute cost by reusing detections between frames.
- `--decision-window-sec` enables time-window majority vote. `0` keeps raw per-frame output.
- `--state-csv` writes raw/stable states for later poster metrics (flip rate, false alarms, delay).
- `--det-csv` exports sampled-frame boxes/classes/conf for YOLO acceleration workflow.
- `--iou-on-threshold` and `--iou-off-threshold` enable hysteresis to reduce ON/OFF flicker.
- `--uncertain-grace-sec` and `UNCERTAIN` state handle short missing detections.
- `--max-seconds` is useful for quick A/B tests on long videos.
- `analyze_state_csv.py` summarizes transitions/flip-rate reduction and can sweep multiple windows from one raw run.

## YOLO acceleration workflow (recommended)

Use GroundingDINO output as pseudo labels, then train a lightweight detector.

1) Generate detection CSVs (`--det-csv`) on selected videos/segments.

2) Convert det-csv files to YOLO dataset:

```bash
python driver_monitor/build_wheel_yolo_dataset.py \
  --det-csv "driver_monitor/output/*.dets.csv" \
  --out-dir driver_monitor/output/wheel_yolo_ds \
  --use-roi-crop \
  --include-negatives \
  --neg-keep-prob 0.2 \
  --val-ratio 0.2
```

3) Train a small model (example with Ultralytics CLI):

```bash
yolo detect train \
  data=driver_monitor/output/wheel_yolo_ds/data.yaml \
  model=yolo11n.pt \
  imgsz=640 \
  epochs=60 \
  batch=16 \
  device=0 \
  name=wheel_yolo11n_pseudo
```

4) Validate and compare with GroundingDINO baseline:
- speed (`fps` / wall-clock on same segments)
- state agreement (`ON/OFF/UNCERTAIN` consistency)
- transition stability after same `--decision-window-sec`

5) Deploy with `hand_on_wheel.py --detector yolo` using trained `best.pt`.
