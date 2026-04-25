#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <participant> <wheel_gpu>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="${AUTODRI_WORKSPACE:-${REPO_ROOT}_workspace}"

PARTICIPANT="$1"
WHEEL_GPU="$2"

BASE_PY="${BASE_PY:-python}"
ADRI_PY="${ADRI_PY:-$BASE_PY}"

PLAN_CSV="${WORKSPACE}/data/natural_driving/${PARTICIPANT}/analysis/${PARTICIPANT}_infer_plan.current.csv"
WINDOWS_CSV="${WORKSPACE}/data/natural_driving/${PARTICIPANT}/analysis/${PARTICIPANT}_windows.20s.current.csv"
GAZE_MAP_CSV="${WORKSPACE}/data/natural_driving/${PARTICIPANT}/analysis/${PARTICIPANT}_gaze_map.current.csv"
WHEEL_MAP_CSV="${WORKSPACE}/data/natural_driving/${PARTICIPANT}/analysis/${PARTICIPANT}_wheel_map.current.csv"
METRICS_CSV="${WORKSPACE}/data/natural_driving/${PARTICIPANT}/analysis/${PARTICIPANT}_window_metrics.20s.current.csv"
GAZE_MODEL="${WORKSPACE}/models/gaze_cls_${PARTICIPANT}_200shot_driveonly_ft_v1.onnx"
WHEEL_WEIGHTS="${WORKSPACE}/models/groundingdino_swint_ogc.pth"

if [ ! -f "$PLAN_CSV" ]; then
  echo "missing plan csv: $PLAN_CSV" >&2
  exit 3
fi

if [ ! -f "$GAZE_MODEL" ]; then
  echo "missing participant model: $GAZE_MODEL" >&2
  exit 4
fi

echo "[pipeline] participant=${PARTICIPANT} wheel_gpu=${WHEEL_GPU}"
echo "[pipeline] repo_root=${REPO_ROOT}"
echo "[pipeline] workspace=${WORKSPACE}"
echo "[pipeline] plan=${PLAN_CSV}"
echo "[pipeline] model=${GAZE_MODEL}"

cd "${REPO_ROOT}"
export AUTODRI_WORKSPACE="${WORKSPACE}"
export CUDA_VISIBLE_DEVICES="${WHEEL_GPU}"
echo "[pipeline] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

"$BASE_PY" -m autodri.cli.run_p1_infer_plan \
  --plan-csv "$PLAN_CSV" \
  --python-bin "$ADRI_PY" \
  --run-gaze \
  --run-wheel \
  --gaze-cls-model "$GAZE_MODEL" \
  --wheel-weights "$WHEEL_WEIGHTS" \
  --wheel-device cuda \
  --no-video \
  --skip-existing

"$BASE_PY" -m autodri.cli.compute_p1_window_metrics \
  --windows-csv "$WINDOWS_CSV" \
  --gaze-map-csv "$GAZE_MAP_CSV" \
  --wheel-map-csv "$WHEEL_MAP_CSV" \
  --out-csv "$METRICS_CSV"

echo "[pipeline] done participant=${PARTICIPANT}"
