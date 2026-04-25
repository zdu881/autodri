# Legacy Inventory

The following content is no longer part of the code-first repository surface and should live under `${AUTODRI_WORKSPACE}`:

- `data/`
- `models/`
- `runs/`
- `logs/`
- `GroundingDINO/`
- root-level videos, PDFs, XLSX inputs, and weight files
- generated `docs/*.csv` and `docs/*.xlsx`
- `gaze_onnx/experiments/label_backups/`
- `gaze_onnx/experiments/roi_refs/`
- participant-specific review packs such as `gaze_onnx/experiments/anno_*`
- dated one-off scripts such as `scripts/refresh_after_fill_*.sh`

Legacy repo-relative paths still work only when a script explicitly falls back to them and emits a deprecation warning.

