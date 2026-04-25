# autodri

`autodri` is a code-first repository for driver monitoring and gaze-analysis workflows built around naturalistic driving video.

The repository currently focuses on:
- driver gaze inference from in-cabin video
- hand-on-wheel inference
- ROI assignment and manual review tooling
- few-shot dataset preparation and model adaptation
- window-level metric aggregation and QC workflows

It is structured as an installable Python package with an external workspace for data, models, and generated artifacts.

## Status

This repository is usable as a public codebase, but it is still an actively cleaned-up research/engineering project rather than a polished end-user product.

What is stable:
- canonical Python entrypoints under `python -m autodri.cli.<name>`
- workspace-aware path resolution via `${AUTODRI_WORKSPACE}`
- core gaze / wheel inference workflows
- window metrics, coverage QC, and summary generation

What is not bundled:
- raw videos
- trained model weights
- large review packs and experiment outputs
- private study spreadsheets or local working archives

## Core Ideas

- Keep the Git repository small and code-oriented.
- Keep data, models, reports, and local review artifacts outside the repository.
- Preserve legacy script paths as thin wrappers so older commands still work.
- Make every main workflow runnable either as a package CLI or as a compatibility script.

## Repository Layout

```text
autodri/
  src/autodri/              Canonical package implementation
  gaze_onnx/                Legacy-compatible wrappers and related files
  driver_monitor/           Legacy-compatible wrappers and wheel utilities
  scripts/                  Small orchestration helpers
  docs/                     Human-written docs and diagrams
  tests/                    Smoke and unit tests
```

The package is split into:
- `autodri.common`: workspace and shared helpers
- `autodri.gaze`: gaze runtime code
- `autodri.wheel`: hand-on-wheel runtime code
- `autodri.workflows`: data prep, review, training, and metric workflows
- `autodri.cli`: canonical command entrypoints

## Workspace Model

Non-code resources should live under `${AUTODRI_WORKSPACE}`.

If `${AUTODRI_WORKSPACE}` is unset, the default workspace is a sibling directory named `autodri_workspace/`.

Expected layout:

```text
autodri/
autodri_workspace/
  data/
  models/
  artifacts/
  archive/
  sources/
```

Directory meanings:
- `data/`: local videos and analysis CSVs
- `models/`: ONNX weights, detector weights, task models
- `artifacts/`: generated reports, review images, merged tables, transient outputs
- `archive/`: local-only historical payloads and non-reproducible backups
- `sources/`: original spreadsheets, third-party sources, and large imported assets

## Installation

Python `>=3.10` is required.

Minimal editable install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

Some workflows also depend on packages not declared in `pyproject.toml`, especially video / model tooling used by the legacy scripts. Check:
- `driver_monitor/requirements.txt`
- imports inside the workflow you actually intend to run

## Quick Start

Set the workspace:

```bash
export AUTODRI_WORKSPACE=/path/to/autodri_workspace
```

Inspect supported commands:

```bash
python -m autodri.cli.gaze_state_cls --help
python -m autodri.cli.hand_on_wheel --help
python -m autodri.cli.compute_p1_window_metrics --help
```

Typical examples:

```bash
python -m autodri.cli.gaze_state_cls \
  --video "$AUTODRI_WORKSPACE/data/example.mp4" \
  --scrfd "$AUTODRI_WORKSPACE/models/scrfd_person_2.5g.onnx" \
  --cls-model "$AUTODRI_WORKSPACE/models/gaze_cls_yolov8n.onnx" \
  --csv "$AUTODRI_WORKSPACE/artifacts/example.gaze.csv" \
  --no-video
```

```bash
python -m autodri.cli.hand_on_wheel \
  --video "$AUTODRI_WORKSPACE/data/example.mp4" \
  --weights "$AUTODRI_WORKSPACE/models/groundingdino_swint_ogc.pth" \
  --state-csv "$AUTODRI_WORKSPACE/artifacts/example.wheel.csv" \
  --no-video
```

```bash
python -m autodri.cli.build_all_participants_window_metrics \
  --gaze-coverage-threshold 0.98
```

See [docs/supported_workflows.md](docs/supported_workflows.md) for the maintained command surface.

## Supported Canonical Entry Points

Main commands currently intended for regular use:

- `python -m autodri.cli.gaze_state_cls`
- `python -m autodri.cli.hand_on_wheel`
- `python -m autodri.cli.web_label_tool`
- `python -m autodri.cli.build_participant_video_manifest_from_xlsx`
- `python -m autodri.cli.assign_dual_roi`
- `python -m autodri.cli.build_domains_csv_from_dual_assignment`
- `python -m autodri.cli.create_multidomain_annotation_pack`
- `python -m autodri.cli.build_fewshot_pack`
- `python -m autodri.cli.prepare_cls_dataset_from_pack`
- `python -m autodri.cli.train_gaze_cls`
- `python -m autodri.cli.run_p1_infer_plan`
- `python -m autodri.cli.run_domains_gaze_infer`
- `python -m autodri.cli.build_p1_schedule_windows`
- `python -m autodri.cli.compute_p1_window_metrics`
- `python -m autodri.cli.build_all_participants_window_metrics`
- `python -m autodri.cli.build_participants_results_summary`
- `python -m autodri.cli.export_gaze_qc_review_images`

## Legacy Compatibility

Legacy script paths are still present as thin wrappers. For example:

- `gaze_onnx/gaze_state_cls.py`
- `driver_monitor/hand_on_wheel.py`
- `gaze_onnx/experiments/compute_p1_window_metrics.py`
- `gaze_onnx/experiments/run_p1_infer_plan.py`

They forward to the package implementation and are retained to avoid breaking existing local workflows.

## Development

Run tests:

```bash
pytest
```

Useful targeted checks:

```bash
python -m autodri.cli.build_participants_results_summary --help
python -m autodri.cli.build_all_participants_window_metrics --help
python -m autodri.cli.export_gaze_qc_review_images --help
```

## Documentation

Useful docs in this repository:
- [docs/supported_workflows.md](docs/supported_workflows.md)
- [docs/legacy_inventory.md](docs/legacy_inventory.md)
- [docs/annotation_quickstart.md](docs/annotation_quickstart.md)
- [docs/annotation_workflow.md](docs/annotation_workflow.md)
- [docs/onnxruntime_cuda_fix.md](docs/onnxruntime_cuda_fix.md)
- [docs/perclos_blink_metrics.md](docs/perclos_blink_metrics.md)

## Limitations

- The repository does not ship sample models or sample videos.
- Several workflows are still research-oriented and assume domain-specific CSV schemas.
- Some legacy scripts remain participant- or study-specific even if they are now versioned.
- Public release quality is much higher than before, but not every workflow has a complete fixture dataset.

## What Stays Out Of Git

The repository intentionally excludes:
- raw study videos
- local model checkpoints and large weight files
- generated review images and QC exports
- local archive payloads and manual backup packs

See [docs/legacy_inventory.md](docs/legacy_inventory.md) for the current boundary.
