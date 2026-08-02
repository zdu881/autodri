# Autodri

[![tests](https://github.com/zdu881/autodri/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/zdu881/autodri/actions/workflows/tests.yml)

Autodri is an auditable Python toolkit for deriving driver gaze and hand-on-wheel variables from naturalistic in-cabin video. It combines human-reviewed regions of interest, frame-level inference, temporal stabilization, experiment orchestration, and window-level quality control.

## What It Provides

- AOI gaze classification and temporal stabilization
- hand-on-wheel teacher inference and lightweight state distillation
- participant-level leave-one-participant-out and few-shot experiments
- dual-ROI assignment and human review tools
- window-level behavior metrics and pipeline validation reports
- reproducible AutoUI manuscript figures and supporting artifacts

The repository contains code, tests, documentation, and the accompanying method paper. Raw study video, model weights, review images, and generated experiment outputs remain outside Git.

## Repository Layout

```text
src/autodri/       Maintained Python package
tests/             Unit and workflow-contract tests
scripts/           Reproducible utility and figure scripts
docs/              User guides, design notes, and project artifacts
paper/             AutoUI manuscript, bibliography, PDF, and figures
gaze_onnx/         Legacy-compatible gaze entrypoints
driver_monitor/    Legacy-compatible driver-monitoring entrypoints
```

New integrations should use `python -m autodri.cli.<command>`. The legacy directories are retained only to avoid breaking established local workflows.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiments]"
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Some detector workflows also require GroundingDINO and model checkpoints that are intentionally not distributed in this repository.

## Workspace

Keep data and generated artifacts in an external workspace:

```bash
export AUTODRI_WORKSPACE=/path/to/autodri_workspace
```

If the variable is unset, Autodri uses a sibling directory named `autodri_workspace/`.

```text
autodri_workspace/
  data/          Local videos and analysis tables
  models/        Model checkpoints and exported networks
  artifacts/     Reports, review packs, and experiment outputs
  archive/       Local historical material
  sources/       Original spreadsheets and imported sources
```

## Main Commands

| Purpose | Command |
| --- | --- |
| Gaze inference | `python -m autodri.cli.gaze_state_cls` |
| Hand-on-wheel inference | `python -m autodri.cli.hand_on_wheel` |
| ROI assignment | `python -m autodri.cli.assign_dual_roi` |
| Browser labeling | `python -m autodri.cli.web_label_tool` |
| Gaze model training | `python -m autodri.cli.train_gaze_cls` |
| AOI backbone training | `python -m autodri.cli.train_aoi_backbone` |
| AutoUI experiment matrix | `python -m autodri.cli.autoui_experiments` |
| AOI equivalence analysis | `python -m autodri.cli.aoi_equivalence` |
| Window metrics | `python -m autodri.cli.compute_p1_window_metrics` |
| Pipeline validation | `python -m autodri.cli.pipeline_validation_summary` |

Inspect any interface with `--help`, for example:

```bash
python -m autodri.cli.autoui_experiments --help
python -m autodri.cli.pipeline_validation_summary --help
```

The complete maintained command surface is documented in [docs/supported_workflows.md](docs/supported_workflows.md).

## Paper and Project Material

- [AutoUI manuscript source](paper/autoui.tex)
- [Compiled AutoUI manuscript](paper/autoui.pdf)
- [Paper build instructions](paper/README.md)
- [Undergraduate project summary](docs/project_summarization.md)
- [Project poster](docs/assets/autodri_undergrad_poster.png)
- [Figure redesign brief](docs/autoui_figure_redraw_brief.md)

## Documentation

- [Supported workflows](docs/supported_workflows.md)
- [Annotation quick start](docs/annotation_quickstart.md)
- [Annotation workflow](docs/annotation_workflow.md)
- [Generalization datasets](docs/generalization_datasets.md)
- [AOI equivalence experiment](docs/aoi_equivalence_experiment.md)
- [Wheel-state distillation](docs/wheel_state_distillation_20260603.md)
- [ONNX Runtime CUDA troubleshooting](docs/onnxruntime_cuda_fix.md)
- [Legacy inventory](docs/legacy_inventory.md)

## Reproducibility Boundary

The public repository excludes:

- raw or derived participant video
- model weights (`*.pt`) and locally exported model artifacts
- reviewer images and manual review packs
- private study spreadsheets and participant-specific manifests
- LaTeX intermediate files and runtime caches

Commands that require those resources accept explicit paths or resolve them from `${AUTODRI_WORKSPACE}`. The committed tests exercise code and data-contract behavior without distributing study data.

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff). The method paper in `paper/` should be used when citing the complete annotation workflow.
