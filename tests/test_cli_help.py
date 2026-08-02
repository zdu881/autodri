from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("autodri.cli.compute_p1_window_metrics", REPO_ROOT / "gaze_onnx/experiments/compute_p1_window_metrics.py"),
    ("autodri.cli.run_p1_infer_plan", REPO_ROOT / "gaze_onnx/experiments/run_p1_infer_plan.py"),
    ("autodri.cli.build_participants_results_summary", REPO_ROOT / "gaze_onnx/experiments/build_participants_results_summary.py"),
]

CLI_ONLY_CASES = [
    "autodri.cli.aoi_equivalence",
    "autodri.cli.autoui_experiments",
    "autodri.cli.pipeline_validation_summary",
    "autodri.cli.train_aoi_backbone",
    "autodri.cli.train_gaze_cls",
]


@pytest.mark.parametrize(("module_name", "legacy_path"), CASES)
def test_cli_help(module_name: str, legacy_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert proc.returncode == 0, proc.stderr
    legacy = subprocess.run(
        [sys.executable, str(legacy_path), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert legacy.returncode == 0, legacy.stderr


@pytest.mark.parametrize("module_name", CLI_ONLY_CASES)
def test_cli_only_help(module_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert proc.returncode == 0, proc.stderr


def test_autoui_help_hides_private_worker_commands() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "autodri.cli.autoui_experiments", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert proc.returncode == 0, proc.stderr
    assert "_deployment-one" not in proc.stdout
