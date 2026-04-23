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
