from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib
from pathlib import Path


def test_experiment_extra_includes_onnx_export_dependencies() -> None:
    data = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    deps = {
        dep.split(">", 1)[0].split("=", 1)[0].split("<", 1)[0].strip()
        for dep in data["project"]["optional-dependencies"]["experiments"]
    }

    assert "onnxruntime" in deps
    assert "onnx" in deps
    assert "timm" in deps
