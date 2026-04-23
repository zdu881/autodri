from __future__ import annotations

from pathlib import Path

import pytest

from autodri.common import paths


def test_workspace_root_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    monkeypatch.setenv(paths.WORKSPACE_ENV_VAR, str(workspace))
    assert paths.workspace_root() == workspace


def test_resolve_existing_prefers_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    (workspace / "models").mkdir(parents=True)
    (repo / "models").mkdir(parents=True)
    (workspace / "models" / "demo.onnx").write_text("ws", encoding="utf-8")
    (repo / "models" / "demo.onnx").write_text("repo", encoding="utf-8")
    monkeypatch.setenv(paths.WORKSPACE_ENV_VAR, str(workspace))
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    resolved = paths.resolve_existing_path(
        "",
        workspace_rel="models/demo.onnx",
        legacy_rels=("models/demo.onnx",),
        description="demo model",
    )
    assert resolved == workspace / "models" / "demo.onnx"


def test_resolve_existing_warns_on_legacy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "demo.onnx").write_text("repo", encoding="utf-8")
    monkeypatch.setenv(paths.WORKSPACE_ENV_VAR, str(workspace))
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    with pytest.deprecated_call():
        resolved = paths.resolve_existing_path(
            "",
            workspace_rel="models/demo.onnx",
            legacy_rels=("models/demo.onnx",),
            description="demo model",
        )
    assert resolved == repo / "models" / "demo.onnx"


def test_resolve_existing_uses_cli_value(tmp_path: Path) -> None:
    direct = tmp_path / "direct.csv"
    direct.write_text("ok\n", encoding="utf-8")
    resolved = paths.resolve_existing_path(str(direct), description="direct file")
    assert resolved == direct


def test_resolve_existing_raises_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    monkeypatch.setenv(paths.WORKSPACE_ENV_VAR, str(workspace))
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    with pytest.raises(FileNotFoundError):
        paths.resolve_existing_path(
            "",
            workspace_rel="models/missing.onnx",
            legacy_rels=("models/missing.onnx",),
            description="missing model",
        )

