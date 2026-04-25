from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable, Sequence


WORKSPACE_ENV_VAR = "AUTODRI_WORKSPACE"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_workspace_root() -> Path:
    root = repo_root()
    return root.parent / f"{root.name}_workspace"


def workspace_root(create: bool = False) -> Path:
    raw = str(os.environ.get(WORKSPACE_ENV_VAR, "")).strip()
    root = Path(raw).expanduser() if raw else default_workspace_root()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def workspace_path(*parts: str, create: bool = False) -> Path:
    path = workspace_root(create=create).joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def models_root(create: bool = False) -> Path:
    return workspace_path("models", create=create)


def data_root(create: bool = False) -> Path:
    return workspace_path("data", create=create)


def artifacts_root(create: bool = False) -> Path:
    return workspace_path("artifacts", create=create)


def archive_root(create: bool = False) -> Path:
    return workspace_path("archive", create=create)


def sources_root(create: bool = False) -> Path:
    return workspace_path("sources", create=create)


def reports_root(create: bool = False) -> Path:
    path = artifacts_root(create=create) / "reports"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def manifests_current_root(create: bool = False) -> Path:
    path = artifacts_root(create=create) / "manifests" / "current"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def participant_analysis_dir(participant: str, create: bool = False) -> Path:
    participant = str(participant).strip()
    if participant == "p1":
        path = data_root(create=create) / "natural_driving_p1" / "analysis"
    else:
        path = data_root(create=create) / "natural_driving" / participant / "analysis"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def participant_videos_root(participant: str) -> Path:
    participant = str(participant).strip()
    if participant == "p1":
        return data_root() / "natural_driving_p1" / "p1_剪辑好的视频"
    return data_root() / "natural_driving" / participant / "剪辑好的视频"


def _warn_legacy(path: Path, description: str) -> None:
    warnings.warn(
        f"{description} resolved from legacy repo path {path}; move it under "
        f"{workspace_root()} or set {WORKSPACE_ENV_VAR}.",
        DeprecationWarning,
        stacklevel=2,
    )


def _candidate_paths(workspace_rel: str | None, legacy_rels: Sequence[str]) -> Iterable[tuple[Path, bool]]:
    if workspace_rel:
        yield workspace_root() / workspace_rel, False
    for rel in legacy_rels:
        yield repo_root() / rel, True


def resolve_existing_path(
    cli_value: str,
    *,
    workspace_rel: str | None = None,
    legacy_rels: Sequence[str] = (),
    description: str = "resource",
) -> Path:
    raw = str(cli_value or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"{description} not found: {path}")

    for candidate, is_legacy in _candidate_paths(workspace_rel, legacy_rels):
        if candidate.exists():
            if is_legacy:
                _warn_legacy(candidate, description)
            return candidate

    raise FileNotFoundError(
        f"Unable to resolve {description}. Checked workspace_rel={workspace_rel!r}, "
        f"legacy_rels={list(legacy_rels)!r}."
    )


def resolve_output_path(
    cli_value: str,
    *,
    workspace_rel: str | None = None,
    legacy_rel: str | None = None,
    create_parent: bool = True,
) -> Path:
    raw = str(cli_value or "").strip()
    if raw:
        path = Path(raw).expanduser()
    elif workspace_rel:
        path = workspace_root(create=create_parent) / workspace_rel
    elif legacy_rel:
        path = repo_root() / legacy_rel
        _warn_legacy(path, "output path")
    else:
        raise ValueError("resolve_output_path requires cli_value, workspace_rel, or legacy_rel")
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_workspace_or_repo_path(raw: str) -> Path:
    path = Path(str(raw or "").strip()).expanduser()
    if path.is_absolute():
        return path
    repo_candidate = repo_root() / path
    workspace_candidate = workspace_root() / path
    if workspace_candidate.exists():
        return workspace_candidate
    if repo_candidate.exists():
        return repo_candidate
    if path.parts and path.parts[0] in {"data", "models", "artifacts", "archive", "sources"}:
        return workspace_candidate
    return repo_candidate
