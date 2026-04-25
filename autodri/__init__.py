from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_SRC_PKG = _ROOT / "src" / "autodri"

if _SRC_PKG.exists():
    __path__.append(str(_SRC_PKG))  # type: ignore[name-defined]

