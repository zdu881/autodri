#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap ONNXRuntime CUDA shared-library resolution for conda environments.

Background:
- On some machines, `onnxruntime-gpu` is installed correctly and reports
  `CUDAExecutionProvider` as available.
- Session creation still falls back to CPU because `libonnxruntime_providers_cuda.so`
  cannot resolve `libcudnn.so.8`.
- The cuDNN library is often present under `<conda-env>/lib`, but that directory is
  missing from `LD_LIBRARY_PATH` for the current process.

This helper prepends the active environment's `lib/` directory to
`LD_LIBRARY_PATH` and re-execs the current Python process once, before
`onnxruntime` is imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List


_ENV_FLAG = "AUTODRI_ORT_LD_BOOTSTRAPPED"


def _candidate_lib_dirs() -> List[Path]:
    out: List[Path] = []
    seen = set()

    conda_prefix = str(os.environ.get("CONDA_PREFIX", "")).strip()
    if conda_prefix:
        p = Path(conda_prefix) / "lib"
        if p not in seen:
            out.append(p)
            seen.add(p)

    exe = Path(sys.executable).resolve()
    exe_env_lib = exe.parent.parent / "lib"
    if exe_env_lib not in seen:
        out.append(exe_env_lib)
        seen.add(exe_env_lib)

    return [p for p in out if p.is_dir()]


def ensure_onnxruntime_cuda_runtime() -> None:
    if os.name != "posix":
        return
    if os.environ.get(_ENV_FLAG) == "1":
        return

    lib_dirs = _candidate_lib_dirs()
    if not lib_dirs:
        return

    current = [x for x in str(os.environ.get("LD_LIBRARY_PATH", "")).split(":") if x]
    missing = [str(p) for p in lib_dirs if str(p) not in current]
    if not missing:
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + current)
    os.environ[_ENV_FLAG] = "1"
    argv = list(getattr(sys, "orig_argv", []) or [sys.executable, *sys.argv])
    if argv and argv[0] != sys.executable:
        argv[0] = sys.executable
    sys.stderr.write(
        "[autodri] Re-exec with updated LD_LIBRARY_PATH for ONNXRuntime CUDA: "
        + os.environ["LD_LIBRARY_PATH"]
        + "\n"
    )
    sys.stderr.flush()
    os.execvpe(sys.executable, argv, os.environ)
