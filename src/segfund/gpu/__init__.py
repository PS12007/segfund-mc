"""GPU engines (PyTorch unfused + fused Triton). Import only when CUDA is wanted.

Windows note: triton-windows looks for its bundled TinyCC compiler and CUDA
headers under the *running* interpreter's site-packages. If torch/triton live in
a different site-packages (e.g. shared from another venv through a .pth file),
those lookups miss and Triton falls back to whatever `gcc` is on PATH, which
fails on `\\\\?\\` paths. Triton honours CC and CUDA_PATH first, so we point
them at the copies bundled inside the triton package itself, wherever it is.
Existing user settings are left alone.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _configure_triton_windows() -> None:
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("triton")
    if spec is None or not spec.submodule_search_locations:
        return
    root = Path(next(iter(spec.submodule_search_locations)))
    tcc = root / "runtime" / "tcc" / "tcc.exe"
    cuda = root / "backends" / "nvidia"
    if "CC" not in os.environ and tcc.exists():
        os.environ["CC"] = str(tcc)
    if "CUDA_PATH" not in os.environ and "CUDA_HOME" not in os.environ and (cuda / "bin").exists():
        os.environ["CUDA_PATH"] = str(cuda)


_configure_triton_windows()
