"""Ways to add up N per-path values into one total, for the precision study.

Why this matters: fp32 has a 24-bit significand (~7 digits). When a running total
S is much larger than the next term x, S + x is rounded to a multiple of S's last
digit, so a large part of x can be lost. A sequential fp32 sum of 10^7 fee values
of ~2,500 reaches ~2.5e10, where the spacing of fp32 numbers is 2048: every
addition throws away up to ~1,000 dollars of rounding, always biased the same way
when the terms are similar in size.

Methods (all inputs are the fp32 values the GPU kernel produced):
  sequential_fp32   one running fp32 accumulator, in order (a single thread or an atomic add)
  kahan_lanes_fp32  4096 lanes each Kahan-sum a strided slice, then Kahan over the lanes
                    (the shape a GPU reduction takes: per-thread loop, then combine)
  pairwise_fp32     numpy's fp32 sum (blocked pairwise summation)
  torch_fp32_gpu    torch.sum on the GPU in fp32 (tree reduction across blocks)
  torch_fp64acc_gpu torch.sum on the GPU with fp64 accumulation of the fp32 values
  reference         fp64 pairwise sum of the same fp32 values (checked against math.fsum)
"""

from __future__ import annotations

import math

import numpy as np


def sequential_fp32(x: np.ndarray) -> float:
    return float(np.cumsum(np.asarray(x, dtype=np.float32), dtype=np.float32)[-1])


def _kahan_scalar_fp32(values: np.ndarray) -> np.float32:
    s = np.float32(0.0)
    c = np.float32(0.0)
    for v in values.astype(np.float32):
        y = np.float32(v - c)
        t = np.float32(s + y)
        c = np.float32(np.float32(t - s) - y)
        s = t
    return s


def kahan_lanes_fp32(x: np.ndarray, lanes: int = 4096) -> float:
    x = np.asarray(x, dtype=np.float32)
    rows = -(-x.size // lanes)
    pad = np.zeros(rows * lanes, dtype=np.float32)
    pad[: x.size] = x
    m = pad.reshape(rows, lanes)
    s = np.zeros(lanes, dtype=np.float32)
    c = np.zeros(lanes, dtype=np.float32)
    for r in range(rows):
        y = m[r] - c
        t = s + y
        c = (t - s) - y
        s = t
    return float(_kahan_scalar_fp32(s))


def pairwise_fp32(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.float32), dtype=np.float32))


def torch_fp32_gpu(x) -> float:
    import torch

    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x, dtype=np.float32), device="cuda")
    return float(torch.sum(t, dtype=torch.float32).item())


def torch_fp64acc_gpu(x) -> float:
    import torch

    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x, dtype=np.float32), device="cuda")
    return float(torch.sum(t, dtype=torch.float64).item())


def reference(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.float32).astype(np.float64)))


def exact(x: np.ndarray) -> float:
    """Correctly rounded sum (math.fsum). Slow; used to check `reference` on smaller N."""
    return math.fsum(np.asarray(x, dtype=np.float32).astype(np.float64).tolist())


CPU_METHODS = {
    "sequential_fp32": sequential_fp32,
    "kahan_lanes_fp32": kahan_lanes_fp32,
    "pairwise_fp32": pairwise_fp32,
}
GPU_METHODS = {
    "torch_fp32_gpu": torch_fp32_gpu,
    "torch_fp64acc_gpu": torch_fp64acc_gpu,
}
