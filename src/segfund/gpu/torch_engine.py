"""Unfused PyTorch GPU engine.

Same algorithm as the NumPy reference, written as ordinary tensor operations.
Every operation is its own CUDA kernel that reads its inputs from GPU memory and
writes its output back, so each time step makes dozens of round trips through
DRAM. That is the baseline the fused Triton kernel is compared against, and the
fallback if Triton is unavailable.

Philox here uses int64 tensors holding uint32 values. PyTorch has no uint64, and
a 32x32-bit product overflows int64, so mulhilo is done with 16-bit limbs.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from segfund.config import Model
from segfund.gpu.table import C, pack
from segfund.reference.engine import PathResults
from segfund.rng import philox as P

_M16 = 0xFFFF
_M32 = 0xFFFFFFFF


def _mulhilo(a: torch.Tensor, b: int):
    """(hi, lo) 32-bit halves of a * b for a < 2^32 (int64 tensor), b < 2^32 (int)."""
    a0, a1 = a & _M16, a >> 16
    b0, b1 = b & _M16, b >> 16
    p00, p01, p10, p11 = a0 * b0, a0 * b1, a1 * b0, a1 * b1
    mid = (p00 >> 16) + (p01 & _M16) + (p10 & _M16)
    lo = ((mid & _M16) << 16) | (p00 & _M16)
    hi = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16)
    return hi, lo


def philox4x32_torch(c0: torch.Tensor, c1: int, c2: int, c3: int, seed: int, n_rounds: int = P.N_ROUNDS):
    seed &= 0xFFFFFFFFFFFFFFFF
    k0, k1 = seed & _M32, seed >> 32
    c1 = torch.full_like(c0, c1 & _M32)
    c2 = torch.full_like(c0, c2 & _M32)
    c3 = torch.full_like(c0, c3 & _M32)
    for _ in range(n_rounds):
        hiB, loB = _mulhilo(c2, P.M_B)
        hiA, loA = _mulhilo(c0, P.M_A)
        c0, c1, c2, c3 = hiB ^ c1 ^ k0, loB, hiA ^ c3 ^ k1, loA
        k0 = (k0 + P.W_A) & _M32
        k1 = (k1 + P.W_B) & _M32
    return c0, c1, c2, c3


def normals3_torch(path_ids: torch.Tensor, step: int, seed: int, stream: int, dtype) -> torch.Tensor:
    r0, r1, r2, r3 = philox4x32_torch(path_ids, step, stream, 0, seed)
    u = [((r >> 9).to(dtype) + 0.5) * P.U23_SCALE for r in (r0, r1, r2, r3)]
    rad01 = torch.sqrt(-2.0 * torch.log(u[0]))
    ang01 = math.pi * (2.0 * u[1] - 1.0)
    rad2 = torch.sqrt(-2.0 * torch.log(u[2]))
    ang2 = math.pi * (2.0 * u[3] - 1.0)
    return -rad01 * torch.cos(ang01), -rad01 * torch.sin(ang01), -rad2 * torch.cos(ang2)


def simulate_torch(
    model: Model,
    n_paths: int,
    seed: int,
    path_offset: int = 0,
    stream: int = P.STREAM_MAIN,
    dtype=torch.float32,
    device: str = "cuda",
    antithetic_half: int = 0,
) -> PathResults:
    pm = pack(model)
    tab = pm.table  # fp64 on host; each scalar is rounded to `dtype` when used with tensors
    g = torch.arange(path_offset, path_offset + n_paths, device=device, dtype=torch.int64)
    second = (g >= antithetic_half) if antithetic_half > 0 else torch.zeros_like(g, dtype=torch.bool)
    ids = torch.where(second, g - antithetic_half, g)
    sgn = torch.where(second, -1.0, 1.0).to(dtype)
    zeros = lambda: torch.zeros(n_paths, device=device, dtype=dtype)  # noqa: E731
    x, F, GM, GD = zeros(), zeros(), zeros(), zeros()
    U = torch.ones(n_paths, device=device, dtype=dtype)
    D = torch.ones(n_paths, device=device, dtype=dtype)
    l = torch.ones(n_paths, device=device, dtype=dtype)
    pv_death, pv_fee = zeros(), zeros()

    for k in range(pm.n_steps):
        row = [float(v) for v in tab[k]]  # python floats: keep tensor math in `dtype`
        dep = row[C["deposit"]]
        if dep > 0:
            F += row[C["fund_add"]]
            GM += pm.gmmb_pct * dep
            GD += pm.gmdb_pct * dep
        if row[C["reset"]] > 0.5:
            GM = torch.maximum(GM, pm.gmmb_pct * F)
            GD = torch.maximum(GD, pm.gmdb_pct * F)
        h = float(row[C["h"]])
        mult = torch.clamp(torch.pow(F / GM, pm.lapse_k), pm.lapse_lo, pm.lapse_hi)
        surv_lapse = torch.exp(-pm.lapse_hazard * h * mult)

        z0, z1, z2 = normals3_torch(ids, k, seed, stream, dtype)
        z0, z1, z2 = z0 * sgn, z1 * sgn, z2 * sgn
        e0 = row[C["L00"]] * z0
        e1 = row[C["L10"]] * z0 + row[C["L11"]] * z1
        e2 = row[C["L20"]] * z0 + row[C["L21"]] * z1 + row[C["L22"]] * z2
        int_r = x * row[C["Bh"]] + (row[C["y_shift"]] + row[C["phi_int"]]) + e1
        x = x * row[C["decay"]] + row[C["x_shift"]] + e0
        growth = torch.exp(int_r + row[C["lnF_drift"]] + e2)
        F = F * growth
        U = U * growth
        D = D * torch.exp(-int_r)

        qd = float(row[C["q_death"]])
        pv_death += D * (l * qd) * torch.clamp(GD - F, min=0.0)
        pv_fee += D * l * (pm.guarantee_fee * h) * F
        l = l * (1.0 - qd) * surv_lapse

    pv_mat = D * l * torch.clamp(GM - F, min=0.0)
    out = [t.double().cpu().numpy() for t in (pv_mat, pv_death, pv_fee, F, D, l, U)]
    pv_mat, pv_death, pv_fee, F, D, l, U = out
    return PathResults(pv_mat + pv_death - pv_fee, pv_mat, pv_death, pv_fee, F, D, l, U)
