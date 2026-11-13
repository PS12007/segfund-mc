"""Fused Triton kernel: whole contract simulation in one launch.

Each program instance owns BLOCK paths. Their state (x, fund, discount factor,
guarantees, in-force, three cash-flow accumulators) lives in registers for all
time steps. Random numbers are generated in-register from (path, step) with
Philox. Per step the only memory traffic is one 16-float table row, which every
path in the block shares and which stays in cache. At the end each path writes
six floats. Memory traffic is O(paths), not O(paths x steps x state variables).

Precision: all arithmetic is fp32. tl.exp / tl.log / tl.sin / tl.cos compile to
fast hardware approximations; phase 2 measures the resulting error against the
fp64 reference (see segfund.gpu.equality).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl

from segfund.config import Model
from segfund.gpu.table import C, NCOL, pack
from segfund.reference.engine import PathResults
from segfund.rng import philox as P

N_OUT = 6  # pv_mat, pv_death, pv_fee, F_T, D_T, l_T


@triton.jit
def _u23(r):
    return ((r >> 9).to(tl.float32) + 0.5) * 1.1920928955078125e-07  # 2^-23


@triton.jit
def philox_normals_kernel(out_ptr, n, path_offset, step, stream, seed, BLOCK: tl.constexpr):
    """Test helper: writes raw Philox words and the 3 normals for one step (SoA, 7 x n)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c0 = offs + path_offset
    zero = offs * 0
    r0, r1, r2, r3 = tl.philox(seed, c0, zero + step, zero + stream, zero)
    PI: tl.constexpr = 3.141592653589793
    rad01 = tl.sqrt(-2.0 * tl.log(_u23(r0)))
    a01 = PI * (2.0 * _u23(r1) - 1.0)
    rad2 = tl.sqrt(-2.0 * tl.log(_u23(r2)))
    a2 = PI * (2.0 * _u23(r3) - 1.0)
    ip = out_ptr.to(tl.pointer_type(tl.int32))
    tl.store(ip + 0 * n + offs, r0.to(tl.int32, bitcast=True), mask=mask)
    tl.store(ip + 1 * n + offs, r1.to(tl.int32, bitcast=True), mask=mask)
    tl.store(ip + 2 * n + offs, r2.to(tl.int32, bitcast=True), mask=mask)
    tl.store(ip + 3 * n + offs, r3.to(tl.int32, bitcast=True), mask=mask)
    tl.store(out_ptr + 4 * n + offs, -rad01 * tl.cos(a01), mask=mask)
    tl.store(out_ptr + 5 * n + offs, -rad01 * tl.sin(a01), mask=mask)
    tl.store(out_ptr + 6 * n + offs, -rad2 * tl.cos(a2), mask=mask)


@triton.jit
def segfund_fused_kernel(
    tab_ptr, out_ptr, n_paths, path_offset, stream, seed, n_steps,
    gmmb_pct, gmdb_pct, gfee, lapse_haz, lapse_k, lapse_lo, lapse_hi,
    BLOCK: tl.constexpr, NC: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_paths
    c0 = offs + path_offset
    zero = offs * 0
    PI: tl.constexpr = 3.141592653589793

    x = tl.zeros([BLOCK], dtype=tl.float32)
    F = tl.zeros([BLOCK], dtype=tl.float32)
    GM = tl.zeros([BLOCK], dtype=tl.float32)
    GD = tl.zeros([BLOCK], dtype=tl.float32)
    D = tl.full([BLOCK], 1.0, dtype=tl.float32)
    l = tl.full([BLOCK], 1.0, dtype=tl.float32)
    pv_death = tl.zeros([BLOCK], dtype=tl.float32)
    pv_fee = tl.zeros([BLOCK], dtype=tl.float32)

    for k in range(n_steps):
        row = tab_ptr + k * NC
        decay = tl.load(row + 0)
        Bh = tl.load(row + 1)
        phi_int = tl.load(row + 2)
        x_shift = tl.load(row + 3)
        y_shift = tl.load(row + 4)
        lnF_drift = tl.load(row + 5)
        L00 = tl.load(row + 6)
        L10 = tl.load(row + 7)
        L11 = tl.load(row + 8)
        L20 = tl.load(row + 9)
        L21 = tl.load(row + 10)
        L22 = tl.load(row + 11)
        h = tl.load(row + 12)
        qd = tl.load(row + 13)
        dep = tl.load(row + 14)
        rst = tl.load(row + 15)

        # events at the start of the step (dep = 0 / rst = 0 on most steps)
        F = F + dep
        GM = GM + gmmb_pct * dep
        GD = GD + gmdb_pct * dep
        GM = tl.where(rst > 0.5, tl.maximum(GM, gmmb_pct * F), GM)
        GD = tl.where(rst > 0.5, tl.maximum(GD, gmdb_pct * F), GD)
        mult = tl.minimum(tl.maximum(tl.exp(lapse_k * tl.log(F / GM)), lapse_lo), lapse_hi)
        surv_lapse = tl.exp(-lapse_haz * h * mult)

        # correlated shocks from in-register Philox normals
        r0, r1, r2, r3 = tl.philox(seed, c0, zero + k, zero + stream, zero)
        rad01 = tl.sqrt(-2.0 * tl.log(_u23(r0)))
        a01 = PI * (2.0 * _u23(r1) - 1.0)
        rad2 = tl.sqrt(-2.0 * tl.log(_u23(r2)))
        z0 = -rad01 * tl.cos(a01)
        z1 = -rad01 * tl.sin(a01)
        z2 = -rad2 * tl.cos(PI * (2.0 * _u23(r3) - 1.0))
        e0 = L00 * z0
        e1 = L10 * z0 + L11 * z1
        e2 = L20 * z0 + L21 * z1 + L22 * z2

        int_r = x * Bh + (y_shift + phi_int) + e1
        x = x * decay + x_shift + e0
        F = F * tl.exp(int_r + lnF_drift + e2)
        D = D * tl.exp(-int_r)

        pv_death += D * (l * qd) * tl.maximum(GD - F, 0.0)
        pv_fee += D * l * (gfee * h) * F
        l = l * (1.0 - qd) * surv_lapse

    pv_mat = D * l * tl.maximum(GM - F, 0.0)
    tl.store(out_ptr + 0 * n_paths + offs, pv_mat, mask=mask)
    tl.store(out_ptr + 1 * n_paths + offs, pv_death, mask=mask)
    tl.store(out_ptr + 2 * n_paths + offs, pv_fee, mask=mask)
    tl.store(out_ptr + 3 * n_paths + offs, F, mask=mask)
    tl.store(out_ptr + 4 * n_paths + offs, D, mask=mask)
    tl.store(out_ptr + 5 * n_paths + offs, l, mask=mask)


def _check_range(n_paths: int, path_offset: int):
    if path_offset < 0 or path_offset + n_paths >= 2**31:
        raise ValueError("path ids must fit in int32 (fused kernel uses 32-bit counters)")


class FusedEngine:
    """Holds the packed table on the GPU so repeated launches (benchmarks) don't re-upload."""

    def __init__(self, model: Model, block: int = 256, num_warps: int = 4, device: str = "cuda"):
        import segfund.gpu  # noqa: F401  (sets CC / CUDA_PATH for triton-windows)

        self.pm = pack(model)
        self.table = torch.tensor(self.pm.table, dtype=torch.float32, device=device).contiguous()
        self.block = block
        self.num_warps = num_warps
        self.device = device

    def launch(self, out: torch.Tensor, n_paths: int, seed: int, path_offset: int = 0, stream: int = P.STREAM_MAIN):
        _check_range(n_paths, path_offset)
        pm = self.pm
        grid = (triton.cdiv(n_paths, self.block),)
        segfund_fused_kernel[grid](
            self.table, out, n_paths, path_offset, stream, seed, pm.n_steps,
            pm.gmmb_pct, pm.gmdb_pct, pm.guarantee_fee, pm.lapse_hazard, pm.lapse_k, pm.lapse_lo, pm.lapse_hi,
            BLOCK=self.block, NC=NCOL, num_warps=self.num_warps,
        )

    def alloc(self, n_paths: int) -> torch.Tensor:
        return torch.empty(N_OUT * n_paths, dtype=torch.float32, device=self.device)

    def simulate(self, n_paths: int, seed: int, path_offset: int = 0, stream: int = P.STREAM_MAIN) -> PathResults:
        out = self.alloc(n_paths)
        self.launch(out, n_paths, seed, path_offset, stream)
        a = out.view(N_OUT, n_paths).double().cpu().numpy()
        return PathResults(a[0] + a[1] - a[2], a[0], a[1], a[2], a[3], a[4], a[5])


def simulate_fused(model: Model, n_paths: int, seed: int, path_offset: int = 0, **kw) -> PathResults:
    return FusedEngine(model, **kw).simulate(n_paths, seed, path_offset)


def philox_normals_gpu(n: int, step: int, seed: int, path_offset: int = 0, stream: int = P.STREAM_MAIN):
    """Returns (raw uint32 words as int64 numpy (4, n), normals float64 numpy (3, n)) from the Triton RNG."""
    import segfund.gpu  # noqa: F401

    _check_range(n, path_offset)
    out = torch.empty(7 * n, dtype=torch.float32, device="cuda")
    BLOCK = 1024
    philox_normals_kernel[(triton.cdiv(n, BLOCK),)](out, n, path_offset, step, stream, seed, BLOCK=BLOCK)
    raw = out.view(7, n)[:4].contiguous().view(torch.int32).cpu().numpy().astype(np.int64) & 0xFFFFFFFF
    z = out.view(7, n)[4:].double().cpu().numpy()
    return raw, z
