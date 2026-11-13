"""Fused Triton kernel: whole contract simulation in one launch.

Each program instance owns BLOCK paths. Their state (x, fund, discount factor,
guarantees, in-force, three cash-flow accumulators) lives in registers for all
time steps. Random numbers are generated in-register from (path, step) with
Philox. Per step the only memory traffic is one 16-float table row, which every
path in the block shares and which stays in cache. At the end each path writes
seven floats. Memory traffic is O(paths), not O(paths x steps x state variables).

Precision: all arithmetic is fp32. FAST_MATH=False (default) uses libdevice's
correctly-rounded-ish exp/log/sin/cos. FAST_MATH=True uses tl.exp etc., which
compile to hardware approximations (ex2.approx...). Phase 2 found tl.exp is
biased low by ~1.1e-8 per call; with two exp calls per step that biased the
20-year fee PV by -1.1e-2 SE and failed V8b. See LEARNING.md section 2.5.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from segfund.config import Model
from segfund.gpu.table import C, NCOL, pack
from segfund.reference.engine import PathResults
from segfund.rng import philox as P

N_OUT = 7  # pv_mat, pv_death, pv_fee, F_T, D_T, l_T, unit_T
N_OUT_STATE = 10  # ... plus x_end, GM_end, GD_end (segment runs for nested simulation)


@triton.jit
def _exp(x, FAST: tl.constexpr):
    if FAST:
        return tl.exp(x)
    else:
        return libdevice.exp(x)


@triton.jit
def _log(x, FAST: tl.constexpr):
    if FAST:
        return tl.log(x)
    else:
        return libdevice.log(x)


@triton.jit
def _cos(x, FAST: tl.constexpr):
    if FAST:
        return tl.cos(x)
    else:
        return libdevice.cos(x)


@triton.jit
def _sin(x, FAST: tl.constexpr):
    if FAST:
        return tl.sin(x)
    else:
        return libdevice.sin(x)


@triton.jit
def _u23(r):
    return ((r >> 9).to(tl.float32) + 0.5) * 1.1920928955078125e-07  # 2^-23


@triton.jit
def philox_normals_kernel(out_ptr, n, path_offset, step, stream, seed, BLOCK: tl.constexpr, FAST: tl.constexpr):
    """Test helper: writes raw Philox words and the 3 normals for one step (SoA, 7 x n)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c0 = offs + path_offset
    zero = offs * 0
    r0, r1, r2, r3 = tl.philox(seed, c0, zero + step, zero + stream, zero)
    PI: tl.constexpr = 3.141592653589793
    rad01 = tl.sqrt(-2.0 * _log(_u23(r0), FAST))
    a01 = PI * (2.0 * _u23(r1) - 1.0)
    rad2 = tl.sqrt(-2.0 * _log(_u23(r2), FAST))
    a2 = PI * (2.0 * _u23(r3) - 1.0)
    ip = out_ptr.to(tl.pointer_type(tl.int32))
    tl.store(ip + 0 * n + offs, r0.to(tl.int32, bitcast=True), mask=mask)
    tl.store(ip + 1 * n + offs, r1.to(tl.int32, bitcast=True), mask=mask)
    tl.store(ip + 2 * n + offs, r2.to(tl.int32, bitcast=True), mask=mask)
    tl.store(ip + 3 * n + offs, r3.to(tl.int32, bitcast=True), mask=mask)
    tl.store(out_ptr + 4 * n + offs, -rad01 * _cos(a01, FAST), mask=mask)
    tl.store(out_ptr + 5 * n + offs, -rad01 * _sin(a01, FAST), mask=mask)
    tl.store(out_ptr + 6 * n + offs, -rad2 * _cos(a2, FAST), mask=mask)


@triton.jit
def segfund_fused_kernel(
    tab_ptr, out_ptr, state_ptr, n_paths, path_offset, stream, seed, k_start, k_end, anti_half, n_inner,
    gmmb_pct, gmdb_pct, gfee, lapse_haz, lapse_k, lapse_lo, lapse_hi,
    BLOCK: tl.constexpr, NC: tl.constexpr, FAST: tl.constexpr, KAHAN: tl.constexpr,
    HAS_STATE: tl.constexpr, OUT_STATE: tl.constexpr, FINAL: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_paths
    g = offs + path_offset
    # antithetic pairing in blocks of 2*anti_half (rng.philox.antithetic_counter):
    # the second half of each block reuses the first half's counters with -z
    blk = 2 * anti_half + 1 - tl.minimum(anti_half, 1)  # = 2*anti_half, or 1 when there is no pairing
    second = (anti_half > 0) & ((g % blk) >= anti_half)
    c0 = tl.where(second, g - anti_half, g)
    sgn = tl.where(second, -1.0, 1.0)
    zero = offs * 0
    PI: tl.constexpr = 3.141592653589793

    if HAS_STATE:
        # nested simulation: n_inner consecutive paths start from one outer scenario's state
        n_states = n_paths // n_inner
        sidx = offs // n_inner
        x = tl.load(state_ptr + 0 * n_states + sidx, mask=mask, other=0.0)
        F = tl.load(state_ptr + 1 * n_states + sidx, mask=mask, other=1.0)
        GM = tl.load(state_ptr + 2 * n_states + sidx, mask=mask, other=1.0)
        GD = tl.load(state_ptr + 3 * n_states + sidx, mask=mask, other=1.0)
        l = tl.load(state_ptr + 4 * n_states + sidx, mask=mask, other=0.0)
    else:
        x = tl.zeros([BLOCK], dtype=tl.float32)
        F = tl.zeros([BLOCK], dtype=tl.float32)
        GM = tl.zeros([BLOCK], dtype=tl.float32)
        GD = tl.zeros([BLOCK], dtype=tl.float32)
        l = tl.full([BLOCK], 1.0, dtype=tl.float32)
    U = tl.full([BLOCK], 1.0, dtype=tl.float32)
    D = tl.full([BLOCK], 1.0, dtype=tl.float32)
    pv_death = tl.zeros([BLOCK], dtype=tl.float32)
    pv_fee = tl.zeros([BLOCK], dtype=tl.float32)
    # Kahan compensation terms (only used when KAHAN): integrated rate and the two PV sums
    I = tl.zeros([BLOCK], dtype=tl.float32)
    cI = tl.zeros([BLOCK], dtype=tl.float32)
    c_death = tl.zeros([BLOCK], dtype=tl.float32)
    c_fee = tl.zeros([BLOCK], dtype=tl.float32)

    for k in range(k_start, k_end):
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
        fund_add = tl.load(row + 16)  # = dep, except the t=0 deposit times the fund shock

        # events at the start of the step (dep = 0 / rst = 0 on most steps)
        F = F + fund_add
        GM = GM + gmmb_pct * dep
        GD = GD + gmdb_pct * dep
        GM = tl.where(rst > 0.5, tl.maximum(GM, gmmb_pct * F), GM)
        GD = tl.where(rst > 0.5, tl.maximum(GD, gmdb_pct * F), GD)
        mult = tl.minimum(tl.maximum(_exp(lapse_k * _log(F / GM, FAST), FAST), lapse_lo), lapse_hi)
        surv_lapse = _exp(-lapse_haz * h * mult, FAST)

        # correlated shocks from in-register Philox normals
        r0, r1, r2, r3 = tl.philox(seed, c0, zero + k, zero + stream, zero)
        rad01 = tl.sqrt(-2.0 * _log(_u23(r0), FAST))
        a01 = PI * (2.0 * _u23(r1) - 1.0)
        rad2 = tl.sqrt(-2.0 * _log(_u23(r2), FAST))
        z0 = -rad01 * _cos(a01, FAST) * sgn
        z1 = -rad01 * _sin(a01, FAST) * sgn
        z2 = -rad2 * _cos(PI * (2.0 * _u23(r3) - 1.0), FAST) * sgn
        e0 = L00 * z0
        e1 = L10 * z0 + L11 * z1
        e2 = L20 * z0 + L21 * z1 + L22 * z2

        int_r = x * Bh + (y_shift + phi_int) + e1
        x = x * decay + x_shift + e0
        growth = _exp(int_r + lnF_drift + e2, FAST)
        F = F * growth
        U = U * growth
        if KAHAN:
            # discount factor from a compensated running sum of int r: D = exp(-I)
            yI = int_r - cI
            tI = I + yI
            cI = (tI - I) - yI
            I = tI
            D = _exp(-I, FAST)
            term = D * (l * qd) * tl.maximum(GD - F, 0.0) - c_death
            t = pv_death + term
            c_death = (t - pv_death) - term
            pv_death = t
            term = D * l * (gfee * h) * F - c_fee
            t = pv_fee + term
            c_fee = (t - pv_fee) - term
            pv_fee = t
        else:
            D = D * _exp(-int_r, FAST)
            pv_death += D * (l * qd) * tl.maximum(GD - F, 0.0)
            pv_fee += D * l * (gfee * h) * F
        l = l * (1.0 - qd) * surv_lapse

    if FINAL:
        pv_mat = D * l * tl.maximum(GM - F, 0.0)
    else:
        pv_mat = tl.zeros([BLOCK], dtype=tl.float32)
    tl.store(out_ptr + 0 * n_paths + offs, pv_mat, mask=mask)
    tl.store(out_ptr + 1 * n_paths + offs, pv_death, mask=mask)
    tl.store(out_ptr + 2 * n_paths + offs, pv_fee, mask=mask)
    tl.store(out_ptr + 3 * n_paths + offs, F, mask=mask)
    tl.store(out_ptr + 4 * n_paths + offs, D, mask=mask)
    tl.store(out_ptr + 5 * n_paths + offs, l, mask=mask)
    tl.store(out_ptr + 6 * n_paths + offs, U, mask=mask)
    if OUT_STATE:
        tl.store(out_ptr + 7 * n_paths + offs, x, mask=mask)
        tl.store(out_ptr + 8 * n_paths + offs, GM, mask=mask)
        tl.store(out_ptr + 9 * n_paths + offs, GD, mask=mask)


def _check_range(n_paths: int, path_offset: int):
    if path_offset < 0 or path_offset + n_paths >= 2**31:
        raise ValueError("path ids must fit in int32 (fused kernel uses 32-bit counters)")


class FusedEngine:
    """Holds the packed table on the GPU so repeated launches (benchmarks) don't re-upload."""

    def __init__(
        self,
        model: Model,
        block: int = 256,
        num_warps: int = 4,
        device: str = "cuda",
        fast_math: bool = False,
        kahan: bool = False,
    ):
        import segfund.gpu  # noqa: F401  (sets CC / CUDA_PATH for triton-windows)

        self.model = model
        self.pm = pack(model)
        self.table = torch.tensor(self.pm.table, dtype=torch.float32, device=device).contiguous()
        self.block = block
        self.num_warps = num_warps
        self.device = device
        self.fast_math = fast_math
        self.kahan = kahan

    def launch(
        self,
        out: torch.Tensor,
        n_paths: int,
        seed: int,
        path_offset: int = 0,
        stream: int = P.STREAM_MAIN,
        antithetic_half: int = 0,
        k_start: int = 0,
        k_end: int | None = None,
        state: torch.Tensor | None = None,
        n_inner: int = 1,
        out_state: bool = False,
    ):
        """Launch the kernel. For nested simulation pass a (5, n_paths // n_inner) fp32
        `state` (rows x, F, GM, GD, l) and a step range; `out` must then have
        N_OUT_STATE rows when out_state=True."""
        _check_range(n_paths, path_offset)
        pm = self.pm
        k_end = pm.n_steps if k_end is None else k_end
        if not 0 <= k_start < k_end <= pm.n_steps:
            raise ValueError("need 0 <= k_start < k_end <= n_steps")
        if state is not None:
            if n_paths % n_inner or state.shape != (5, n_paths // n_inner) or not state.is_contiguous():
                raise ValueError("state must be contiguous (5, n_paths // n_inner)")
        if antithetic_half and (n_inner > 1) and 2 * antithetic_half != n_inner:
            raise ValueError("nested antithetic pairs must sit inside one outer scenario (2*half == n_inner)")
        grid = (triton.cdiv(n_paths, self.block),)
        segfund_fused_kernel[grid](
            self.table, out, state if state is not None else self.table, n_paths, path_offset, stream, seed,
            k_start, k_end, antithetic_half, n_inner,
            pm.gmmb_pct, pm.gmdb_pct, pm.guarantee_fee, pm.lapse_hazard, pm.lapse_k, pm.lapse_lo, pm.lapse_hi,
            BLOCK=self.block, NC=NCOL, FAST=self.fast_math, KAHAN=self.kahan,
            HAS_STATE=state is not None, OUT_STATE=out_state, FINAL=k_end == pm.n_steps,
            num_warps=self.num_warps,
        )

    def alloc(self, n_paths: int, out_state: bool = False) -> torch.Tensor:
        return torch.empty((N_OUT_STATE if out_state else N_OUT) * n_paths, dtype=torch.float32, device=self.device)

    def simulate(
        self,
        n_paths: int,
        seed: int,
        path_offset: int = 0,
        stream: int = P.STREAM_MAIN,
        antithetic: bool | None = None,
    ) -> PathResults:
        anti = self.model.sim.antithetic if antithetic is None else antithetic
        if anti and (n_paths % 2 or path_offset):
            raise ValueError("antithetic runs need an even path count and path_offset = 0")
        out = self.alloc(n_paths)
        self.launch(out, n_paths, seed, path_offset, stream, n_paths // 2 if anti else 0)
        return results_from_buffer(out, n_paths)


def results_from_buffer(out: torch.Tensor, n_paths: int) -> PathResults:
    rows = out.numel() // n_paths
    a = out.view(rows, n_paths).double().cpu().numpy()
    extra = (a[7], a[8], a[9]) if rows == N_OUT_STATE else (None, None, None)
    return PathResults(a[0] + a[1] - a[2], a[0], a[1], a[2], a[3], a[4], a[5], a[6], *extra)


def philox_normals_gpu(n: int, step: int, seed: int, path_offset: int = 0, stream: int = P.STREAM_MAIN, fast_math: bool = False):
    """Returns (raw uint32 words as int64 numpy (4, n), normals float64 numpy (3, n)) from the Triton RNG."""
    import segfund.gpu  # noqa: F401

    _check_range(n, path_offset)
    out = torch.empty(7 * n, dtype=torch.float32, device="cuda")
    BLOCK = 1024
    philox_normals_kernel[(triton.cdiv(n, BLOCK),)](out, n, path_offset, step, stream, seed, BLOCK=BLOCK, FAST=fast_math)
    raw = out.view(7, n)[:4].contiguous().view(torch.int32).cpu().numpy().astype(np.int64) & 0xFFFFFFFF
    z = out.view(7, n)[4:].double().cpu().numpy()
    return raw, z
