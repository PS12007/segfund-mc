"""PyTorch engine with torch.compile: the "just compile it" baseline.

The unfused engine (torch_engine.py) launches dozens of small kernels per time
step from Python. torch.compile (TorchInductor) can fuse a step's element-wise
operations into a few generated Triton kernels. That is the obvious question to
ask of a hand-written fused kernel: how much of its speed does the compiler get
for free?

Design, chosen to give the compiler a fair chance:
  * One time step is compiled, branch-free (deposits are added every step, mostly
    zero; resets are a torch.where), with the step's table row passed as a tensor
    so every step reuses the same compiled graph. Compiling the unrolled 120-step
    loop instead would bake 120 sets of constants into one huge graph.
  * The normals come from torch.randn with the same generator calls as the eager
    torch.randn engine, so the two run identical scenarios and can be compared
    path by path.
  * mode="default" fuses within the step; mode="reduce-overhead" also captures
    the step in a CUDA graph, which removes most of the per-kernel launch cost.
"""

from __future__ import annotations

import torch

import segfund.gpu  # noqa: F401  (CC / CUDA_PATH for triton-windows)
from segfund.config import Model
from segfund.gpu.table import C, pack


def _step(x, F, GM, GD, U, D, l, pvd, pvf, row, z, gmmb: float, gmdb: float, fee: float,
          hz: float, lk: float, llo: float, lhi: float):
    dep = row[C["deposit"]:C["deposit"] + 1]
    F = F + row[C["fund_add"]:C["fund_add"] + 1]
    GM = GM + gmmb * dep
    GD = GD + gmdb * dep
    reset = row[C["reset"]:C["reset"] + 1] > 0.5
    GM = torch.where(reset, torch.maximum(GM, gmmb * F), GM)
    GD = torch.where(reset, torch.maximum(GD, gmdb * F), GD)
    h = row[C["h"]:C["h"] + 1]
    mult = torch.clamp(torch.pow(F / GM, lk), llo, lhi)
    surv_lapse = torch.exp(-hz * h * mult)
    z0, z1, z2 = z[0], z[1], z[2]
    e0 = row[C["L00"]:C["L00"] + 1] * z0
    e1 = row[C["L10"]:C["L10"] + 1] * z0 + row[C["L11"]:C["L11"] + 1] * z1
    e2 = row[C["L20"]:C["L20"] + 1] * z0 + row[C["L21"]:C["L21"] + 1] * z1 + row[C["L22"]:C["L22"] + 1] * z2
    int_r = x * row[C["Bh"]:C["Bh"] + 1] + (row[C["y_shift"]:C["y_shift"] + 1] + row[C["phi_int"]:C["phi_int"] + 1]) + e1
    x = x * row[C["decay"]:C["decay"] + 1] + row[C["x_shift"]:C["x_shift"] + 1] + e0
    growth = torch.exp(int_r + row[C["lnF_drift"]:C["lnF_drift"] + 1] + e2)
    F = F * growth
    U = U * growth
    D = D * torch.exp(-int_r)
    q = row[C["q_death"]:C["q_death"] + 1]
    pvd = pvd + D * (l * q) * torch.clamp(GD - F, min=0.0)
    pvf = pvf + D * l * (fee * h) * F
    l = l * (1.0 - q) * surv_lapse
    return x, F, GM, GD, U, D, l, pvd, pvf


def _step_inplace(S, row, z, gmmb: float, gmdb: float, fee: float, hz: float, lk: float, llo: float, lhi: float):
    """One step on the packed state S (9, n): rows x, F, GM, GD, U, D, l, pvd, pvf, updated in place."""
    S.copy_(torch.stack(_step(*S.unbind(0), row, z, gmmb, gmdb, fee, hz, lk, llo, lhi)))


_COMPILED = None


def compiled_step():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = torch.compile(_step_inplace, fullgraph=True, dynamic=False)
    return _COMPILED


class CompiledEngine:
    """torch.compile'd time step, optionally replayed from a CUDA graph.

    mode="compile": the compiled step is launched from Python each month.
    mode="compile+graph": the compiled step is captured once in a torch.cuda.CUDAGraph
    over fixed buffers; each month copies in the table row and the normals and
    replays it, so the step's kernels cost one launch instead of several.
    simulate() returns device tensors in the order of simulate_torch_tensors.
    """

    def __init__(self, model: Model, mode: str = "compile", device: str = "cuda", dtype=torch.float32):
        if mode not in ("compile", "compile+graph"):
            raise ValueError("mode must be 'compile' or 'compile+graph'")
        self.pm = pm = pack(model)
        self.rows = torch.tensor(pm.table, dtype=dtype, device=device)
        self.mode, self.device, self.dtype = mode, device, dtype
        self.step = compiled_step()
        # plain Python floats: a NumPy float64 here becomes a CPU tensor in the graph,
        # and a CPU kernel needs MSVC on Windows
        self.consts = tuple(float(v) for v in (pm.gmmb_pct, pm.gmdb_pct, pm.guarantee_fee, pm.lapse_hazard,
                                               pm.lapse_k, pm.lapse_lo, pm.lapse_hi))
        self._graphs: dict[int, tuple] = {}

    def _initial_state(self, n: int, S: torch.Tensor | None = None) -> torch.Tensor:
        if S is None:
            S = torch.empty((9, n), device=self.device, dtype=self.dtype)
        S.zero_()
        S[4:7].fill_(1.0)  # U, D, l start at 1
        return S

    def _graph(self, n: int):
        if n not in self._graphs:
            S = self._initial_state(n)
            row = self.rows[0].clone()
            z = torch.zeros((3, n), device=self.device, dtype=self.dtype)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # warm up (compile) outside the capture
                for _ in range(2):
                    self.step(S, row, z, *self.consts)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.step(S, row, z, *self.consts)
            self._graphs[n] = (g, S, row, z)
        return self._graphs[n]

    def simulate(self, n_paths: int, seed: int) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
        shape = (3, n_paths)
        if self.mode == "compile+graph":
            g, S, row, z = self._graph(n_paths)
            self._initial_state(n_paths, S)
            for k in range(self.pm.n_steps):
                torch.randn(shape, generator=gen, device=self.device, dtype=self.dtype, out=z)
                row.copy_(self.rows[k])
                g.replay()
            S = S.clone()  # the graph's buffers are reused by the next call
        else:
            S = self._initial_state(n_paths)
            for k in range(self.pm.n_steps):
                z = torch.randn(shape, generator=gen, device=self.device, dtype=self.dtype)
                self.step(S, self.rows[k], z, *self.consts)
        x, F, GM, GD, U, D, l, pvd, pvf = S.unbind(0)
        pv_mat = D * l * torch.clamp(GM - F, min=0.0)
        return pv_mat, pvd, pvf, F, D, l, U
