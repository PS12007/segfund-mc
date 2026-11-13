"""Pack the per-step schedule into one dense table for GPU kernels.

One row per time step, NCOL float columns. The kernel loads a row per step; every
path in a block reads the same row, so it is served from cache, not DRAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segfund.config import Model
from segfund.reference.engine import build_schedule

COLS = (
    "decay", "Bh", "phi_int", "x_shift", "y_shift", "lnF_drift",
    "L00", "L10", "L11", "L20", "L21", "L22",
    "h", "q_death", "deposit", "reset", "fund_add",
)
NCOL = len(COLS)
C = {name: i for i, name in enumerate(COLS)}


@dataclass(frozen=True)
class PackedModel:
    table: np.ndarray  # (n_steps, NCOL) float64; cast to fp32 on upload
    gmmb_pct: float
    gmdb_pct: float
    guarantee_fee: float
    lapse_hazard: float  # 0 when lapse is disabled
    lapse_k: float
    lapse_lo: float
    lapse_hi: float

    @property
    def n_steps(self) -> int:
        return self.table.shape[0]


def pack(model: Model) -> PackedModel:
    s = build_schedule(model)
    t = s.tab
    n = t.n_steps
    tab = np.zeros((n, NCOL))
    tab[:, C["decay"]] = t.decay
    tab[:, C["Bh"]] = t.Bh
    tab[:, C["phi_int"]] = t.phi_int
    tab[:, C["x_shift"]] = t.x_shift
    tab[:, C["y_shift"]] = t.y_shift
    tab[:, C["lnF_drift"]] = t.lnF_drift
    for name, (i, j) in {"L00": (0, 0), "L10": (1, 0), "L11": (1, 1), "L20": (2, 0), "L21": (2, 1), "L22": (2, 2)}.items():
        tab[:, C[name]] = t.L[:, i, j]
    if np.any(t.L[:, 0, 1:]) or np.any(t.L[:, 1, 2]):
        raise AssertionError("Cholesky factor must be lower triangular")
    tab[:, C["h"]] = t.h
    tab[:, C["q_death"]] = s.q_death
    tab[:, C["deposit"]] = s.deposits
    tab[:, C["reset"]] = s.resets.astype(np.float64)
    tab[:, C["fund_add"]] = s.fund_adds
    c, lp = model.contract, model.lapse
    return PackedModel(
        table=tab,
        gmmb_pct=c.gmmb_pct,
        gmdb_pct=c.gmdb_pct,
        guarantee_fee=c.guarantee_fee,
        lapse_hazard=lp.base_hazard if lp.enabled else 0.0,
        lapse_k=lp.k,
        lapse_lo=lp.lo,
        lapse_hi=lp.hi,
    )
