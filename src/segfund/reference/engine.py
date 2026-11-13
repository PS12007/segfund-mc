"""NumPy fp64 reference simulator. This is the ground truth the GPU code must match.

Structure deliberately mirrors what a fused GPU kernel does: one loop over time
steps, a small per-path state (x, integrated rate, fund, guarantees, in-force),
and cash flows accumulated as we go. Nothing per-step is stored unless asked.

Per step k, from s = t_k to t = t_{k+1}:
  1. at s: add deposits scheduled at s; apply a guarantee reset if s is a reset date
  2. draw z ~ N(0, I_3) per path; eps = L_k z
  3. x_t = x_s e^{-ah} + shift + eps1;  int r = x_s B(h) + shift + eps2 + int phi
  4. F_t = F_s exp(int r + lnF_drift + eps3);  D_t = D_s exp(-int r)
  5. decrements over the step: deaths q_k (age-based), lapses from moneyness at s
  6. cash flows at t: fees, death claims; at maturity, the maturity claim
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segfund.config import Model
from segfund.models.equity import StepTable, build_step_table
from segfund.rng.philox import PhiloxNormals


class NumpyNormals:
    """Sequential standard normals from NumPy's Philox bit generator.

    Used in phase 1 before the counter-based stream existed; kept so phase 1
    results can be reproduced. The default RNG is now rng.philox.PhiloxNormals,
    which the GPU kernels reproduce bit for bit.
    """

    def __init__(self, seed: int):
        self._g = np.random.Generator(np.random.Philox(seed))

    def normals(self, step: int, n_paths: int) -> np.ndarray:
        return self._g.standard_normal((3, n_paths))


@dataclass
class PathResults:
    pv_net: np.ndarray  # PV(claims) - PV(fees), per path: the insurer's liability
    pv_maturity_claims: np.ndarray
    pv_death_claims: np.ndarray
    pv_fees: np.ndarray
    fund_T: np.ndarray
    discount_T: np.ndarray  # exp(-int_0^T r)
    inforce_T: np.ndarray
    # Value at T of 1 dollar invested at t=0 in the fund (same returns and MER as F,
    # but no later deposits). Underlying of the control variate.
    unit_T: np.ndarray

    @property
    def n_paths(self) -> int:
        return self.pv_net.size


def mean_and_se(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def _reset_steps(model: Model, times: np.ndarray) -> np.ndarray:
    r = model.contract.resets
    flags = np.zeros(times.size - 1, dtype=bool)
    if not r.enabled:
        return flags
    T = model.contract.maturity_years
    for k in range(1, times.size - 1):
        t = times[k]
        n_ann = t / r.every_years
        if abs(n_ann - round(n_ann)) < 1e-9 and T - t >= r.min_years_to_maturity - 1e-12:
            flags[k] = True
    return flags


def _deposit_steps(model: Model, times: np.ndarray) -> np.ndarray:
    amounts = np.zeros(times.size - 1)
    for d in model.contract.deposits:
        k = int(round(d.t / model.sim.dt))
        if abs(times[k] - d.t) > 1e-9:
            raise ValueError(f"deposit at t={d.t} is not on the time grid (dt={model.sim.dt})")
        amounts[k] += d.amount
    return amounts


@dataclass(frozen=True)
class Schedule:
    """Everything deterministic per step: market table + decrement/deposit/reset schedule.

    Shared by the CPU reference and the GPU engines so they apply the same model.
    """

    tab: StepTable
    q_death: np.ndarray
    deposits: np.ndarray  # added to the guarantee bases (times gmmb/gmdb pct)
    fund_adds: np.ndarray  # added to the fund: deposits, with the t=0 one times model.fund_shock
    resets: np.ndarray


def build_schedule(model: Model) -> Schedule:
    c = model.contract
    times = model.time_grid()
    tab = build_step_table(model.hw, model.equity, times, c.mer, model.lambda_r)
    q_death = model.mortality.step_death_probs(c.issue_age, times) if model.mortality else np.zeros(tab.n_steps)
    deposits = _deposit_steps(model, times)
    fund_adds = deposits.copy()
    fund_adds[0] *= model.fund_shock
    return Schedule(tab, q_death, deposits, fund_adds, _reset_steps(model, times))


def simulate(model: Model, n_paths: int | None = None, rng=None, pathwise_delta: bool = False) -> PathResults:
    """Simulate the contract. With pathwise_delta=True, also returns (results, dPV/dS0 per path).

    Pathwise delta differentiates each path's PV with respect to the fund value S0
    just after the t=0 deposit, holding the guarantee fixed. It is only exact when
    every path's cash flows depend on S0 through F alone: single deposit, no
    resets, no dynamic lapse (all checked). Then F_t = S0 * unit_t, so
    d/dS0 max(G - F_t, 0) = -1{G > F_t} unit_t and d/dS0 (fee * F_t) = fee * unit_t.
    """
    c = model.contract
    n = n_paths or model.sim.n_paths
    if rng is None:
        half = 0
        if model.sim.antithetic:
            if n % 2:
                raise ValueError("antithetic sampling needs an even number of paths")
            half = n // 2
        rng = PhiloxNormals(model.sim.seed, antithetic_half=half)
    if model.sim.precision != "fp64":
        raise ValueError("the NumPy reference engine is fp64 only")

    sch = build_schedule(model)
    tab, q_death, deposits, resets = sch.tab, sch.q_death, sch.deposits, sch.resets
    if pathwise_delta and (len(c.deposits) > 1 or resets.any() or model.lapse.enabled):
        raise ValueError("pathwise delta needs a single deposit, no resets and no dynamic lapse")
    dpv = np.zeros(n) if pathwise_delta else None

    x = np.zeros(n)
    U = np.ones(n)
    D = np.ones(n)
    F = np.zeros(n)
    G_M = np.zeros(n)
    G_D = np.zeros(n)
    l = np.ones(n)
    pv_mat = np.zeros(n)
    pv_death = np.zeros(n)
    pv_fee = np.zeros(n)

    for k in range(tab.n_steps):
        # 1. events at the start of the step
        if deposits[k] > 0:
            F += sch.fund_adds[k]
            G_M += c.gmmb_pct * deposits[k]
            G_D += c.gmdb_pct * deposits[k]
        if resets[k]:
            np.maximum(G_M, c.gmmb_pct * F, out=G_M)
            np.maximum(G_D, c.gmdb_pct * F, out=G_D)
        q_lapse = model.lapse.step_prob(F, G_M, tab.h[k])

        # 2-4. market step (exact Gaussian transition)
        z = rng.normals(k, n)
        eps = tab.L[k] @ z
        int_r = x * tab.Bh[k] + tab.y_shift[k] + eps[1] + tab.phi_int[k]
        x = x * tab.decay[k] + tab.x_shift[k] + eps[0]
        growth = np.exp(int_r + tab.lnF_drift[k] + eps[2])
        F = F * growth
        U = U * growth
        D = D * np.exp(-int_r)

        # 5-6. decrements and cash flows at the end of the step
        deaths = l * q_death[k]
        pv_death += D * deaths * np.maximum(G_D - F, 0.0)
        pv_fee += D * l * c.guarantee_fee * tab.h[k] * F
        if dpv is not None:
            dpv += -D * deaths * (G_D > F) * U - D * l * c.guarantee_fee * tab.h[k] * U
        l = l * (1.0 - q_death[k]) * (1.0 - q_lapse)

    pv_mat += D * l * np.maximum(G_M - F, 0.0)
    res = PathResults(pv_mat + pv_death - pv_fee, pv_mat, pv_death, pv_fee, F, D, l, U)
    if dpv is not None:
        dpv += -D * l * (G_M > F) * U
        return res, dpv
    return res


def simulate_rates(
    model: Model,
    n_paths: int,
    record_steps,
    rng=None,
    t0: float = 0.0,
    x0: float = 0.0,
    horizon: float | None = None,
):
    """Rates only (x and integrated r), optionally starting at time t0 with x(t0) = x0.

    Returns {step: (x, int_{t0}^{t_step} r)} for each requested step. Used by the
    bond-price validations and, later, by nested simulation.
    """
    rng = rng or PhiloxNormals(model.sim.seed)
    T = horizon if horizon is not None else model.contract.maturity_years
    nsteps = int(round((T - t0) / model.sim.dt))
    times = t0 + np.arange(nsteps + 1) * model.sim.dt
    tab = build_step_table(model.hw, model.equity, times, model.contract.mer, model.lambda_r)
    want = set(record_steps)
    x = np.full(n_paths, float(x0))
    I = np.zeros(n_paths)
    out = {}
    for k in range(tab.n_steps):
        z = rng.normals(k, n_paths)
        eps = tab.L[k] @ z
        I = I + x * tab.Bh[k] + tab.y_shift[k] + eps[1] + tab.phi_int[k]
        x = x * tab.decay[k] + tab.x_shift[k] + eps[0]
        if k + 1 in want:
            out[k + 1] = (x.copy(), I.copy())
    return times, out
