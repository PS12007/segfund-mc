"""Nested simulation and least-squares Monte Carlo (LSMC).

The problem
-----------
Capital is about what the guarantee could be worth at a *future* date t0
(here 1 year) in a bad scenario. That needs two layers:

  outer: real-world scenarios from 0 to t0 (what might happen to markets)
  inner: in each outer scenario, the risk-neutral VALUE of the remaining
         guarantee at t0, which itself is a Monte Carlo over t0..T

Loss of outer scenario i, as at time 0 (discounted along the outer path):

    L_i = PV_[0,t0] of claims - fees on the outer path + D_i(0,t0) * V_i(t0)

and the capital metric is CTE(95) of L over outer scenarios. Brute force costs
N_outer x N_inner paths, which is why insurers use proxies.

LSMC proxy
----------
Simulate many outer scenarios with only a couple of inner paths each (a very
noisy but unbiased estimate of V_i), then regress those noisy values on
functions ("basis functions") of the outer state. Regression averages the noise
across scenarios with similar states. The fitted function is the proxy V(state).

State at t0: x (rate factor), F (fund), GM and GD (guarantees), l (in-force).
The value is linear in l (in-force just scales every future cash flow), so we
regress V / l on functions of moneyness m = ln(F / GM) and x, then multiply by l.

Inner value estimates
---------------------
Antithetic pairs inside each outer scenario. Optionally a control variate: the
discounted maturity put on F(t0) * unit(T) with strike GM, whose value at t0 is
known in closed form given (x(t0), F(t0)) (closed_form.hw_gbm_put_at). Its beta
is pooled across all outer scenarios, so its estimation bias is O(1/(N_out N_in)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from segfund.config import Model
from segfund.reference import closed_form as cf
from segfund.rng import philox as P

STATE_ROWS = ("x", "F", "GM", "GD", "l")


@dataclass
class OuterScenarios:
    k0: int  # time step index of the valuation date t0
    t0: float
    x: np.ndarray
    F: np.ndarray
    GM: np.ndarray
    GD: np.ndarray
    l: np.ndarray
    D: np.ndarray  # discount factor 0 -> t0 along the (real-world) outer path
    pv_early: np.ndarray  # PV at 0 of death claims - fees paid in (0, t0]

    @property
    def n(self) -> int:
        return self.x.size

    def state_matrix(self) -> np.ndarray:
        return np.stack([getattr(self, k) for k in STATE_ROWS])

    def subset(self, idx) -> "OuterScenarios":
        return OuterScenarios(self.k0, self.t0, *(getattr(self, k)[idx] for k in
                                                 ("x", "F", "GM", "GD", "l", "D", "pv_early")))


def valuation_step(model: Model, t0: float) -> int:
    k0 = int(round(t0 / model.sim.dt))
    if abs(k0 * model.sim.dt - t0) > 1e-9 or not 0 < k0 < model.time_grid().size - 1:
        raise ValueError("t0 must be a grid point strictly inside the contract term")
    return k0


def simulate_outer(model_rw: Model, n: int, seed: int, t0: float = 1.0, engine: str = "gpu") -> OuterScenarios:
    """Real-world outer scenarios from 0 to t0 (stream 0)."""
    k0 = valuation_step(model_rw, t0)
    if engine == "gpu":
        from segfund.gpu.kernels import FusedEngine, results_from_buffer

        eng = FusedEngine(model_rw)
        out = eng.alloc(n, out_state=True)
        eng.launch(out, n, seed, k_start=0, k_end=k0, out_state=True)
        r = results_from_buffer(out, n)
    else:
        from segfund.reference.engine import simulate
        from segfund.rng.philox import PhiloxNormals

        r = simulate(model_rw, n, PhiloxNormals(seed), k_start=0, k_end=k0)
    return OuterScenarios(k0, t0, r.x_end, r.fund_T, r.gm_end, r.gd_end, r.inforce_T, r.discount_T,
                          r.pv_death_claims - r.pv_fees)


@dataclass
class InnerValues:
    v: np.ndarray  # value at t0 per outer scenario (per policy at issue; includes in-force l)
    se: np.ndarray  # standard error of each v
    n_inner: int
    antithetic: bool
    control: bool
    beta: float | None
    paths: int  # total inner paths simulated


def _per_outer_moments(y, x, n_inner, antithetic, xp):
    """Per-outer means / variances / covariance of y (and x) over independent units (pairs if antithetic)."""
    m = y.shape[0] // n_inner
    y = y.reshape(m, n_inner)
    x = x.reshape(m, n_inner) if x is not None else None
    if antithetic:
        h = n_inner // 2
        y = 0.5 * (y[:, :h] + y[:, h:])
        x = 0.5 * (x[:, :h] + x[:, h:]) if x is not None else None
    u = y.shape[1]
    my = y.mean(1)
    dy = y - my[:, None]
    out = {"my": my, "vy": (dy * dy).sum(1) / (u - 1), "units": u}
    if x is not None:
        mx = x.mean(1)
        dx = x - mx[:, None]
        out.update(mx=mx, vx=(dx * dx).sum(1) / (u - 1), cxy=(dx * dy).sum(1) / (u - 1))
    if xp is not np:
        out = {k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in out.items()}
    return out


def inner_values(
    model_rn: Model,
    outer: OuterScenarios,
    n_inner: int,
    seed: int,
    antithetic: bool = True,
    control: bool = True,
    engine: str = "gpu",
    chunk_paths: int = 1 << 22,
) -> InnerValues:
    """Risk-neutral value at t0 of the remaining contract, for every outer scenario."""
    if antithetic and n_inner % 2:
        raise ValueError("antithetic inner sampling needs an even n_inner")
    units = n_inner // 2 if antithetic else n_inner
    if units < 2:
        raise ValueError("need at least 2 independent inner units per outer scenario for an SE")
    c = model_rn.contract
    T, k0, t0 = c.maturity_years, outer.k0, outer.t0
    n_out = outer.n
    per_chunk = max(1, chunk_paths // n_inner)
    S = outer.state_matrix()
    keys = ("my", "vy", "mx", "vx", "cxy") if control else ("my", "vy")
    acc = {k: np.empty(n_out) for k in keys}

    if engine == "gpu":
        import torch

        from segfund.gpu.kernels import N_OUT, FusedEngine

        eng = FusedEngine(model_rn)
    for s in range(0, n_out, per_chunk):
        e = min(n_out, s + per_chunk)
        m = e - s
        n = m * n_inner
        half = n_inner // 2 if antithetic else 0
        if engine == "gpu":
            st = torch.tensor(S[:, s:e], dtype=torch.float32, device="cuda").contiguous()
            out = eng.alloc(n)
            eng.launch(out, n, seed, path_offset=s * n_inner, stream=P.STREAM_INNER, antithetic_half=half,
                       k_start=k0, state=st, n_inner=n_inner)
            o = out.view(N_OUT, n).double()
            y = o[0] + o[1] - o[2]
            x = None
            if control:
                Fi = torch.tensor(S[1, s:e], dtype=torch.float64, device="cuda").repeat_interleave(n_inner)
                Ki = torch.tensor(S[2, s:e], dtype=torch.float64, device="cuda").repeat_interleave(n_inner)
                x = o[4] * torch.clamp(Ki - Fi * o[6], min=0.0)
            mom = _per_outer_moments(y, x, n_inner, antithetic, torch)
            del out, o, y, x
        else:
            from segfund.reference.engine import simulate
            from segfund.rng.philox import PhiloxNormals

            init = {k: np.repeat(S[i, s:e], n_inner) for i, k in enumerate(STATE_ROWS)}
            r = simulate(model_rn, n, PhiloxNormals(seed, path_offset=s * n_inner, stream=P.STREAM_INNER,
                                                    antithetic_half=half), k_start=k0, init_state=init)
            x = r.discount_T * np.maximum(init["GM"] - init["F"] * r.unit_T, 0.0) if control else None
            mom = _per_outer_moments(r.pv_net, x, n_inner, antithetic, np)
        for k in keys:
            acc[k][s:e] = mom[k]

    beta = None
    if control:
        ex = cf.hw_gbm_put_at(model_rn.hw, model_rn.equity, t0, outer.x, outer.F, outer.GM, T, c.mer)
        beta = float(acc["cxy"].sum() / acc["vx"].sum())
        v = acc["my"] - beta * (acc["mx"] - ex)
        var_unit = acc["vy"] - 2 * beta * acc["cxy"] + beta**2 * acc["vx"]
    else:
        v = acc["my"]
        var_unit = acc["vy"]
    se = np.sqrt(np.maximum(var_unit, 0.0) / units)
    return InnerValues(v, se, n_inner, antithetic, control, beta, n_out * n_inner)


def losses(outer: OuterScenarios, v: np.ndarray) -> np.ndarray:
    """Loss as at time 0 per outer scenario: early cash flows + discounted value at t0."""
    return outer.pv_early + outer.D * v


# ------------------------------------------------------------------------------------
# LSMC
# ------------------------------------------------------------------------------------

BASES = ("poly", "poly+put")


def basis(model_rn: Model, outer: OuterScenarios, kind: str) -> np.ndarray:
    """Basis functions of the state at t0 (for V / l).

    poly:     1, m, m^2, m^3, m^4, x, x^2, m x, m^2 x       with m = ln(F / GM), x scaled by 100
    poly+put: poly plus the closed-form t0 values of the maturity put (strike GM)
              and of a death-benefit proxy put (strike GD), each per unit in force
    """
    if kind not in BASES:
        raise ValueError(f"basis must be one of {BASES}")
    m = np.log(outer.F / outer.GM)
    x = 100.0 * outer.x  # rates in percent-ish units keep the columns comparable
    cols = [np.ones_like(m), m, m**2, m**3, m**4, x, x**2, m * x, m**2 * x]
    if kind == "poly+put":
        c = model_rn.contract
        args = (model_rn.hw, model_rn.equity, outer.t0, outer.x, outer.F)
        cols.append(cf.hw_gbm_put_at(*args, outer.GM, c.maturity_years, c.mer) / 1e3)
        cols.append(cf.hw_gbm_put_at(*args, outer.GD, c.maturity_years, c.mer) / 1e3)
    return np.column_stack(cols)


@dataclass
class LSMCProxy:
    kind: str
    coef: np.ndarray
    n_fit: int
    fit_paths: int
    r2_noisy: float  # R^2 against the noisy fit targets (low is expected; the targets are mostly noise)

    def predict(self, model_rn: Model, outer: OuterScenarios) -> np.ndarray:
        return outer.l * (basis(model_rn, outer, self.kind) @ self.coef)


def lsmc_fit(model_rn: Model, outer_fit: OuterScenarios, y_fit: np.ndarray, kind: str, fit_paths: int) -> LSMCProxy:
    A = basis(model_rn, outer_fit, kind)
    t = y_fit / outer_fit.l
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    resid = t - A @ coef
    r2 = 1.0 - float(resid.var() / t.var())
    return LSMCProxy(kind, coef, outer_fit.n, fit_paths, r2)


def compare(v_proxy: np.ndarray, gt: InnerValues, outer: OuterScenarios, alphas=(0.70, 0.95)) -> dict:
    """Accuracy of a value estimate against the ground truth on the same outer scenarios."""
    from segfund.risk import cte

    d = v_proxy - gt.v
    noise_floor = float(np.sqrt(np.mean(gt.se**2)))
    L_gt = losses(outer, gt.v)
    L_px = losses(outer, v_proxy)
    out = dict(
        rmse=float(np.sqrt(np.mean(d**2))),
        # subtract the ground truth's own noise from the RMSE (independent errors add in variance)
        rmse_net_of_gt_noise=float(math.sqrt(max(np.mean(d**2) - noise_floor**2, 0.0))),
        gt_noise_rms=noise_floor,
        mean_error=float(d.mean()),
        max_abs_error=float(np.abs(d).max()),
    )
    for a in alphas:
        cg, cp = cte(L_gt, a), cte(L_px, a)
        out[f"cte{int(a * 100)}_gt"] = cg.cte
        out[f"cte{int(a * 100)}_proxy"] = cp.cte
        out[f"cte{int(a * 100)}_error"] = cp.cte - cg.cte
        out[f"cte{int(a * 100)}_outer_se"] = cg.se  # outer sampling error, common to both
    return out
