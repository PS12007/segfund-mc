"""Monte Carlo estimators: plain, antithetic, control variate, and both.

Antithetic variates
-------------------
Paths g and g + N/2 use the same normals with opposite signs (rng.philox.antithetic_counter).
The two halves of a pair are *not* independent, so the standard error must be
computed from the N/2 pair averages, never from the N individual paths. Treating
them as independent would understate the SE whenever the pair is positively
correlated and overstate it when (as intended) negatively correlated.

Control variate
---------------
X = D(T) * max(K - F0 * unit(T), 0): the discounted maturity put on the first
deposit, with K = gmmb_pct * F0. Its exact mean under the risk-neutral measure is
the Hull-White + GBM put (reference.closed_form.hw_gbm_put, validated in V6).
The estimator is

    Y_cv = Y - beta (X - E[X]),   beta = Cov(Y, X) / Var(X)

beta is estimated from the same paths. That adds an O(1/N) bias, negligible at
the path counts used here (phase 3 checks it against closed forms).

Why this control and not the constant-rate Black-Scholes put suggested in the
brief: the BS put's mean is only known when rates are constant. Under Hull-White
rates its expectation is not the BS value, so using it would bias the estimator.

Only valid under the risk-neutral measure (under real-world, E[X] is not the
closed-form price).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from segfund.config import Model
from segfund.reference import closed_form as cf
from segfund.reference.engine import PathResults


@dataclass(frozen=True)
class Estimate:
    mean: float
    se: float
    n_paths: int
    n_independent: int  # pairs when antithetic, paths otherwise
    antithetic: bool
    control: str
    beta: float | None = None
    corr_yx: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def control_payoff(res: PathResults, model: Model) -> tuple[np.ndarray, float]:
    """Per-path control X and its exact mean E[X]."""
    if model.measure != "risk_neutral":
        raise ValueError("the closed-form control mean is only valid under the risk-neutral measure")
    c = model.contract
    K = c.gmmb_pct * c.initial_deposit
    S0 = c.initial_deposit * model.fund_shock  # fund value just after the t=0 deposit
    X = res.discount_T * np.maximum(K - S0 * res.unit_T, 0.0)
    EX = float(cf.hw_gbm_put(model.hw, model.equity, S0, K, c.maturity_years, c.mer))
    return X, EX


def _pairs(v: np.ndarray) -> np.ndarray:
    h = v.size // 2
    if v.size % 2:
        raise ValueError("antithetic estimate needs an even number of paths")
    return 0.5 * (v[:h] + v[h:])


def estimate(
    res: PathResults,
    model: Model,
    field: str = "pv_net",
    antithetic: bool | None = None,
    control: str | None = None,
) -> Estimate:
    antithetic = model.sim.antithetic if antithetic is None else antithetic
    control = model.sim.control_variate if control is None else control
    y = np.asarray(getattr(res, field), dtype=np.float64)
    n = y.size
    x = ex = None
    if control == "hw_put":
        x, ex = control_payoff(res, model)
    elif control != "none":
        raise ValueError(f"unknown control variate {control!r} (none | hw_put)")
    if antithetic:
        y = _pairs(y)
        if x is not None:
            x = _pairs(x)
    beta = corr = None
    if x is not None:
        cov = np.cov(y, x, ddof=1)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0
        corr = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])) if cov[0, 0] > 0 and cov[1, 1] > 0 else 0.0
        y = y - beta * (x - ex)
    m = float(y.mean())
    se = float(y.std(ddof=1) / np.sqrt(y.size))
    return Estimate(m, se, n, y.size, bool(antithetic), control, beta, corr)
