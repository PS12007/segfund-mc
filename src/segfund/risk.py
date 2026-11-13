"""Risk metrics: CTE tail measures and sensitivities (delta, rho).

CTE (conditional tail expectation)
----------------------------------
CTE(alpha) = average loss in the worst (1 - alpha) share of scenarios.
CTE(95) is the mean of the worst 5%. Losses here are each path's present value
of guarantee claims minus fee income (pv_net), discounted along the path's own
short rate, under **real-world** scenarios.

This is a simplification of what Canadian capital rules use. Those apply CTE to
a "greatest present value of accumulated deficiency" with prescribed scenario
calibration; here the loss is just the PV of net cash flows. Listed in Limits.

Estimator: sort the N losses, take the k = round(N (1 - alpha)) largest, average.
VaR(alpha) is the k-th largest. Standard error (Manistre & Hancock, "Variance of
the CTE estimator", NAAJ 2005):

    Var(CTE_hat) ~ [ Var(L | L >= VaR) + alpha (CTE - VaR)^2 ] / (N (1 - alpha))

The second term is the extra uncertainty from not knowing exactly where the
tail starts. phase 4 checks this formula against the spread of CTE estimates
over independent replications (scripts/risk_study.py).

Sensitivities
-------------
Bump-and-revalue with **common random numbers** (CRN): price the contract at
S0 (1 +/- h) or with the curve shifted +/- s, using the *same* Philox stream
(same seed, same path ids). The difference is then almost free of Monte Carlo
noise, because both prices see the same scenarios. The SE of the difference is
computed from the per-path (or per-antithetic-pair) differences.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from segfund.config import Model, bumped


# ---------------------------------------------------------------------------------
# CTE
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class CTE:
    alpha: float
    cte: float
    var: float  # value at risk at the same level (the tail threshold)
    se: float
    k: int  # number of tail scenarios averaged
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def cte(losses, alpha: float) -> CTE:
    L = np.asarray(losses, dtype=np.float64)
    n = L.size
    k = int(round(n * (1.0 - alpha)))
    if not 0 < alpha < 1 or k < 2:
        raise ValueError("need 0 < alpha < 1 and at least 2 tail scenarios")
    tail = np.partition(L, n - k)[n - k:]  # the k largest, unordered
    c = float(tail.mean())
    v = float(tail.min())
    var_tail = float(tail.var(ddof=1))
    se = math.sqrt((var_tail + alpha * (c - v) ** 2) / (n * (1.0 - alpha)))
    return CTE(alpha, c, v, se, k, n)


def cte_gbm_put_closed_form(S0, K, T, r, mer, sigma, erp, alpha):
    """CTE(alpha) and VaR(alpha) of L = e^{-rT} max(K - F_T, 0) with real-world lognormal F_T.

    ln F_T ~ N(m, s^2), m = ln S0 + (r + erp - mer - sigma^2/2) T, s = sigma sqrt(T).
    The loss is large exactly when F_T is small, so the worst (1-alpha) share of
    losses is the lowest (1-alpha) share of F_T. With b = 1 - alpha:
      * if P(F_T < K) >= b: q = b-quantile of F_T, VaR = e^{-rT}(K - q),
        CTE = e^{-rT} (K - E[F_T; F_T <= q] / b)
      * otherwise the tail also contains zero losses: VaR = 0,
        CTE = e^{-rT} (K P(F_T < K) - E[F_T; F_T < K]) / b
    with the lognormal partial expectation E[F; F <= x] = e^{m + s^2/2} N((ln x - m - s^2)/s).
    """
    from statistics import NormalDist

    N = NormalDist()
    m = math.log(S0) + (r + erp - mer - 0.5 * sigma**2) * T
    s = sigma * math.sqrt(T)
    b = 1.0 - alpha
    disc = math.exp(-r * T)

    def partial(x):
        return math.exp(m + 0.5 * s * s) * N.cdf((math.log(x) - m - s * s) / s)

    p_itm = N.cdf((math.log(K) - m) / s)
    if p_itm >= b:
        q = math.exp(m + s * N.inv_cdf(b))
        return disc * (K - partial(q) / b), disc * (K - q)
    return disc * (K * p_itm - partial(K)) / b, 0.0


# ---------------------------------------------------------------------------------
# Sensitivities by common random numbers
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sensitivity:
    name: str
    bump: float
    value: float  # derivative estimate in the stated units
    se: float
    units: str
    pv_up: float
    pv_down: float

    def as_dict(self) -> dict:
        return asdict(self)


def _paired(diff: np.ndarray, antithetic: bool) -> np.ndarray:
    if antithetic:
        h = diff.size // 2
        return 0.5 * (diff[:h] + diff[h:])
    return diff


def _central(sim, model_up: Model, model_dn: Model, n: int, seed: int, antithetic: bool):
    up = sim(model_up, n, seed, antithetic).pv_net
    dn = sim(model_dn, n, seed, antithetic).pv_net
    d = _paired(up - dn, antithetic)
    return float(up.mean()), float(dn.mean()), float(d.mean()), float(d.std(ddof=1) / math.sqrt(d.size))


def delta(sim, model: Model, n: int, seed: int, h: float = 0.01, antithetic: bool = False) -> Sensitivity:
    """dPV/dS0 per dollar of fund value: (PV(S0(1+h)) - PV(S0(1-h))) / (2 h S0), same scenarios.

    Negative for a put-like liability: when markets rise the guarantee is worth less.
    """
    S0 = model.contract.initial_deposit * model.fund_shock
    pu, pd, d, se = _central(sim, bumped(model, fund_shock=1 + h), bumped(model, fund_shock=1 - h), n, seed, antithetic)
    return Sensitivity("delta", h, d / (2 * h * S0), se / (2 * h * S0), "PV dollars per dollar of fund value", pu, pd)


def rho(sim, model: Model, n: int, seed: int, shift: float = 0.001, antithetic: bool = False) -> Sensitivity:
    """dPV per +1bp parallel shift of the initial zero curve, same scenarios."""
    pu, pd, d, se = _central(sim, bumped(model, rate_shift=shift), bumped(model, rate_shift=-shift), n, seed, antithetic)
    scale = 1e-4 / (2 * shift)
    return Sensitivity("rho", shift, d * scale, se * scale, "PV dollars per +1bp parallel shift", pu, pd)


def independent_delta_se(sim, model: Model, n: int, seed: int, h: float = 0.01) -> float:
    """Same delta, but up and down priced with *different* seeds: shows what CRN buys."""
    S0 = model.contract.initial_deposit * model.fund_shock
    up = sim(bumped(model, fund_shock=1 + h), n, seed, False).pv_net
    dn = sim(bumped(model, fund_shock=1 - h), n, seed + 7919, False).pv_net
    se = math.sqrt(up.var(ddof=1) / n + dn.var(ddof=1) / n)
    return se / (2 * h * S0)


def bs_delta_rho(S0, K, T, r, q, sigma):
    """Black-Scholes put delta (per dollar of spot) and rho (per +1bp) with dividend yield q."""
    from statistics import NormalDist

    N = NormalDist()
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    dlt = -math.exp(-q * T) * N.cdf(-d1)
    rh = -K * T * math.exp(-r * T) * N.cdf(-d2) * 1e-4
    return dlt, rh


# ---------------------------------------------------------------------------------
# Simulators with a common signature: sim(model, n, seed, antithetic) -> PathResults
# ---------------------------------------------------------------------------------


def cpu_sim(model: Model, n: int, seed: int, antithetic: bool = False):
    from segfund.reference.engine import simulate
    from segfund.rng.philox import PhiloxNormals

    return simulate(model, n, PhiloxNormals(seed, antithetic_half=n // 2 if antithetic else 0))


def gpu_sim(model: Model, n: int, seed: int, antithetic: bool = False):
    from segfund.gpu.kernels import FusedEngine

    return FusedEngine(model).simulate(n, seed, antithetic=antithetic)
