"""Closed-form prices used as validation targets (fp64)."""

from __future__ import annotations

import math

import numpy as np

from segfund.models.equity import EquityParams
from segfund.models.hull_white import HullWhite, B, var_int_x

_erfc = np.vectorize(math.erfc, otypes=[np.float64])


def norm_cdf(x):
    return 0.5 * _erfc(-np.asarray(x, dtype=np.float64) / math.sqrt(2.0))


def black_put(P0T: float, fwd, K, total_var):
    """Black's formula for a put: P0T * [K N(-d2) - fwd N(-d1)], total_var = vol^2 * T."""
    fwd = np.asarray(fwd, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if np.any(total_var <= 0):
        return P0T * np.maximum(K - fwd, 0.0)
    v = np.sqrt(total_var)
    d1 = (np.log(fwd / K) + 0.5 * total_var) / v
    d2 = d1 - v
    return P0T * (K * norm_cdf(-d2) - fwd * norm_cdf(-d1))


def bs_put(S0, K, T, r, q, sigma):
    """Black-Scholes European put with continuous dividend yield q."""
    P = math.exp(-r * T)
    fwd = S0 * math.exp((r - q) * T)
    return black_put(P, fwd, K, sigma**2 * T)


def hw_gbm_put(hw: HullWhite, eq: EquityParams, F0: float, K, T: float, mer: float):
    """European put on the fund under Hull-White rates + correlated GBM.

    ln F_T is Gaussian. Under the T-forward measure its mean gives the forward
    F0 e^{-mer T} / P(0,T) and its variance is the same as under Q:

        v^2 = sigma_S^2 T + V(0,T) + 2 rho sigma sigma_S (T - B(T)) / a

    where V(0,T) = Var(int_0^T x). Then the put is Black's formula with v^2.
    With sigma_HW = 0 this reduces to Black-Scholes.
    """
    a, s = hw.a, hw.sigma
    P = float(hw.curve.discount(T))
    fwd = F0 * math.exp(-mer * T) / P
    v2 = eq.sigma**2 * T + float(var_int_x(a, s, T)) + 2 * eq.rho * s * eq.sigma * (T - float(B(a, T))) / a
    return black_put(P, fwd, K, v2)


def gmdb_gmmb_constant_rate(
    F0: float, G_D: float, G_M: float, T: float, r: float, mer: float, sigma: float, q_steps, times
):
    """Constant rates, deterministic mortality, no lapse, no fee, no resets.

    PV = sum_k P(alive at t_k) q_k put(t_{k+1}, G_D) + P(alive at T) put(T, G_M).
    Exact for the discrete-time contract the simulator implements, because the
    decrements are independent of the fund.
    """
    q_steps = np.asarray(q_steps, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    alive = np.concatenate([[1.0], np.cumprod(1.0 - q_steps)])
    pv = 0.0
    for k in range(q_steps.size):
        t = times[k + 1]
        pv += alive[k] * q_steps[k] * float(bs_put(F0, G_D, t, r, mer, sigma))
    pv += alive[-1] * float(bs_put(F0, G_M, T, r, mer, sigma))
    return pv
