"""Equity fund under Hull-White rates, and the joint one-step Gaussian table.

Fund value F (after fees):

    dF/F = (r(t) - mer + erp) dt + sigma_S dW_S,     d<W_S, W_r> = rho dt

  * risk-neutral measure: erp = 0, so the discounted fund grows at -mer
    (the MER acts like a dividend yield paid away from the policyholder).
  * real-world measure: erp = equity risk premium > 0, and the rate factor gets
    a constant market price of risk lambda_r (dx gains drift sigma * lambda_r).

Over one step [s, t], h = t - s, given x_s the vector

    eps = (eps1, eps2, eps3) = (x_t - E x_t, Y - E Y, sigma_S (W_S(t) - W_S(s)))

is Gaussian with mean 0 and covariance C (below). We draw eps = L z with
z ~ N(0, I_3) and L the Cholesky factor of C. This is the "Cholesky correlation"
step: it turns 3 independent normals into correlated rate and equity shocks.

    C11 = sigma^2 (1 - e^{-2ah}) / (2a)
    C22 = sigma^2/a^2 [h - 2B(h) + (1 - e^{-2ah})/(2a)]
    C12 = sigma^2/(2a^2) (1 - e^{-ah})^2
    C13 = rho sigma sigma_S B(h)               (Cov of x_t with W_S increment)
    C23 = rho sigma sigma_S (h - B(h)) / a     (Cov of int x with W_S increment)
    C33 = sigma_S^2 h
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segfund.models.hull_white import HullWhite


def cholesky_psd(C: np.ndarray, tol: float = 1e-300) -> np.ndarray:
    """Lower Cholesky factor of a symmetric PSD matrix, allowing zero pivots.

    numpy.linalg.cholesky refuses singular matrices, but sigma_HW = 0 (the
    Black-Scholes validation case) makes the rate rows exactly zero. A zero
    pivot here just means that factor has no randomness; we zero its column.
    """
    n = C.shape[0]
    L = np.zeros_like(C, dtype=np.float64)
    for j in range(n):
        d = C[j, j] - np.dot(L[j, :j], L[j, :j])
        if d < -1e-12 * max(1.0, abs(C[j, j])):
            raise ValueError(f"matrix is not PSD (pivot {j} = {d})")
        if d <= tol:
            continue
        L[j, j] = np.sqrt(d)
        for i in range(j + 1, n):
            L[i, j] = (C[i, j] - np.dot(L[i, :j], L[j, :j])) / L[j, j]
    return L


@dataclass(frozen=True)
class EquityParams:
    sigma: float
    rho: float  # correlation between equity and the rate Brownian motion
    erp: float = 0.0  # equity risk premium (real-world only)

    def __post_init__(self):
        if self.sigma < 0:
            raise ValueError("equity sigma must be >= 0")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")


@dataclass(frozen=True)
class StepTable:
    """Everything deterministic the simulator needs per time step k (s=t[k], t=t[k+1]).

    Shapes: n_steps for vectors, (n_steps, 3, 3) for L. Shared verbatim by the
    CPU reference and (later) the GPU kernels, so both apply the same model.
    """

    times: np.ndarray  # n_steps + 1 grid points, times[0] = start time
    h: np.ndarray
    decay: np.ndarray  # e^{-a h}
    Bh: np.ndarray  # B(h)
    phi_int: np.ndarray  # int phi over the step
    x_shift: np.ndarray  # real-world drift added to x_t mean
    y_shift: np.ndarray  # real-world drift added to Y mean
    lnF_drift: np.ndarray  # (erp - mer - sigma_S^2/2) h
    L: np.ndarray

    @property
    def n_steps(self) -> int:
        return self.h.size


def build_step_table(
    hw: HullWhite,
    eq: EquityParams,
    times: np.ndarray,
    mer: float,
    lambda_r: float = 0.0,
) -> StepTable:
    times = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("time grid must be strictly increasing")
    n = times.size - 1
    a, sr, ss, rho = hw.a, hw.sigma, eq.sigma, eq.rho
    h = np.diff(times)
    decay = np.empty(n)
    Bh = np.empty(n)
    L = np.empty((n, 3, 3))
    cache: dict[float, tuple] = {}
    for k in range(n):
        hk = float(h[k])
        key = round(hk, 14)
        if key not in cache:
            d, b, v11, v22, v12 = hw.step_moments(hk)
            C = np.array(
                [
                    [v11, v12, rho * sr * ss * b],
                    [v12, v22, rho * sr * ss * (hk - b) / a],
                    [rho * sr * ss * b, rho * sr * ss * (hk - b) / a, ss**2 * hk],
                ]
            )
            cache[key] = (d, b, cholesky_psd(C))
        decay[k], Bh[k], L[k] = cache[key]
    phi_int = hw.phi_integral(times[:-1], times[1:])
    x_shift = sr * lambda_r * Bh
    y_shift = sr * lambda_r * (h - Bh) / a
    lnF_drift = (eq.erp - mer - 0.5 * ss**2) * h
    return StepTable(times, h, decay, Bh, phi_int, x_shift, y_shift, lnF_drift, L)
