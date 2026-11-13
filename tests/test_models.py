import math

import numpy as np
import pytest

from segfund.curves import FlatCurve, NelsonSiegelCurve
from segfund.models.equity import EquityParams, build_step_table, cholesky_psd
from segfund.models.hull_white import HullWhite, B
from segfund.models.lapse import DynamicLapse
from segfund.models.mortality import GompertzMakeham
from segfund.reference import closed_form as cf

CURVE = NelsonSiegelCurve(0.035, -0.01, 0.005, 2.0)


def test_cholesky_psd_matches_numpy_on_pd():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    C = A @ A.T + 0.1 * np.eye(3)
    assert np.allclose(cholesky_psd(C), np.linalg.cholesky(C))


def test_cholesky_psd_handles_zero_rate_vol():
    C = np.diag([0.0, 0.0, 0.04])
    L = cholesky_psd(C)
    assert np.allclose(L @ L.T, C)


def test_step_covariance_against_quadrature():
    """Check the closed-form step covariance by integrating the OU kernels numerically."""
    a, s, ss, rho, h = 0.08, 0.012, 0.2, -0.3, 0.5
    hw = HullWhite(CURVE, a, s)
    tab = build_step_table(hw, EquityParams(ss, rho), np.array([0.0, h]), mer=0.0)
    C = tab.L[0] @ tab.L[0].T
    u = np.linspace(0.0, h, 200001)
    kx = np.exp(-a * (h - u))  # x_t = int kx dW
    ky = B(a, h - u)  # Y = int ky dW
    q = lambda f: np.trapezoid(f, u)
    expect = np.array([
        [s**2 * q(kx * kx), s**2 * q(kx * ky), rho * s * ss * q(kx)],
        [s**2 * q(kx * ky), s**2 * q(ky * ky), rho * s * ss * q(ky)],
        [rho * s * ss * q(kx), rho * s * ss * q(ky), ss**2 * h],
    ])
    assert np.allclose(C, expect, rtol=1e-8, atol=1e-14)


def test_zcb_boundary_conditions():
    hw = HullWhite(CURVE, 0.05, 0.01)
    assert hw.zcb(3.0, 3.0, 0.02) == pytest.approx(1.0)
    assert hw.zcb(0.0, 7.0, 0.0) == pytest.approx(float(CURVE.discount(7.0)), rel=1e-14)
    assert hw.zcb(2.0, 7.0, 0.01) < hw.zcb(2.0, 7.0, -0.01)  # higher rates, lower bond price


def test_phi_integral_matches_quadrature():
    hw = HullWhite(CURVE, 0.05, 0.01)
    t = np.linspace(2.0, 3.0, 20001)
    assert hw.phi_integral(2.0, 3.0) == pytest.approx(np.trapezoid(hw.phi(t), t), rel=1e-7)


def test_mortality_survival_matches_hazard_integral():
    m = GompertzMakeham()
    ages = np.linspace(55, 65, 100001)
    expect = math.exp(-np.trapezoid(m.hazard(ages), ages))
    assert m.survival(55.0, 10.0) == pytest.approx(expect, rel=1e-9)
    q = m.step_death_probs(55, np.linspace(0, 10, 121))
    assert np.all((q > 0) & (q < 1)) and np.all(np.diff(q) > 0)


def test_lapse_multiplier_direction_and_bounds():
    lp = DynamicLapse(base_annual=0.05, k=1.0, lo=0.2, hi=2.0)
    itm = lp.step_prob(60.0, 100.0, 1 / 12)
    otm = lp.step_prob(150.0, 100.0, 1 / 12)
    assert itm < otm
    assert lp.multiplier(1.0, 100.0) == pytest.approx(0.2)
    assert lp.multiplier(1000.0, 100.0) == pytest.approx(2.0)
    assert lp.step_prob(100.0, 100.0, 1.0) == pytest.approx(0.05)
    assert DynamicLapse(enabled=False).step_prob(np.array([1.0]), 1.0, 1.0)[0] == 0.0


def test_bs_put_call_parity():
    S, K, T, r, q, v = 100.0, 90.0, 7.0, 0.03, 0.02, 0.2
    put = float(cf.bs_put(S, K, T, r, q, v))
    d1 = (math.log(S / K) + (r - q + 0.5 * v * v) * T) / (v * math.sqrt(T))
    call = S * math.exp(-q * T) * float(cf.norm_cdf(d1)) - K * math.exp(-r * T) * float(cf.norm_cdf(d1 - v * math.sqrt(T)))
    assert call - put == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), rel=1e-12)


def test_hw_put_reduces_to_bs_when_rate_vol_zero():
    hw = HullWhite(FlatCurve(0.03), 0.05, 0.0)
    eq = EquityParams(0.18, -0.2)
    assert float(cf.hw_gbm_put(hw, eq, 100.0, 100.0, 10.0, 0.0225)) == pytest.approx(
        float(cf.bs_put(100.0, 100.0, 10.0, 0.03, 0.0225, 0.18)), rel=1e-12
    )
