import math
from statistics import NormalDist

import numpy as np
import pytest

from segfund import validate as V
from segfund.config import bumped, model_from_dict
from segfund.curves import FlatCurve, ShiftedCurve
from segfund.reference.engine import simulate
from segfund.risk import cte, cte_gbm_put_closed_form
from segfund.rng.philox import PhiloxNormals

BASE = V._load_cfg()
ROOT = V.DEFAULT_CONFIG.parents[1]


def test_cte_on_normal_sample_matches_formula():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2_000_000)
    for a in (0.7, 0.95):
        z = NormalDist().inv_cdf(a)
        exact = NormalDist().pdf(z) / (1 - a)  # E[X | X > z] for a standard normal
        est = cte(x, a)
        assert abs(est.cte - exact) < 4 * est.se
        assert est.var == pytest.approx(z, abs=0.01)


def test_cte_rejects_too_small_tail():
    with pytest.raises(ValueError):
        cte(np.arange(10.0), 0.95)


def test_cte_closed_form_matches_direct_lognormal_sampling():
    S0, K, T, r, mer, sig, erp = 100.0, 90.0, 10.0, 0.03, 0.02, 0.2, 0.04
    rng = np.random.default_rng(2)
    m = math.log(S0) + (r + erp - mer - 0.5 * sig**2) * T
    F = np.exp(m + sig * math.sqrt(T) * rng.standard_normal(4_000_000))
    L = math.exp(-r * T) * np.maximum(K - F, 0)
    for a in (0.7, 0.95, 0.99):
        exact, _ = cte_gbm_put_closed_form(S0, K, T, r, mer, sig, erp, a)
        est = cte(L, a)
        assert abs(est.cte - exact) < 4 * est.se


def test_shifted_curve():
    c = ShiftedCurve(FlatCurve(0.03), 0.001)
    assert c.zero_rate(5.0) == pytest.approx(0.031)
    assert c.discount(10.0) == pytest.approx(math.exp(-0.31))


def test_fund_shock_moves_fund_not_guarantee():
    m = model_from_dict(BASE, ROOT)
    a = simulate(m, 1000, PhiloxNormals(3))
    b = simulate(bumped(m, fund_shock=1.05), 1000, PhiloxNormals(3))
    # without dynamic lapse F scales exactly; with it (base has lapse) the fund still scales
    assert np.allclose(b.fund_T, 1.05 * a.fund_T, rtol=1e-12)
    assert b.pv_maturity_claims.mean() < a.pv_maturity_claims.mean()  # guarantee unchanged, fund higher


def test_pathwise_requires_simple_contract():
    m = model_from_dict(BASE, ROOT)  # base has dynamic lapse
    with pytest.raises(ValueError, match="pathwise"):
        simulate(m, 10, pathwise_delta=True)


def test_v11_v12_small():
    recs = V.v11_cte(BASE, 50_000, 301) + V.v12_sensitivities(BASE, 50_000, 302)
    bad = [r for r in recs if not r["passed"]]
    assert not bad, V.format_table(bad)
