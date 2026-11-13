import numpy as np
import pytest

from segfund import nested as N
from segfund import validate as V
from segfund.config import model_from_dict
from segfund.reference.engine import simulate
from segfund.rng.philox import PhiloxNormals, antithetic_counter

BASE = V._load_cfg()
ROOT = V.DEFAULT_CONFIG.parents[1]


def test_segments_compose_exactly():
    """Running 0 -> k0, stopping, and restarting from the saved state with the same
    random stream reproduces the uninterrupted run: PV_full = PV_early + D(t0) * PV_after."""
    m = model_from_dict(BASE, ROOT)
    n, k0 = 2000, 12
    full = simulate(m, n, PhiloxNormals(7))
    a = simulate(m, n, PhiloxNormals(7), k_end=k0)
    state = {"x": a.x_end, "F": a.fund_T, "GM": a.gm_end, "GD": a.gd_end, "l": a.inforce_T}
    b = simulate(m, n, PhiloxNormals(7), k_start=k0, init_state=state)
    assert np.allclose(full.pv_net, a.pv_net + a.discount_T * b.pv_net, rtol=1e-11, atol=1e-8)
    assert np.allclose(full.fund_T, b.fund_T, rtol=1e-12)
    assert np.all(a.pv_maturity_claims == 0)  # no maturity claim before maturity


def test_antithetic_blocks_stay_inside_outer_scenarios():
    ids = np.arange(32, dtype=np.uint64)
    ctr, sign = antithetic_counter(ids, 4)  # blocks of 8
    assert np.array_equal(ctr[4:8], ids[0:4]) and np.all(sign[4:8] == -1)
    assert np.array_equal(ctr[12:16], ids[8:12]) and np.all(sign[8:12] == 1)


def test_valuation_date_must_be_inside_term():
    m = model_from_dict(BASE, ROOT)
    with pytest.raises(ValueError):
        N.valuation_step(m, 10.0)
    with pytest.raises(ValueError):
        N.valuation_step(m, 1.03)


def test_lsmc_recovers_a_known_function():
    """Regression machinery: noisy targets built from the put basis are recovered."""
    m = model_from_dict(BASE, ROOT)
    o = N.simulate_outer(m, 4000, 3, engine="cpu")
    A = N.basis(m, o, "poly+put")
    true = A @ np.linspace(1, 2, A.shape[1])
    y = o.l * (true + np.random.default_rng(0).standard_normal(o.n) * 50)
    px = N.lsmc_fit(m, o, y, "poly+put", fit_paths=0)
    assert np.sqrt(np.mean((px.predict(m, o) / o.l - true) ** 2)) < 5.0


def test_v13_nested_cpu():
    recs = V.v13_nested(BASE, 404, engine="cpu", n_outer_tower=2048)
    bad = [r for r in recs if not r["passed"]]
    assert not bad, V.format_table(bad)
