import copy

import numpy as np
import pytest

from segfund import validate as V
from segfund.config import model_from_dict
from segfund.estimators import estimate
from segfund.reference.engine import simulate
from segfund.rng.philox import PhiloxNormals

BASE = V._load_cfg()
ROOT = V.DEFAULT_CONFIG.parents[1]


def test_antithetic_pairs_are_exact_mirrors():
    z = PhiloxNormals(9, antithetic_half=500).normals(3, 1000)
    assert np.array_equal(z[:, 500:], -z[:, :500])
    plain = PhiloxNormals(9).normals(3, 500)
    assert np.array_equal(z[:, :500], plain)  # first half is the ordinary stream


def test_antithetic_needs_even_paths():
    cfg = copy.deepcopy(BASE)
    cfg["simulation"]["antithetic"] = True
    with pytest.raises(ValueError, match="even"):
        simulate(model_from_dict(cfg, ROOT), 1001)


def test_unit_fund_matches_fund_for_single_deposit():
    m = model_from_dict(BASE, ROOT)
    r = simulate(m, 1000, PhiloxNormals(4))
    assert np.allclose(r.fund_T, m.contract.initial_deposit * r.unit_T, rtol=1e-12)


def test_control_variate_rejected_under_real_world():
    cfg = copy.deepcopy(BASE)
    cfg["economy"]["measure"] = "real_world"
    m = model_from_dict(cfg, ROOT)
    r = simulate(m, 100, PhiloxNormals(1))
    with pytest.raises(ValueError, match="risk-neutral"):
        estimate(r, m, control="hw_put")


def test_antithetic_se_uses_pairs():
    m = model_from_dict(BASE, ROOT)
    r = simulate(m, 2000, PhiloxNormals(2, antithetic_half=1000))
    e = estimate(r, m, antithetic=True, control="none")
    assert e.n_independent == 1000 and e.n_paths == 2000
    pairs = 0.5 * (r.pv_net[:1000] + r.pv_net[1000:])
    assert e.se == pytest.approx(pairs.std(ddof=1) / np.sqrt(1000))


def test_v10_unbiased_and_reduces_variance():
    recs = V.v10_variance_reduction(BASE, 50_000, 201)
    bad = [r for r in recs if not r["passed"]]
    assert not bad, V.format_table(bad)
    assert all(r["vrf"] > 1.0 for r in recs)
