import copy

import numpy as np
import pytest

from segfund.config import load_model, model_from_dict
from segfund.reference.engine import simulate
from segfund.rng.philox import PhiloxNormals
from segfund.validate import DEFAULT_CONFIG, _load_cfg

BASE = _load_cfg()
ROOT = DEFAULT_CONFIG.parents[1]


def _m(**edits):
    cfg = copy.deepcopy(BASE)
    for k, v in edits.items():
        node = cfg
        parts = k.split("__")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return model_from_dict(cfg, ROOT)


def test_base_config_loads():
    m = load_model(DEFAULT_CONFIG)
    assert m.time_grid().size == 121
    assert m.mortality is not None and m.lapse.enabled


def test_unknown_key_rejected():
    cfg = copy.deepcopy(BASE)
    cfg["contract"]["gmmb_pc"] = 0.75
    with pytest.raises(ValueError, match="unknown key"):
        model_from_dict(cfg, ROOT)


def test_fee_above_mer_rejected():
    with pytest.raises(ValueError, match="guarantee_fee"):
        _m(contract__fees__guarantee_fee=0.03)


def test_maturity_must_be_whole_steps():
    with pytest.raises(ValueError, match="whole number"):
        _m(simulation__dt=0.3).time_grid()


def test_same_seed_same_answer():
    m = _m()
    a = simulate(m, 2000, PhiloxNormals(7)).pv_net
    b = simulate(m, 2000, PhiloxNormals(7)).pv_net
    assert np.array_equal(a, b)


def test_deterministic_when_no_volatility():
    m = _m(economy__hull_white__sigma=0.0, economy__equity__sigma=0.0,
           economy__curve={"source": "flat", "flat_rate": 0.03},
           decrements__mortality__enabled=False, decrements__lapse__enabled=False)
    r = simulate(m, 10, PhiloxNormals(1))
    T, mer = m.contract.maturity_years, m.contract.mer
    assert np.allclose(r.fund_T, 100000 * np.exp((0.03 - mer) * T), rtol=1e-12)
    assert np.allclose(r.discount_T, np.exp(-0.03 * T), rtol=1e-12)


def test_resets_never_lower_the_guarantee():
    off = simulate(_m(), 20000, PhiloxNormals(3))
    on = simulate(_m(contract__resets={"enabled": True, "every_years": 1, "min_years_to_maturity": 0}),
                  20000, PhiloxNormals(3))
    # Same random numbers: a ratchet can only raise G, so the maturity claim can only rise.
    assert np.all(on.pv_maturity_claims >= off.pv_maturity_claims - 1e-9)
    assert on.pv_maturity_claims.mean() > off.pv_maturity_claims.mean()


def test_later_deposit_must_be_on_grid():
    m = _m(contract__deposits=[{"t": 0.0, "amount": 50000}, {"t": 1.01, "amount": 50000}])
    with pytest.raises(ValueError, match="time grid"):
        simulate(m, 10)


def test_inforce_decreases():
    r = simulate(_m(), 1000, PhiloxNormals(5))
    assert np.all((r.inforce_T > 0) & (r.inforce_T < 1))
