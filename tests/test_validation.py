"""Phase 1 closed-form validations at a CI-friendly path count.

The committed results in results/validation/ use 200k paths (`segfund validate`).
Here 50k paths keeps CI fast; the pass rule is the same |z| < 4.
"""

import pytest

from segfund import validate as V

BASE = V._load_cfg()
N = 50_000


def _check(recs):
    recs = recs if isinstance(recs, list) else [recs]
    bad = [r for r in recs if not r["passed"]]
    assert not bad, V.format_table(bad)


def test_v1_curve_fit():
    _check(V.v1_curve_fit(BASE))


@pytest.mark.slow
def test_v2_zcb():
    _check(V.v2_zcb(BASE, N, 101))


def test_v3_martingale():
    _check(V.v3_martingale(BASE, N, 102))


def test_v4_black_scholes():
    _check(V.v4_black_scholes(BASE, N, 103))


def test_v5_gmdb():
    _check(V.v5_gmdb(BASE, N, 104))


def test_v6_hw_put():
    _check(V.v6_hw_put(BASE, N, 105))
