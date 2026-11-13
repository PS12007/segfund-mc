"""GPU vs CPU equality (skipped automatically without CUDA + torch + triton, e.g. in CI)."""

import pytest

pytestmark = pytest.mark.gpu


def _assert_all(recs):
    from segfund.gpu.equality import format_table

    bad = [r for r in recs if r.get("gated", True) and not r["passed"]]
    assert not bad, format_table(bad)


def test_v7_rng_bitwise_and_normals():
    from segfund.gpu.equality import v7_rng

    _assert_all(v7_rng(n=1 << 16))


def test_v8_engines_match_reference():
    from segfund.gpu.equality import v8_engines

    recs, _ = v8_engines(n_paths=20_000, seed=777)
    _assert_all(recs)


def test_partial_block_and_offset():
    """n_paths not a multiple of BLOCK, nonzero path offset: masked tail must be right."""
    import numpy as np

    from segfund.config import load_model
    from segfund.gpu.kernels import FusedEngine
    from segfund.reference.engine import simulate
    from segfund.rng.philox import PhiloxNormals
    from segfund.validate import DEFAULT_CONFIG

    m = load_model(DEFAULT_CONFIG)
    n, off = 1000 + 37, 123_456
    g = FusedEngine(m, block=256).simulate(n, 99, path_offset=off)
    c = simulate(m, n, PhiloxNormals(99, path_offset=off))
    assert np.max(np.abs(g.fund_T / c.fund_T - 1)) < 1e-4


def test_v8_phase3_antithetic_and_kahan():
    from segfund.gpu.equality import v8_phase3

    _assert_all(v8_phase3(n_paths=20_000, seed=778))