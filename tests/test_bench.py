"""Benchmark harness: equality gate runs before timing, rows carry the clock verdict."""

import pytest

pytestmark = pytest.mark.gpu


def test_torch_randn_baseline_is_statistically_consistent():
    import numpy as np
    import torch

    from segfund.config import load_model
    from segfund.gpu.torch_engine import simulate_torch
    from segfund.reference.engine import mean_and_se, simulate
    from segfund.rng.philox import PhiloxNormals
    from segfund.validate import DEFAULT_CONFIG

    m = load_model(DEFAULT_CONFIG)
    n = 20_000
    a = simulate(m, n, PhiloxNormals(5))
    b = simulate_torch(m, n, 5, dtype=torch.float32, normals="randn")
    (ma, sa), (mb, sb) = mean_and_se(a.pv_net), mean_and_se(b.pv_net)
    assert abs(ma - mb) <= 4 * np.hypot(sa, sb)
    assert not np.allclose(a.pv_net, b.pv_net)  # really a different stream


def test_run_bench_small(tmp_path):
    from segfund.bench.run import ENGINES, run_bench, summarize

    sizes = {e: [1 << 10] for e in ENGINES}
    res = run_bench(sizes, n_check=1 << 11, seed=3, numpy_reps=1, log=lambda *a: None)
    assert all(v["passed"] for s in res["checks"].values() for v in s.values())
    timed = [r for r in res["rows"] if "median_ms" in r]
    assert len(timed) == 2 * len(ENGINES)  # two scenarios, every engine timed
    assert all(r["median_ms"] > 0 and "quotable" in r for r in timed)
    assert "x NumPy" in summarize(res)
