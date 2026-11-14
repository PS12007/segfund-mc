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


def test_compiled_engine_matches_eager_randn():
    """torch.compile (with and without a CUDA graph) runs the same scenarios as the eager engine."""
    import numpy as np

    from segfund.config import load_model
    from segfund.gpu.torch_compiled import CompiledEngine
    from segfund.gpu.torch_engine import simulate_torch_tensors
    from segfund.validate import DEFAULT_CONFIG

    m = load_model(DEFAULT_CONFIG)
    n, seed = 4096, 21
    eager = [t.double().cpu().numpy() for t in simulate_torch_tensors(m, n, seed, normals="randn")]
    for mode in ("compile", "compile+graph"):
        eng = CompiledEngine(m, mode)
        for _ in range(2):  # the second call reuses the captured graph and must reset its state
            got = [t.double().cpu().numpy() for t in eng.simulate(n, seed)]
            for a, b in zip(got[3:], eager[3:]):
                np.testing.assert_allclose(a, b, rtol=1.2e-4)  # 1e-6 x 120 steps, the phase 2 V8a budget
            pv, pve = got[0] + got[1] - got[2], eager[0] + eager[1] - eager[2]
            assert abs(pv.mean() - pve.mean()) < 0.01


def test_run_bench_small(tmp_path):
    from segfund.bench.run import ENGINES, run_bench, summarize

    plain = [e for e in ENGINES if not e.startswith("torch_compile")]  # compiled engines: own test (slow to compile)
    sizes = {e: [1 << 10] for e in plain}
    res = run_bench(sizes, n_check=1 << 11, seed=3, numpy_reps=1, log=lambda *a: None, with_compile=False)
    assert all(v["passed"] for s in res["checks"].values() for e, v in s.items() if e in plain)
    timed = [r for r in res["rows"] if "median_ms" in r]
    assert len(timed) == 2 * len(plain)  # two scenarios, every engine timed
    assert all(r["median_ms"] > 0 and "quotable" in r for r in timed)
    assert "x NumPy" in summarize(res)
