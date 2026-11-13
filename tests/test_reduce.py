import numpy as np

from segfund import reduce as R


def test_sequential_fp32_loses_small_terms():
    # 2^24 + 1 is not representable in fp32: adding 1.0 to 2^24 does nothing.
    x = np.concatenate([[2.0**24], np.ones(1000, dtype=np.float32)]).astype(np.float32)
    assert R.sequential_fp32(x) == 2.0**24
    assert R.reference(x) == 2.0**24 + 1000
    assert R.kahan_lanes_fp32(x) == 2.0**24 + 1000


def test_compensated_and_pairwise_beat_sequential():
    rng = np.random.default_rng(0)
    x = rng.uniform(2000, 3000, 2_000_000).astype(np.float32)
    ref = R.exact(x)
    assert R.reference(x) == ref  # fp64 pairwise of fp32 inputs equals the exact sum here
    err = {k: abs(f(x) - ref) / ref for k, f in R.CPU_METHODS.items()}
    assert err["kahan_lanes_fp32"] < 1e-7
    assert err["pairwise_fp32"] < 1e-6
    assert err["sequential_fp32"] > 100 * err["kahan_lanes_fp32"]
