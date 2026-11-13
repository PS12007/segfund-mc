import numpy as np
import pytest

from segfund.rng.philox import PhiloxNormals, philox4x32, uint_to_uniform

# Known-answer vectors for Philox4x32-10 from Random123 (kat_vectors).
# key (k0, k1) is passed as one 64-bit seed: seed = k1 << 32 | k0.
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


@pytest.mark.parametrize("ctr,key,expect", KAT)
def test_philox_known_answers(ctr, key, expect):
    seed = (key[1] << 32) | key[0]
    out = philox4x32(np.array([ctr[0]]), ctr[1], ctr[2], ctr[3], seed)
    assert tuple(int(o[0]) for o in out) == expect


def test_uniforms_open_interval_and_exact():
    u = uint_to_uniform(np.array([0, 0xFFFFFFFF, 511, 512], dtype=np.uint64))
    assert u[0] == 2.0**-24 and u[1] == 1.0 - 2.0**-24
    assert u[2] == u[0] and u[3] == 1.5 * 2.0**-23
    assert np.all(u.astype(np.float32).astype(np.float64) == u)  # exactly representable in fp32


def test_counter_based_batches_agree():
    """Paths 1000..1999 are the same whether generated alone or inside a bigger batch."""
    big = PhiloxNormals(42).normals(7, 3000)
    part = PhiloxNormals(42, path_offset=1000).normals(7, 1000)
    assert np.array_equal(big[:, 1000:2000], part)


def test_normals_moments():
    z = PhiloxNormals(2026).normals(0, 400_000)
    assert np.abs(z.mean(axis=1)).max() < 4 / np.sqrt(400_000)
    assert np.abs(z.var(axis=1) - 1).max() < 4 * np.sqrt(2 / 400_000)
    c = np.corrcoef(z)
    assert np.abs(c[np.triu_indices(3, 1)]).max() < 4 / np.sqrt(400_000)
    assert np.abs(z).max() < 5.77  # tail cap from 23-bit uniforms
