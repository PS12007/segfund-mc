"""Counter-based random numbers: Philox4x32-10, bit-identical across NumPy, PyTorch and Triton.

Why counter-based? A normal PRNG (Mersenne Twister, PCG) is a sequence: number k
depends on numbers 0..k-1, so 10^6 GPU threads would have to share or split one
stream. Philox (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3",
SC'11) is a keyed bijection on 128-bit counters:

    (r0, r1, r2, r3) = Philox_key(c0, c1, c2, c3)

Any thread can compute any number directly from its coordinates. Here
    c0 = global path index, c1 = time step, c2 = stream id, c3 = 0, key = 64-bit seed.
Path 17 at step 40 gets the same numbers whether it runs on CPU or GPU, in any
batch, in any order. That is what makes GPU-vs-CPU comparisons on *identical
scenarios* possible, and what makes common-random-number sensitivities exact.

The round function matches `triton.language.philox_impl` (and Random123):
    hi_B, lo_B = mulhilo(0xCD9E8D57, c2);  hi_A, lo_A = mulhilo(0xD2511F53, c0)
    (c0, c1, c2, c3) = (hi_B ^ c1 ^ k0, lo_B, hi_A ^ c3 ^ k1, lo_A)
    k0 += 0x9E3779B9;  k1 += 0xBB67AE85
repeated 10 times.

Uniforms: u = ((r >> 9) + 0.5) * 2^-23, i.e. (2m + 1) * 2^-24 with m < 2^23. The
numerator needs at most 24 bits, so u is exact in fp32 as well as fp64: CPU and
GPU start from *identical* uniforms in (0, 1), and 0 and 1 are impossible.
(A first version used 24 bits, (r >> 8) + 0.5; that needs 25 bits and is NOT
exact in fp32. tests/test_philox.py caught it.) The cost: the smallest u is
2^-24, which caps |z| at sqrt(48 ln 2) = 5.77 in Box-Muller; the normal tail
mass beyond that is about 8e-9. Listed in Limits.

Normals: Box-Muller written with the angle in [-pi, pi) (better for GPU sin/cos):
    rad = sqrt(-2 ln u1),  z_a = -rad cos(pi (2 u2 - 1)),  z_b = -rad sin(pi (2 u2 - 1))
which equals the textbook rad * cos(2 pi u2), rad * sin(2 pi u2).
Per (path, step) we draw one Philox block: words (r0, r1) -> (z0, z1), (r2, r3) -> z2.
"""

from __future__ import annotations

import numpy as np

M_A = 0xD2511F53
M_B = 0xCD9E8D57
W_A = 0x9E3779B9
W_B = 0xBB67AE85
MASK32 = 0xFFFFFFFF
N_ROUNDS = 10
U23_SCALE = 2.0**-23

STREAM_MAIN = 0  # c2 values; later phases use other streams (e.g. nested inner paths)


def philox4x32(c0, c1, c2, c3, seed: int, n_rounds: int = N_ROUNDS):
    """Vectorised Philox4x32. Inputs are integer arrays (values < 2^32); returns 4 uint64 arrays < 2^32."""
    c0 = np.asarray(c0, dtype=np.uint64) & MASK32
    c1 = np.broadcast_to(np.asarray(c1, dtype=np.uint64) & MASK32, c0.shape)
    c2 = np.broadcast_to(np.asarray(c2, dtype=np.uint64) & MASK32, c0.shape)
    c3 = np.broadcast_to(np.asarray(c3, dtype=np.uint64) & MASK32, c0.shape)
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    k0 = np.uint64(seed & MASK32)
    k1 = np.uint64(seed >> 32)
    mA, mB, m32 = np.uint64(M_A), np.uint64(M_B), np.uint64(MASK32)
    s32 = np.uint64(32)
    for _ in range(n_rounds):
        pB = mB * c2  # < 2^64, exact in uint64
        pA = mA * c0
        c0, c1, c2, c3 = (pB >> s32) ^ c1 ^ k0, pB & m32, (pA >> s32) ^ c3 ^ k1, pA & m32
        k0 = np.uint64((int(k0) + W_A) & MASK32)
        k1 = np.uint64((int(k1) + W_B) & MASK32)
    return c0, c1, c2, c3


def uint_to_uniform(r):
    return ((np.asarray(r, dtype=np.uint64) >> np.uint64(9)).astype(np.float64) + 0.5) * U23_SCALE


def box_muller(u1, u2):
    rad = np.sqrt(-2.0 * np.log(u1))
    ang = np.pi * (2.0 * u2 - 1.0)
    return -rad * np.cos(ang), -rad * np.sin(ang)


def normals3(path_ids, step: int, seed: int, stream: int = STREAM_MAIN) -> np.ndarray:
    """(3, n) standard normals for the given paths at one step."""
    r0, r1, r2, r3 = philox4x32(path_ids, step, stream, 0, seed)
    z0, z1 = box_muller(uint_to_uniform(r0), uint_to_uniform(r1))
    z2, _ = box_muller(uint_to_uniform(r2), uint_to_uniform(r3))
    return np.stack([z0, z1, z2])


def antithetic_counter(ids: np.ndarray, half: int):
    """Antithetic pairing used by every engine: path g < half uses counter g with +z;
    path g >= half uses counter g - half with -z. half = 0 means no pairing."""
    ids = np.asarray(ids, dtype=np.uint64)
    if half <= 0:
        return ids, np.ones(ids.shape)
    second = ids >= np.uint64(half)
    ctr = np.where(second, ids - np.uint64(half), ids)
    return ctr, np.where(second, -1.0, 1.0)


class PhiloxNormals:
    """RNG object for the reference engine: normals(step, n) for paths [offset, offset + n)."""

    def __init__(self, seed: int, path_offset: int = 0, stream: int = STREAM_MAIN, antithetic_half: int = 0):
        self.seed = int(seed)
        self.path_offset = int(path_offset)
        self.stream = int(stream)
        self.antithetic_half = int(antithetic_half)

    def normals(self, step: int, n_paths: int) -> np.ndarray:
        ids = np.arange(self.path_offset, self.path_offset + n_paths, dtype=np.uint64)
        ctr, sign = antithetic_counter(ids, self.antithetic_half)
        return normals3(ctr, step, self.seed, self.stream) * sign
