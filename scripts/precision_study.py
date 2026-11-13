"""Phase 3 precision study. Writes results/precision/precision_study.{json,txt}.

Part A  per-path accuracy of three fused-kernel variants vs the fp64 NumPy reference
        on identical scenarios: FAST_MATH (tl.exp etc.), default (libdevice), default + KAHAN.
Part B  error of the *portfolio sum* of N fp32 per-path values, N = 2^16 .. 2^26,
        for five summation methods, compared with the Monte Carlo standard error.
Part C  cost of each kernel variant: CUDA-event timing with the clock gate
        (segfund.bench.timing). Rows that fail the gate are reported as not quotable.

Usage: python scripts/precision_study.py [--n-a 500000] [--max-log2 26]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segfund import reduce as R  # noqa: E402
from segfund.bench.timing import ClockMonitor, gpu_static, time_gpu  # noqa: E402
from segfund.cli import _meta  # noqa: E402
from segfund.config import load_model  # noqa: E402
from segfund.gpu.equality import gpu_info  # noqa: E402
from segfund.gpu.kernels import N_OUT, FusedEngine  # noqa: E402
from segfund.reference.engine import mean_and_se, simulate  # noqa: E402
from segfund.rng.philox import PhiloxNormals  # noqa: E402

VARIANTS = {
    "FAST_MATH (tl.exp/log/sin/cos)": dict(fast_math=True, kahan=False),
    "default (libdevice)": dict(fast_math=False, kahan=False),
    "default + KAHAN (compensated int r and PV sums)": dict(fast_math=False, kahan=True),
}


def part_a(model, n, seed):
    cpu = simulate(model, n, PhiloxNormals(seed))
    rows = []
    for label, kw in VARIANTS.items():
        g = FusedEngine(model, **kw).simulate(n, seed, antithetic=False)
        row = {"variant": label}
        for f in ("fund_T", "discount_T", "inforce_T"):
            rel = np.abs(getattr(g, f) / getattr(cpu, f) - 1)
            row[f"{f}_max_rel"] = float(rel.max())
            row[f"{f}_mean_rel"] = float(rel.mean())
            row[f"{f}_mean_signed_rel"] = float((getattr(g, f) / getattr(cpu, f) - 1).mean())
        for f in ("pv_net", "pv_fees", "pv_maturity_claims"):
            d = getattr(g, f) - getattr(cpu, f)
            mc, se = mean_and_se(getattr(cpu, f))
            row[f"{f}_mean_diff"] = float(d.mean())
            row[f"{f}_mean_diff_over_se"] = float(d.mean() / se)
            row[f"{f}_path_max_abs"] = float(np.abs(d).max())
        rows.append(row)
    return rows


def part_b(model, max_log2, seed):
    eng = FusedEngine(model)
    n_max = 1 << max_log2
    out = eng.alloc(n_max)
    eng.launch(out, n_max, seed)
    torch.cuda.synchronize()
    v = out.view(N_OUT, n_max)
    cols = {"pv_fees": v[2].clone(), "pv_net": (v[0] + v[1] - v[2])}  # pv_net formed in fp32 on the GPU
    del out, v
    rows = []
    for name, col_gpu in cols.items():
        col = col_gpu.cpu().numpy()
        for k in range(16, max_log2 + 1, 2):
            n = 1 << k
            x = col[:n]
            ref = R.reference(x)
            mean_ref = ref / n
            se = float(np.std(x.astype(np.float64), ddof=1) / np.sqrt(n))
            row = {"quantity": name, "n": n, "mean_reference": mean_ref, "mc_se": se}
            if n <= 1 << 22:
                row["reference_minus_fsum"] = ref - R.exact(x)
            for mname, f in R.CPU_METHODS.items():
                t = time.perf_counter()
                s = f(x)
                row[f"{mname}_mean_err"] = s / n - mean_ref
                row[f"{mname}_cpu_s"] = time.perf_counter() - t
            for mname, f in R.GPU_METHODS.items():
                s = f(col_gpu[:n])
                row[f"{mname}_mean_err"] = s / n - mean_ref
            rows.append(row)
        del col_gpu
    torch.cuda.empty_cache()
    return rows


def part_c(model, n, seed):
    rows = []
    with ClockMonitor() as mon:
        for label, kw in VARIANTS.items():
            eng = FusedEngine(model, **kw)
            out = eng.alloc(n)
            fn = lambda: eng.launch(out, n, seed)  # noqa: E731
            fn()
            torch.cuda.synchronize()
            t = time_gpu(label, fn, mon)
            d = t.as_dict()
            d["paths_per_s"] = n / (t.median_ms / 1e3)
            rows.append(d)
        static = gpu_static()
    return rows, static


def fmt(a_rows, b_rows, c_rows, n_a, n_c) -> str:
    L = [f"PART A: per-path accuracy vs NumPy fp64, base contract, {n_a} identical paths"]
    for r in a_rows:
        L.append(f"  {r['variant']}")
        L.append(f"    fund_T   max rel {r['fund_T_max_rel']:.2e}  mean signed rel {r['fund_T_mean_signed_rel']:+.2e}")
        L.append(f"    disc_T   max rel {r['discount_T_max_rel']:.2e}  mean signed rel {r['discount_T_mean_signed_rel']:+.2e}")
        for f in ("pv_net", "pv_fees", "pv_maturity_claims"):
            L.append(f"    {f:<18} mean diff {r[f + '_mean_diff']:+.3e} ({r[f + '_mean_diff_over_se']:+.1e} SE)  worst path {r[f + '_path_max_abs']:.3f}")
    L.append("\nPART B: error of the portfolio MEAN from summing N fp32 per-path values (fused kernel, default math)")
    L.append("  err = method mean - fp64 reference mean, in dollars per policy; SE = Monte Carlo standard error")
    meths = list(R.CPU_METHODS) + list(R.GPU_METHODS)
    for q in ("pv_fees", "pv_net"):
        L.append(f"  {q}:")
        L.append("    " + f"{'N':>10} {'MC SE':>9} " + " ".join(f"{m:>18}" for m in meths))
        for r in [r for r in b_rows if r["quantity"] == q]:
            L.append("    " + f"{r['n']:>10} {r['mc_se']:>9.3f} " + " ".join(f"{r[m + '_mean_err']:>+18.3e}" for m in meths))
    chk = [r for r in b_rows if "reference_minus_fsum" in r]
    L.append(f"  fp64 reference vs exact math.fsum (N <= 2^22): max |diff| of totals = {max(abs(r['reference_minus_fsum']) for r in chk):.3e}")
    L.append(f"\nPART C: kernel cost, {n_c} paths, CUDA events, clock-gated")
    for r in c_rows:
        q = "QUOTABLE" if r["quotable"] else "NOT QUOTABLE: " + "; ".join(r["reasons"])
        ck = r["clocks"]
        L.append(f"  {r['label']}: median {r['median_ms']:.3f} ms  CI95 [{r['ci95_ms'][0]:.3f}, {r['ci95_ms'][1]:.3f}]"
                 f"  IQR {r['iqr_frac']:.1%}  reps {r['n_reps']}  SM {ck.get('sm_mhz_min', 0):.0f}-{ck.get('sm_mhz_max', 0):.0f} MHz"
                 f"  ({ck.get('n_samples', 0)} samples)  {r['paths_per_s']:.3e} paths/s  [{q}]")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a", type=int, default=500_000)
    ap.add_argument("--n-c", type=int, default=1 << 22)
    ap.add_argument("--max-log2", type=int, default=26)
    ap.add_argument("--seed", type=int, default=31337)
    a = ap.parse_args()
    model = load_model(ROOT / "configs" / "base.yaml")
    t0 = time.perf_counter()
    a_rows = part_a(model, a.n_a, a.seed)
    b_rows = part_b(model, a.max_log2, a.seed + 1)
    c_rows, static = part_c(model, a.n_c, a.seed + 2)
    text = fmt(a_rows, b_rows, c_rows, a.n_a, a.n_c)
    print(text)
    meta = _meta() | gpu_info() | {"gpu_static": static, "args": vars(a), "wall_s": time.perf_counter() - t0}
    out = ROOT / "results" / "precision"
    out.mkdir(parents=True, exist_ok=True)
    (out / "precision_study.json").write_text(json.dumps(dict(meta=meta, part_a=a_rows, part_b=b_rows, part_c=c_rows), indent=2))
    (out / "precision_study.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n\n" + text + "\n")
    print(f"wrote {out / 'precision_study.txt'}")


if __name__ == "__main__":
    main()
