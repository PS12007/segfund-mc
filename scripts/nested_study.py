"""Phase 5: brute-force nested simulation vs LSMC. Writes results/nested/nested_study.{json,txt} and nested.png.

Setup: base contract, valuation date t0 = 1 year (12 of 120 monthly steps).
  * test set: N_TEST real-world outer scenarios (fixed seed)
  * ground truth (GT): on the test set, N_IN_GT risk-neutral inner paths per outer
    scenario, antithetic + control variate
  * brute force with fewer inner paths (with and without control variate)
  * LSMC: independent real-world fit scenarios, 2 inner paths each (one antithetic
    pair, no control variate), regressed on basis functions of the t0 state;
    evaluated on the test set
Each cheaper method is repeated R times with fresh seeds; errors are measured
against GT on the same test scenarios (so outer sampling error cancels).

Cost is reported as total inner paths (hardware-independent) and wall time.
Wall times are medians over the R reps (so a first call's Triton compile is excluded),
with CUDA synchronisation, NOT clock-gated benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from segfund import nested as N  # noqa: E402
from segfund.cli import _meta  # noqa: E402
from segfund.config import load_config, with_measure  # noqa: E402
from segfund.gpu.equality import gpu_info  # noqa: E402


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=8192)
    ap.add_argument("--n-in-gt", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--t0", type=float, default=1.0)
    a = ap.parse_args()
    cfg, base = load_config(ROOT / "configs" / "base.yaml")
    rn, rw = with_measure(cfg, "risk_neutral", base), with_measure(cfg, "real_world", base)
    seeds = iter(int(s) for s in np.random.default_rng(5150).integers(1, 2**62, size=10_000))
    t_start = time.perf_counter()

    test = N.simulate_outer(rw, a.n_test, next(seeds), t0=a.t0)
    gt, gt_s = timed(lambda: N.inner_values(rn, test, a.n_in_gt, next(seeds), antithetic=True, control=True))
    L_gt = N.losses(test, gt.v)
    from segfund.risk import cte

    c95, c70 = cte(L_gt, 0.95), cte(L_gt, 0.70)
    lines = [
        f"Base contract, t0 = {a.t0} y, test set {a.n_test} real-world outer scenarios",
        f"Ground truth: {a.n_in_gt} inner paths per outer (antithetic + CV, pooled beta {gt.beta:.3f}),"
        f" {gt.paths:,} inner paths, {gt_s:.2f} s wall",
        f"  GT value noise (RMS of per-outer SE): {np.sqrt(np.mean(gt.se ** 2)):.3f}  (values range {gt.v.min():,.0f} .. {gt.v.max():,.0f})",
        f"  GT CTE(95) {c95.cte:,.2f} (outer-sampling SE {c95.se:.2f}),  CTE(70) {c70.cte:,.2f} (SE {c70.se:.2f})",
        "",
    ]

    rows = []

    def record(method, cost_paths, results, times):
        keys = results[0].keys()
        agg = {k: [r[k] for r in results] for k in keys}
        secs = float(np.median(times))  # median over reps: excludes the one-off Triton compile of a first call
        row = dict(method=method, inner_paths=cost_paths, wall_s_median=secs, wall_s_reps=[float(t) for t in times],
                   rmse_value=float(np.mean(agg["rmse_net_of_gt_noise"])),
                   cte95_err_mean=float(np.mean(agg["cte95_error"])),
                   cte95_err_rms=float(np.sqrt(np.mean(np.square(agg["cte95_error"])))),
                   cte70_err_mean=float(np.mean(agg["cte70_error"])),
                   cte70_err_rms=float(np.sqrt(np.mean(np.square(agg["cte70_error"])))),
                   reps=len(results), raw=results)
        rows.append(row)
        print(f"{method:<34} paths {cost_paths:>12,}  RMSE(V) {row['rmse_value']:8.2f}  "
              f"CTE95 err mean {row['cte95_err_mean']:+8.2f} rms {row['cte95_err_rms']:7.2f}  "
              f"CTE70 err rms {row['cte70_err_rms']:6.2f}  wall {secs:.3f}s", flush=True)

    # brute force with fewer inner paths
    for ctrl in (False, True):
        for n_in in (8, 32, 128, 512, 2048):
            res, times = [], []
            for r in range(a.reps):
                iv, s = timed(lambda: N.inner_values(rn, test, n_in, next(seeds), antithetic=True, control=ctrl))
                times.append(s)
                res.append(N.compare(iv.v, gt, test))
            record(f"brute force n_in={n_in}{' +CV' if ctrl else ''}", a.n_test * n_in, res, times)

    # LSMC
    for kind in N.BASES:
        for k in (12, 14, 16, 18, 20):
            n_fit = 1 << k
            res, times = [], []
            for r in range(a.reps):
                def run():
                    of = N.simulate_outer(rw, n_fit, next(seeds), t0=a.t0)
                    y = N.pair_targets(rn, of, next(seeds))
                    px = N.lsmc_fit(rn, of, y, kind, fit_paths=2 * n_fit)
                    return px, px.predict(rn, test)
                (px, vp), s = timed(run)
                times.append(s)
                c = N.compare(vp, gt, test)
                c["r2_noisy"] = px.r2_noisy
                res.append(c)
            record(f"LSMC {kind} n_fit=2^{k}", 2 * n_fit, res, times)

    # CPU cost of brute force (measured on a slice, extrapolated linearly in paths)
    sl = test.subset(np.arange(16))
    t = time.perf_counter()
    N.inner_values(rn, sl, 512, 1, antithetic=True, control=True, engine="cpu")
    cpu_s_per_path = (time.perf_counter() - t) / (16 * 512)

    lines.append(f"{'method':<34} {'inner paths':>13} {'RMSE(V)':>9} {'CTE95 err mean':>15} {'rms':>8} {'CTE70 err rms':>14} {'GPU wall (median)':>18}")
    for r in rows:
        lines.append(f"{r['method']:<34} {r['inner_paths']:>13,} {r['rmse_value']:>9.2f} {r['cte95_err_mean']:>+15.2f} "
                     f"{r['cte95_err_rms']:>8.2f} {r['cte70_err_rms']:>14.2f} {r['wall_s_median']:>17.3f}s")
    lines += [
        "",
        f"RMSE(V): per-outer value error vs GT, net of GT's own noise, averaged over {a.reps} reps.",
        "CTE err: proxy CTE - GT CTE on the same test scenarios (mean = bias, rms = typical size).",
        f"NumPy fp64 CPU cost (measured on 16 x 512 inner paths): {cpu_s_per_path * 1e6:.1f} us per inner path;",
        f"  at that rate the GT run ({gt.paths:,} paths) would take ~{gt.paths * cpu_s_per_path / 60:,.0f} min on this CPU"
        f" (extrapolated, not run). GPU took {gt_s:.2f} s (not clock-gated).",
    ]
    text = "\n".join(lines)
    print(text)

    out = ROOT / "results" / "nested"
    out.mkdir(parents=True, exist_ok=True)
    meta = _meta() | gpu_info() | {"args": vars(a), "wall_s": time.perf_counter() - t_start}
    summary = dict(gt=dict(paths=gt.paths, wall_s=gt_s, beta=gt.beta, noise_rms=float(np.sqrt(np.mean(gt.se ** 2))),
                           cte95=c95.as_dict(), cte70=c70.as_dict()),
                   cpu_s_per_inner_path=cpu_s_per_path, rows=rows)
    (out / "nested_study.json").write_text(json.dumps(dict(meta=meta, summary=summary), indent=2))
    (out / "nested_study.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n\n" + text + "\n")
    plot(rows, c95.se, out / "nested.png")
    print(f"wrote {out / 'nested_study.txt'} and nested.png")


def plot(rows, outer_se, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
    groups = {}
    for r in rows:
        key = "brute force +CV" if "+CV" in r["method"] else r["method"].split(" n_")[0]
        groups.setdefault(key, []).append(r)
    styles = {"brute force": ("#4a4a4a", "o"), "brute force +CV": ("#2f6db5", "o"),
              "LSMC poly": ("#c2552d", "s"), "LSMC poly+put": ("#2e8b57", "s")}
    for key, rs in groups.items():
        col, mk = styles.get(key, ("#999999", "x"))
        ax.loglog([r["inner_paths"] for r in rs], [max(r["cte95_err_rms"], 1e-2) for r in rs], marker=mk, color=col, label=key)
    ax.axhline(outer_se, color="#999999", ls="--", lw=1, label=f"outer-sampling SE of CTE(95) ({outer_se:.0f})")
    ax.set_xlabel("total inner paths (cost)")
    ax.set_ylabel(r"RMS error of CTE(95) vs ground truth (\$)")
    ax.set_title("Nested simulation: brute force vs LSMC (base contract, t0 = 1y)", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()
