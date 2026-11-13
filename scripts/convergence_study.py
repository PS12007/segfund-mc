"""Phase 3 convergence study (V9) and variance-reduction factors (V10, base contract).

For N = 2^10 .. 2^22 and four estimators (plain, antithetic, control variate, both),
run R independent replications (distinct seeds) of the base contract on the fused
GPU kernel and record each replication's estimate and its reported SE.

  * empirical error = standard deviation of the R estimates (true sampling error)
  * reported SE     = mean of the R reported standard errors
  * slope           = OLS fit of log(empirical error) on log(N), with a 95% CI;
                      Monte Carlo theory says -0.5
  * bias check      = mean of all replications at the largest N vs a separate
                      2^25-path antithetic + control-variate reference, in units
                      of the combined standard error

Writes results/convergence/convergence.{json,txt} and convergence.png.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from segfund.cli import _meta  # noqa: E402
from segfund.config import load_model  # noqa: E402
from segfund.estimators import estimate  # noqa: E402
from segfund.gpu.equality import gpu_info  # noqa: E402
from segfund.gpu.kernels import FusedEngine  # noqa: E402

METHODS = [
    ("plain", False, "none"),
    ("antithetic", True, "none"),
    ("control variate", False, "hw_put"),
    ("antithetic + control variate", True, "hw_put"),
]
T_975 = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}  # Student t, two-sided 95%, by degrees of freedom


def ols_slope(x, y):
    x, y = np.asarray(x), np.asarray(y)
    X = np.column_stack([np.ones_like(x), x])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = x.size - 2
    s2 = float(res[0] / dof) if res.size else 0.0
    cov = s2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(cov[1, 1])
    t = T_975.get(dof, 1.96)
    return float(beta[1]), (float(beta[1] - t * se), float(beta[1] + t * se))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-log2", type=int, default=10)
    ap.add_argument("--max-log2", type=int, default=22)
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--ref-log2", type=int, default=25)
    ap.add_argument("--plot-only", action="store_true", help="redraw convergence.png from convergence.json")
    a = ap.parse_args()
    if a.plot_only:
        p = ROOT / "results" / "convergence"
        plot(json.loads((p / "convergence.json").read_text())["summary"], p / "convergence.png")
        return
    model = load_model(ROOT / "configs" / "base.yaml")
    eng = FusedEngine(model)
    t0 = time.perf_counter()

    Ns = [1 << k for k in range(a.min_log2, a.max_log2 + 1, 2)]
    data = {m[0]: {"N": [], "estimates": [], "reported_se": []} for m in METHODS}
    # Seeds: independent random 63-bit Philox keys from a fixed master seed. (Run 1 used
    # keys 1000 apart and R = 32; its plain-estimator slope CI missed -0.5, a check with
    # random keys gave no sign of key correlation, and the protocol was changed to this
    # one. Run 1 is kept in results/convergence/convergence_run1_*.)
    seed_rng = np.random.default_rng(20261113)
    for N in Ns:
        per = {m[0]: ([], []) for m in METHODS}
        for r in range(a.reps):
            seed = int(seed_rng.integers(1, 2**63))
            res_plain = eng.simulate(N, seed, antithetic=False)
            res_anti = eng.simulate(N, seed, antithetic=True)
            for name, anti, ctrl in METHODS:
                e = estimate(res_anti if anti else res_plain, model, antithetic=anti, control=ctrl)
                per[name][0].append(e.mean)
                per[name][1].append(e.se)
        for name in per:
            data[name]["N"].append(N)
            data[name]["estimates"].append(per[name][0])
            data[name]["reported_se"].append(per[name][1])
        print(f"N={N} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # reference: one big antithetic + CV run, in chunks so it fits in memory
    ref_n = 1 << a.ref_log2
    chunk = 1 << 22
    means, ses = [], []
    for i in range(ref_n // chunk):
        res = eng.simulate(chunk, int(seed_rng.integers(1, 2**63)), antithetic=True)
        e = estimate(res, model, antithetic=True, control="hw_put")
        means.append(e.mean)
        ses.append(e.se)
    ref_mean = float(np.mean(means))
    ref_se = float(np.sqrt(np.sum(np.square(ses))) / len(ses))

    summary = {}
    for name, *_ in METHODS:
        d = data[name]
        emp = [float(np.std(est, ddof=1)) for est in d["estimates"]]
        rep = [float(np.mean(s)) for s in d["reported_se"]]
        slope, ci = ols_slope(np.log(d["N"]), np.log(emp))
        big = np.asarray(d["estimates"][-1])
        big_mean = float(big.mean())
        big_se = float(big.std(ddof=1) / math.sqrt(big.size))
        z = (big_mean - ref_mean) / math.sqrt(big_se**2 + ref_se**2)
        summary[name] = dict(
            N=d["N"], empirical_error=emp, mean_reported_se=rep,
            slope=slope, slope_ci95=ci, slope_ci_contains_minus_half=bool(ci[0] <= -0.5 <= ci[1]),
            largest_N_mean=big_mean, largest_N_se_of_mean=big_se, bias_z_vs_reference=z,
        )
    plain_emp = np.asarray(summary["plain"]["empirical_error"])
    for name in summary:
        vrf = (plain_emp / np.asarray(summary[name]["empirical_error"])) ** 2
        summary[name]["vrf_by_N"] = vrf.tolist()

    lines = [f"Base contract (risk-neutral), fused GPU kernel, {a.reps} independent replications per N",
             f"Reference: {ref_n} paths antithetic + CV: {ref_mean:.4f} (SE {ref_se:.4f})", ""]
    for name, s in summary.items():
        lines.append(f"{name}:")
        lines.append(f"  {'N':>9} {'empirical err':>14} {'mean reported SE':>17} {'VRF vs plain':>13}")
        for N, e, r, v in zip(s["N"], s["empirical_error"], s["mean_reported_se"], s["vrf_by_N"]):
            lines.append(f"  {N:>9} {e:>14.4f} {r:>17.4f} {v:>13.2f}")
        lines.append(f"  slope of log(error) vs log(N): {s['slope']:+.3f}  95% CI [{s['slope_ci95'][0]:+.3f}, {s['slope_ci95'][1]:+.3f}]"
                     f"  contains -0.5: {s['slope_ci_contains_minus_half']}")
        lines.append(f"  bias check at N={s['N'][-1]}: mean of {a.reps} reps {s['largest_N_mean']:.4f}"
                     f" (SE {s['largest_N_se_of_mean']:.4f}) vs reference: z = {s['bias_z_vs_reference']:+.2f}")
        lines.append("")
    text = "\n".join(lines)
    print(text)

    out = ROOT / "results" / "convergence"
    out.mkdir(parents=True, exist_ok=True)
    meta = _meta() | gpu_info() | {"args": vars(a), "wall_s": time.perf_counter() - t0}
    (out / "convergence.json").write_text(json.dumps(dict(meta=meta, reference=dict(n=ref_n, mean=ref_mean, se=ref_se),
                                                          summary=summary, raw=data), indent=2))
    (out / "convergence.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n\n" + text + "\n")

    plot(summary, out / "convergence.png")
    print(f"wrote {out / 'convergence.txt'} and convergence.png")


def plot(summary: dict, path: Path) -> None:
    Ns = summary["plain"]["N"]
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=130)
    colors = ["#4a4a4a", "#2f6db5", "#c2552d", "#2e8b57"]
    for (name, s), col in zip(summary.items(), colors):
        ax.loglog(s["N"], s["empirical_error"], "o-", color=col, label=f"{name} (slope {s['slope']:+.2f})")
        ax.loglog(s["N"], s["mean_reported_se"], "x", color=col, alpha=0.7)
    Nn = np.asarray(Ns, dtype=float)
    ref_line = summary["plain"]["empirical_error"][0] * np.sqrt(Nn[0] / Nn)
    ax.loglog(Nn, ref_line, "--", color="#999999", lw=1, label=r"$\propto 1/\sqrt{N}$")
    ax.set_xlabel("paths N")
    ax.set_ylabel(r"error of the PV estimate (\$ per \$100k deposit)")
    ax.set_title("Monte Carlo error vs paths (o = empirical over replications, x = reported SE)", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()
