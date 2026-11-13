"""Phase 4 risk study on the fused GPU kernel. Writes results/risk/risk_study.{json,txt} and loss_hist.png.

1. CTE standard error: is the Manistre-Hancock SE formula right? Compare the
   mean reported SE with the actual spread of CTE estimates over R independent
   real-world replications.
2. Common random numbers: delta / rho SE with CRN vs independent seeds.
3. Bump size: delta for h = 0.1%, 1%, 5% and rho for 1, 10, 50 bp, with CRN SEs,
   to show the finite-difference bias / noise trade-off.
4. Loss distribution plot (real-world PV of net cash flows) with VaR and CTE marked.
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

from segfund import risk  # noqa: E402
from segfund.cli import _meta  # noqa: E402
from segfund.config import load_config, with_measure  # noqa: E402
from segfund.gpu.equality import gpu_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--reps", type=int, default=32)
    a = ap.parse_args()
    cfg, base_dir = load_config(ROOT / "configs" / "base.yaml")
    m_rn = with_measure(cfg, "risk_neutral", base_dir)
    m_rw = with_measure(cfg, "real_world", base_dir)
    sim = risk.gpu_sim
    seeds = np.random.default_rng(4040).integers(1, 2**63, size=1000)
    si = iter(int(s) for s in seeds)
    t0 = time.perf_counter()
    L = [f"Base contract, fused GPU kernel, N = {a.n} paths per run", ""]

    # 1. CTE SE check
    cte_rows = {}
    est = {0.70: [], 0.95: []}
    rep_se = {0.70: [], 0.95: []}
    for _ in range(a.reps):
        r = sim(m_rw, a.n, next(si), False)
        for al in est:
            c = risk.cte(r.pv_net, al)
            est[al].append(c.cte)
            rep_se[al].append(c.se)
    L.append(f"1. CTE standard error: {a.reps} independent real-world replications")
    for al in est:
        e = np.asarray(est[al])
        emp = float(e.std(ddof=1))
        rse = float(np.mean(rep_se[al]))
        # sampling uncertainty of a standard deviation from R normal draws: ~ emp / sqrt(2(R-1))
        band = emp / math.sqrt(2 * (a.reps - 1))
        cte_rows[f"cte{int(al * 100)}"] = dict(mean=float(e.mean()), empirical_sd=emp, mean_reported_se=rse,
                                               ratio=rse / emp, sd_uncertainty=band)
        L.append(f"   CTE({int(al * 100)}): mean {e.mean():,.2f}  empirical SD of estimates {emp:.3f} (+/- {band:.3f})"
                 f"  mean Manistre-Hancock SE {rse:.3f}  ratio {rse / emp:.2f}")
    L.append("")

    # 2 + 3. CRN vs independent, bump sizes
    sens = {"delta": [], "rho": []}
    L.append("2-3. Sensitivities (risk-neutral, antithetic), common random numbers unless stated.")
    L.append("     One seed for every bump size, so differences between rows are finite-difference error, not noise.")
    s_delta, s_rho = next(si), next(si)
    for h in (0.001, 0.01, 0.05):
        d = risk.delta(sim, m_rn, a.n, s_delta, h=h, antithetic=True)
        ind = risk.independent_delta_se(sim, m_rn, a.n, s_delta, h=h, antithetic=True)
        sens["delta"].append(dict(bump=h, value=d.value, se_crn=d.se, se_independent=ind, se_ratio=ind / d.se))
        L.append(f"   delta h={h:>6.1%}: {d.value:+.6f}  SE(CRN) {d.se:.6f}  SE(independent seeds) {ind:.6f}"
                 f"  -> CRN is {ind / d.se:,.0f}x tighter")
    for bp in (1, 10, 50):
        d = risk.rho(sim, m_rn, a.n, s_rho, shift=bp * 1e-4, antithetic=True)
        sens["rho"].append(dict(bump_bp=bp, value=d.value, se_crn=d.se))
        L.append(f"   rho shift={bp:>3d}bp: {d.value:+.4f} per bp  SE(CRN) {d.se:.4f}")
    L.append("")

    # 4. loss distribution plot
    r = sim(m_rw, a.n, next(si), False)
    c70, c95 = risk.cte(r.pv_net, 0.70), risk.cte(r.pv_net, 0.95)
    L.append(f"4. Real-world loss distribution (one run): mean {r.pv_net.mean():,.2f}, "
             f"P(loss > 0) = {np.mean(r.pv_net > 0):.3f}, VaR95 {c95.var:,.2f}, CTE95 {c95.cte:,.2f}, CTE70 {c70.cte:,.2f}")
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.hist(r.pv_net, bins=200, color="#5b7fa6", alpha=0.85)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.axvline(c95.var, color="#c2552d", ls="--", lw=1.2, label=f"VaR(95) {c95.var:,.0f}")
    ax.axvline(c95.cte, color="#c2552d", lw=1.6, label=f"CTE(95) {c95.cte:,.0f}")
    ax.axvline(c70.cte, color="#2e8b57", lw=1.6, label=f"CTE(70) {c70.cte:,.0f}")
    ax.set_yscale("log")
    ax.set_xlabel(r"PV of guarantee claims minus fee income (\$ per \$100k deposit), real-world")
    ax.set_ylabel("paths (log scale)")
    ax.set_title(f"Loss distribution, base contract, {a.n:,} real-world paths", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "results" / "risk"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "loss_hist.png")

    text = "\n".join(L)
    print(text)
    meta = _meta() | gpu_info() | {"args": vars(a), "wall_s": time.perf_counter() - t0}
    (out / "risk_study.json").write_text(json.dumps(dict(meta=meta, cte_se_check=cte_rows, sensitivities=sens), indent=2))
    (out / "risk_study.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n\n" + text + "\n")
    print(f"wrote {out / 'risk_study.txt'} and loss_hist.png")


if __name__ == "__main__":
    main()
