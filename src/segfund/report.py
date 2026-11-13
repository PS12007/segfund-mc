"""Phase 6: one self-contained HTML report built from the saved results.

Nothing is re-simulated here. The report reads the JSON files that the CLI and
the study scripts wrote under results/, so every number on the page has a raw
file behind it (linked by path). Images are embedded as base64 so the HTML can
be opened or mailed on its own. A missing input becomes a visible "not run"
note, never a silently empty section.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

CSS = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#6b6b66;--line:#deded8;--card:#fff;--accent:#2f5d8a;--bad:#a33a2a;--ok:#2e6b3a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#e9e9e4;--muted:#9a9a93;--line:#34342f;--card:#1e1e1c;--accent:#8db4dc;--bad:#e08a7a;--ok:#8fc79a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px;border-bottom:1px solid var(--line);padding-bottom:6px}
.muted{color:var(--muted)}.warn{border-left:3px solid var(--bad);padding:8px 12px;background:var(--card);margin:16px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px;margin:8px 0}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}th{font-weight:600;color:var(--muted)}.scroll{overflow-x:auto}
img{max-width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:4px}.ok{color:var(--ok)}.bad{color:var(--bad)}
code{font-family:ui-monospace,Consolas,monospace;font-size:13px}
"""


def _load(rel: str) -> dict | None:
    p = RES / rel
    return json.loads(p.read_text()) if p.exists() else None


def _img(path: Path | None = None, png: bytes | None = None, alt: str = "") -> str:
    if png is None:
        if path is None or not path.exists():
            return f'<p class="muted">({html.escape(alt)}: image not found)</p>'
        png = path.read_bytes()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def _src(rel: str) -> str:
    return f'<p class="muted">Raw: <code>results/{rel}</code></p>'


def _missing(what: str, cmd: str) -> str:
    return f'<p class="warn">{what} not run yet. Produce it with <code>{html.escape(cmd)}</code>.</p>'


def section_price() -> str:
    d = _load("runs/base_gpu_4M.json")
    if d is None:
        return _missing("Base contract run", "segfund run --config configs/base.yaml --out results/runs/base_gpu_4M.json")
    f = lambda v, nd=2: f"{v:,.{nd}f}"  # noqa: E731
    rows = [
        ["Risk-neutral PV of claims minus fees", f(d["pv"]["mean"]), f(d["pv"]["se"])],
        ["&nbsp;&nbsp;PV maturity claims", f(d["components"]["pv_maturity_claims"]["mean"]), f(d["components"]["pv_maturity_claims"]["se"])],
        ["&nbsp;&nbsp;PV death claims", f(d["components"]["pv_death_claims"]["mean"]), f(d["components"]["pv_death_claims"]["se"])],
        ["&nbsp;&nbsp;PV guarantee fees", f(d["components"]["pv_fees"]["mean"]), f(d["components"]["pv_fees"]["se"])],
        ["Real-world mean PV", f(d["real_world_mean"]["mean"]), f(d["real_world_mean"]["se"])],
        ["Real-world CTE(70)", f(d["cte"]["cte70"]["cte"]), f(d["cte"]["cte70"]["se"])],
        ["Real-world CTE(95)", f(d["cte"]["cte95"]["cte"]), f(d["cte"]["cte95"]["se"])],
        ["Delta (per $ of fund)", f(d["delta"]["value"], 5), f(d["delta"]["se"], 5)],
        ["Rho (per +1bp)", f(d["rho"]["value"], 4), f(d["rho"]["se"], 4)],
    ]
    return (f"<p>Base contract (<code>configs/base.yaml</code>), {d['n_paths']:,} paths on the {d['engine'].upper()} engine, "
            f"per $100,000 deposited. Component SEs are without the control variate.</p>"
            + _table(["Quantity", "Value", "SE"], rows) + _src("runs/base_gpu_4M.json"))


def section_tail() -> str:
    return ("<p>Distribution of the real-world PV of claims minus fees. Positive = loss to the insurer. "
            "CTE(95) is the average of the worst 5%.</p>" + _img(RES / "risk" / "loss_hist.png", alt="loss histogram")
            + _src("risk/risk_study.txt"))


def section_convergence() -> str:
    d = _load("convergence/convergence.json")
    if d is None:
        return _missing("Convergence study", "python scripts/convergence_study.py")
    ref = d["reference"]
    return (f"<p>Empirical error vs number of paths, 64 replications per N. Reference: {ref['mean']:,.2f} "
            f"(SE {ref['se']:.2f}, {ref['n']:,} paths). All estimators converge at the Monte Carlo rate N<sup>-1/2</sup>; "
            "variance reduction shifts the line down.</p>"
            + _img(RES / "convergence" / "convergence.png", alt="convergence") + _src("convergence/convergence.txt"))


def _bench_plot(b: dict) -> bytes | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from segfund.bench.run import ENGINE_LABEL

    rows = [r for r in b["rows"] if "median_ms" in r]
    if not rows:
        return None
    scen = list(dict.fromkeys(r["scenario"] for r in rows))
    fig, axes = plt.subplots(1, len(scen), figsize=(5.2 * len(scen), 4.0), sharey=True, squeeze=False)
    marks = dict(numpy_fp64="s", torch_philox_fp32="v", torch_randn_fp32="^", triton_fused_fp32="o", triton_fused_fp32_fastmath="D")
    for ax, s in zip(axes[0], scen):
        for e in ENGINE_LABEL:
            rs = sorted((r for r in rows if r["scenario"] == s and r["engine"] == e), key=lambda r: r["n_paths"])
            if not rs:
                continue
            xs, ys = [r["n_paths"] for r in rs], [r["path_steps_per_s"] for r in rs]
            ax.plot(xs, ys, marker=marks[e], label=ENGINE_LABEL[e].split(" (")[0], lw=1.4, ms=5)
            bad = [(x, y) for x, y, r in zip(xs, ys, rs) if not r["quotable"] and e != "numpy_fp64"]
            if bad:
                ax.scatter(*zip(*bad), s=90, facecolors="none", edgecolors="red", lw=1, zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("paths")
        ax.grid(True, which="both", alpha=0.25)
    axes[0][0].set_ylabel("path-steps per second")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.text(0.01, 0.01, "red ring = GPU row failed the clock gate (not quotable)", fontsize=7, color="red")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def section_bench() -> str:
    b = _load("bench/bench.json")
    if b is None:
        return _missing("Benchmark", "segfund bench")
    from segfund.bench.run import ENGINE_LABEL

    out = [f"<p>{html.escape(str(b['gpu'].get('name')))}, max SM clock {b['gpu'].get('clocks.max.sm')} MHz. "
           "Every GPU engine was checked against the NumPy fp64 reference on the same paths before it was timed. "
           "GPU times are CUDA-event medians of the launch only; a row is quotable only if the SM clock stayed "
           "&ge; 70% of max, the GPU stayed below 87 &deg;C, and the IQR was &le; 5% of the median.</p>"]
    for s, eq in b["checks"].items():
        rs = [r for r in b["rows"] if r["scenario"] == s and "median_ms" in r]
        ref = max((r["path_steps_per_s"] for r in rs if r["engine"] == "numpy_fp64"), default=None)
        rows = []
        for r in rs:
            gate = '<span class="ok">ok</span>' if r["quotable"] else f'<span class="bad">no: {html.escape("; ".join(r["reasons"]))}</span>'
            clk = r.get("clocks", {})
            mhz = f"{clk['sm_mhz_mean']:,.0f}" if "sm_mhz_mean" in clk else "-"
            rows.append([html.escape(ENGINE_LABEL[r["engine"]]), f"{r['n_paths']:,}", f"{r['median_ms']:,.3f}", mhz,
                         f"{r['path_steps_per_s']:.2e}", f"{r['path_steps_per_s'] / ref:,.0f}&times;" if ref else "-", gate])
        checks = ", ".join(f"{ENGINE_LABEL[e].split(' (')[0].split(',')[0]}: "
                           f"{'<span class=ok>pass</span>' if v['passed'] else '<span class=bad>FAIL</span>'}" for e, v in eq.items()
                           if e != "numpy_fp64")
        out.append(f"<h3>{html.escape(s)}</h3><p class='muted'>Equality before timing: {checks}</p>")
        out.append(_table(["Engine", "Paths", "Median ms", "Mean SM MHz", "Path-steps/s", "vs NumPy", "Clock gate"], rows))
    png = _bench_plot(b)
    if png:
        out.append(_img(png=png, alt="benchmark throughput"))
    out.append("<p class='muted'>vs NumPy = throughput ratio against the best NumPy row of the same scenario "
               "(NumPy is only run at small sizes; it is single-threaded and its rows are not clock-gated).</p>")
    out.append(_src("bench/bench.txt"))
    return "\n".join(out)


def section_validation() -> str:
    out = []
    for rel, label, cmd in [("validation/phase1_cpu.json", "Closed forms, CPU fp64", "segfund validate"),
                            ("validation/phase2_gpu.json", "GPU vs CPU on identical scenarios", "segfund validate --gpu")]:
        d = _load(rel)
        if d is None:
            out.append(_missing(label, cmd))
            continue
        g = [c for c in d["checks"] if c.get("gated", True)]
        n_ok = sum(c["passed"] for c in g)
        cls = "ok" if n_ok == len(g) else "bad"
        out.append(f"<p><b>{label}</b>: <span class='{cls}'>{n_ok}/{len(g)} gated checks pass</span> "
                   f"({len(d['checks']) - len(g)} informational rows), {d['meta'].get('n_paths', 0):,} paths, "
                   f"commit <code>{d['meta'].get('git_commit')}</code>.</p>")
        fails = [c["name"] for c in g if not c["passed"]]
        if fails:
            out.append("<ul>" + "".join(f"<li class='bad'>{html.escape(n)}</li>" for n in fails) + "</ul>")
        out.append(_src(rel.replace(".json", ".txt")))
    return "\n".join(out)


def section_nested() -> str:
    d = _load("nested/nested_study.json")
    if d is None:
        return _missing("Nested study", "python scripts/nested_study.py")
    gt = d["summary"]["gt"]
    return (f"<p>CTE(95) of the loss at a 1-year valuation date. Brute-force ground truth: {gt['cte95']['cte']:,.2f} "
            f"(outer-sampling SE {gt['cte95']['se']:.0f}) from {gt['paths']:,} inner paths. "
            "LSMC with a polynomial + put basis matches brute force with a control variate, but does not beat it.</p>"
            + _img(RES / "nested" / "nested.png", alt="nested") + _src("nested/nested_study.txt"))


def build(out: Path | None = None) -> Path:
    out = out or RES / "report" / "report.html"
    sections = [
        ("Value of the guarantee", section_price()),
        ("Tail risk (real-world)", section_tail()),
        ("Convergence", section_convergence()),
        ("Speed: NumPy vs PyTorch vs fused Triton", section_bench()),
        ("Validation", section_validation()),
        ("Nested simulation vs LSMC", section_nested()),
    ]
    body = "\n".join(f"<h2>{t}</h2>\n{s}" for t, s in sections)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>segfund-mc report</title><style>{CSS}</style></head>
<body><main><h1>segfund-mc report</h1>
<p class="muted">Monte Carlo value and tail risk of a segregated fund GMMB/GMDB guarantee. Built from the files under <code>results/</code>.</p>
<p class="warn">Educational model. Not actuarially approved, not regulatory-grade; every assumption is illustrative.</p>
{body}
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
