"""Phase 6 benchmark: NumPy CPU vs PyTorch unfused vs fused Triton, same scenarios.

Order of operations, per scenario:

  1. Equality first. Every GPU engine is run on the same paths as the NumPy
     fp64 reference and must pass the phase 2 tolerances (V8a/V8b) before it is
     timed. The torch.randn baseline uses different random numbers, so it is
     checked statistically instead: its mean PV must sit within 4 combined SE
     of the reference. An engine that fails is recorded and not timed.
  2. Timing. GPU rows use CUDA events around the launch only (no host copies,
     no allocation of the output buffer for Triton) and the clock gate in
     bench/timing.py. CPU rows use perf_counter, median of a few reps; CPU
     clocks are not monitored, so CPU rows are labelled "not clock-gated".
  3. Throughput is reported as path-steps per second (paths x monthly steps),
     which is linear in both and lets runs of different sizes be compared.
     NumPy is only run at small sizes (it takes ~90 us per path for 10 years);
     speedups vs NumPy are throughput ratios, stated as such.
"""

from __future__ import annotations

import copy
import statistics
import subprocess
import time

import numpy as np

from segfund.config import model_from_dict
from segfund.reference.engine import mean_and_se, simulate
from segfund.rng.philox import PhiloxNormals
from segfund.validate import DEFAULT_CONFIG, _load_cfg

ENGINES = ("numpy_fp64", "torch_philox_fp32", "torch_randn_fp32", "triton_fused_fp32", "triton_fused_fp32_fastmath")
ENGINE_LABEL = {
    "numpy_fp64": "NumPy fp64 (CPU, reference)",
    "torch_philox_fp32": "PyTorch unfused fp32, emulated Philox (bit-exact stream)",
    "torch_randn_fp32": "PyTorch unfused fp32, torch.randn (fair baseline)",
    "triton_fused_fp32": "Triton fused fp32 (default)",
    "triton_fused_fp32_fastmath": "Triton fused fp32 FAST_MATH (biased exp; timing only)",
}


def scenarios() -> dict[str, dict]:
    base = _load_cfg()
    out = {"base 10y (120 steps)": copy.deepcopy(base)}
    c = copy.deepcopy(base)
    c["economy"]["measure"] = "real_world"
    c["economy"]["hull_white"]["market_price_of_risk"] = 0.1
    c["contract"]["maturity_years"] = 20
    out["real-world 20y (240 steps)"] = c
    return out


def gpu_processes() -> list[dict]:
    """Other processes holding a CUDA context (another session can distort timings)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        rows = [ln.split(",") for ln in r.stdout.strip().splitlines() if ln.strip()]
        return [dict(pid=a.strip(), name=b.strip().split("\\")[-1], mem=c.strip()) for a, b, c in rows]
    except Exception:
        return []


def _equality(m, n_steps: int, n_check: int, seed: int) -> dict[str, dict]:
    """Run each GPU engine against the fp64 reference on n_check paths. Returns per-engine verdicts."""
    import torch

    from segfund.gpu.equality import _compare
    from segfund.gpu.kernels import FusedEngine
    from segfund.gpu.torch_engine import simulate_torch

    cpu = simulate(m, n_check, PhiloxNormals(seed))
    out = {}

    def verdict(recs, gated=True):
        bad = [r["name"] for r in recs if r.get("gated", True) and not r["passed"]]
        worst = max(abs(r["diff_over_se"]) for r in recs if r["kind"] == "mean")
        return dict(passed=not bad if gated else True, failed=bad, worst_mean_diff_over_se=worst, n_check=n_check)

    tri = FusedEngine(m).simulate(n_check, seed, antithetic=False)
    out["triton_fused_fp32"] = verdict(_compare("bench", cpu, tri, n_steps, "Triton fused fp32"))
    fast = FusedEngine(m, fast_math=True).simulate(n_check, seed, antithetic=False)
    out["triton_fused_fp32_fastmath"] = verdict(_compare("bench", cpu, fast, n_steps, "FAST_MATH", gated=False), gated=False)
    out["triton_fused_fp32_fastmath"]["note"] = "not gated: FAST_MATH exp is biased (LEARNING 2.5); timed to show what it buys"
    th = simulate_torch(m, n_check, seed, dtype=torch.float32)
    out["torch_philox_fp32"] = verdict(_compare("bench", cpu, th, n_steps, "PyTorch unfused fp32"))

    # torch.randn: independent random numbers -> statistical check only
    rn = simulate_torch(m, n_check, seed, dtype=torch.float32, normals="randn")
    mc, sc = mean_and_se(cpu.pv_net)
    mr, sr = mean_and_se(rn.pv_net)
    z = (mr - mc) / float(np.hypot(sc, sr))
    out["torch_randn_fp32"] = dict(passed=bool(abs(z) <= 4.0), failed=[] if abs(z) <= 4 else ["mean pv_net"],
                                   z_vs_reference=z, mean_ref=mc, mean_randn=mr, n_check=n_check,
                                   note="different random numbers: |mean diff| <= 4 combined SE")
    out["numpy_fp64"] = dict(passed=True, failed=[], note="is the reference")
    return out


def _time_cpu(fn, reps: int) -> dict:
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return dict(n_reps=reps, median_ms=statistics.median(ts), min_ms=min(ts), max_ms=max(ts),
                quotable=False, reasons=["CPU row: clocks not monitored (not clock-gated)"])


def _gpu_fn(engine: str, m, n: int, seed: int):
    import torch

    from segfund.gpu.kernels import FusedEngine
    from segfund.gpu.table import pack
    from segfund.gpu.torch_engine import simulate_torch_tensors

    if engine.startswith("triton"):
        eng = FusedEngine(m, fast_math=engine.endswith("fastmath"))
        buf = eng.alloc(n)
        return lambda: eng.launch(buf, n, seed)
    pm = pack(m)
    normals = "randn" if engine == "torch_randn_fp32" else "philox"
    return lambda: simulate_torch_tensors(m, n, seed, dtype=torch.float32, normals=normals, pm=pm)


def run_bench(sizes: dict[str, list[int]], n_check: int = 1 << 16, seed: int = 20261113, numpy_reps: int = 3,
              log=print) -> dict:
    import torch

    from segfund.bench.timing import ClockMonitor, cooldown, gpu_static, time_gpu

    root = DEFAULT_CONFIG.parents[1]
    procs_before = gpu_processes()
    rows, checks, models = [], {}, {}

    def add_row(sname, engine, n, n_steps, t):
        pss = n * n_steps / (t["median_ms"] * 1e-3)
        row = dict(scenario=sname, engine=engine, n_paths=n, n_steps=n_steps,
                   path_steps_per_s=pss, **{k: v for k, v in t.items() if k != "label"})
        rows.append(row)
        q = "quotable" if row["quotable"] else "NOT quotable: " + "; ".join(row["reasons"])
        log(f"    {engine:<28} n={n:>10,}  {row['median_ms']:>10.3f} ms  {pss:>10.3e} path-steps/s  {q}")

    # 1. equality checks and CPU rows for every scenario (the GPU idles here, so do it all first)
    for sname, cfg in scenarios().items():
        m = model_from_dict(cfg, root)
        n_steps = m.time_grid().size - 1
        models[sname] = (m, n_steps)
        log(f"[{sname}] equality check on {n_check:,} paths ...")
        eq = _equality(m, n_steps, n_check, seed)
        checks[sname] = eq
        for e, v in eq.items():
            log(f"    {e:<28} {'PASS' if v['passed'] else 'FAIL'}  {v.get('note', '')}")
        for engine in ENGINES:
            if not eq[engine]["passed"]:
                rows.append(dict(scenario=sname, engine=engine, skipped="failed equality check", n_steps=n_steps))
        for n in sizes.get("numpy_fp64", []):
            add_row(sname, "numpy_fp64", n, n_steps, _time_cpu(lambda: simulate(m, n, PhiloxNormals(seed)), numpy_reps))

    # 2. GPU rows back to back, so the GPU stays in its full-power state between rows
    with ClockMonitor() as mon:
        for sname, (m, n_steps) in models.items():
            log(f"[{sname}] GPU timing ...")
            for engine in ENGINES[1:]:
                if not checks[sname][engine]["passed"]:
                    continue
                for n in sizes.get(engine, []):
                    fn = _gpu_fn(engine, m, n, seed)
                    fn()
                    torch.cuda.synchronize()  # compile / first-call cost outside the timed window
                    probe = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    probe = time.perf_counter() - probe
                    cool_s = cooldown(mon)
                    t = time_gpu(f"{engine} n={n}", fn, mon, min_reps=20 if probe < 0.1 else 5).as_dict()
                    t["clocks"]["cooldown_seconds"] = cool_s
                    add_row(sname, engine, n, n_steps, t)
                    del fn
                    torch.cuda.empty_cache()
    return dict(gpu=gpu_static(), gpu_processes_before=procs_before, gpu_processes_after=gpu_processes(),
                n_check=n_check, seed=seed, checks=checks, rows=rows)


def summarize(res: dict) -> str:
    lines = []
    others = [p for p in res["gpu_processes_before"] if "python" in p["name"].lower()]
    lines.append(f"GPU: {res['gpu'].get('name')}  max SM clock {res['gpu'].get('clocks.max.sm')} MHz  "
                 f"power limit {res['gpu'].get('power.limit')}")
    lines.append(f"other python processes on the GPU at start: {len(others)} "
                 f"({', '.join(p['pid'] for p in others) or 'none'}; includes this one)")
    for sname in res["checks"]:
        lines.append(f"\n== {sname} ==")
        lines.append("equality before timing (n = %d paths, same streams as the reference):" % res["n_check"])
        for e, v in res["checks"][sname].items():
            extra = ""
            if "worst_mean_diff_over_se" in v:
                extra = f"  worst |mean diff| = {v['worst_mean_diff_over_se']:.1e} SE"
            if "z_vs_reference" in v:
                extra = f"  z = {v['z_vs_reference']:+.2f}"
            lines.append(f"  [{'PASS' if v['passed'] else 'FAIL'}] {ENGINE_LABEL[e]}{extra}")
        rs = [r for r in res["rows"] if r["scenario"] == sname and "median_ms" in r]
        np_rows = [r for r in rs if r["engine"] == "numpy_fp64"]
        ref_pss = max((r["path_steps_per_s"] for r in np_rows), default=None)
        lines.append(f"  {'engine':<52} {'paths':>10} {'median ms':>11} {'95% CI ms':>21} {'path-steps/s':>13} {'x NumPy':>8}  gate")
        for r in rs:
            ci = r.get("ci95_ms")
            ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
            sp = f"{r['path_steps_per_s'] / ref_pss:,.0f}" if ref_pss else "-"
            gate = "ok" if r["quotable"] else "NOT quotable (" + "; ".join(r["reasons"]) + ")"
            lines.append(f"  {ENGINE_LABEL[r['engine']]:<52} {r['n_paths']:>10,} {r['median_ms']:>11.3f} {ci_s:>21} "
                         f"{r['path_steps_per_s']:>13.3e} {sp:>8}  {gate}")
        for r in res["rows"]:
            if r["scenario"] == sname and "skipped" in r:
                lines.append(f"  {ENGINE_LABEL[r['engine']]}: SKIPPED ({r['skipped']})")
    lines.append("\nx NumPy = throughput ratio (path-steps/s) against the best NumPy row of the same scenario.")
    return "\n".join(lines)
