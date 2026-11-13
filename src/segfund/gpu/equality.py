"""Phase 2: GPU vs CPU equality on identical scenarios (V7, V8).

Pass criteria were written down from an fp32 error model *before* the full run
(see LEARNING.md section 2.4), not tuned to whatever came out:

  V7a  Philox words: bitwise identical (NumPy, PyTorch, Triton). Integer arithmetic, no excuse.
  V7b  normals |z_gpu - z_cpu| <= 1e-5. Box-Muller in fp32 with hardware approximate
       log/cos: cos.approx has absolute error ~1e-6, times |rad| <= 5.77 -> ~5e-6.
  V8a  per-path fund, discount factor, in-force: relative error <= 1e-6 * n_steps.
       Each step adds a few fp32 roundings (~6e-8 each) plus the normal's error
       times its loading (~5e-6 * 0.05); summed linearly over 120 steps that is ~5e-5.
  V8b  portfolio mean of each PV component: |mean_gpu - mean_cpu| <= 0.01 * SE.
       A precision error that small cannot change any reported number.
  FAST_MATH kernel rows are reported but not gated: they exist to show the bias
  of hardware-approximate exp (LEARNING.md 2.5).
  Per-path PV differences are reported in dollars (relative error is meaningless
  where claims and fees nearly cancel), not gated.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from segfund.config import model_from_dict
from segfund.gpu.kernels import FusedEngine, philox_normals_gpu
from segfund.gpu.torch_engine import philox4x32_torch, simulate_torch
from segfund.reference.engine import mean_and_se, simulate
from segfund.rng import philox as P
from segfund.rng.philox import PhiloxNormals
from segfund.validate import DEFAULT_CONFIG, _load_cfg

FIELDS_STATE = ("fund_T", "discount_T", "inforce_T", "unit_T")
FIELDS_PV = ("pv_net", "pv_maturity_claims", "pv_death_claims", "pv_fees")


def gpu_info() -> dict:
    import triton

    p = torch.cuda.get_device_properties(0)
    return dict(
        gpu=p.name,
        compute_capability=f"{p.major}.{p.minor}",
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        triton=triton.__version__,
    )


def v7_rng(n: int = 1 << 20) -> list[dict]:
    out = []
    cases = [
        (12345, 0, 0),
        ((0xA4093822 << 32) | 0x299F31D0, 359, 0),
        (2**64 - 1, 7, 2**31 - n - 1),  # largest key, highest path ids the kernel allows
    ]
    for seed, step, off in cases:
        ids = np.arange(off, off + n, dtype=np.uint64)
        ref = np.stack(P.philox4x32(ids, step, P.STREAM_MAIN, 0, seed)).astype(np.int64)
        raw_tr, z_tr = philox_normals_gpu(n, step, seed, off)
        raw_th = torch.stack(philox4x32_torch(torch.arange(off, off + n, device="cuda"), step, 0, 0, seed)).cpu().numpy()
        z_ref = P.normals3(ids, step, seed)
        dz = np.abs(z_tr - z_ref)
        tag = f"seed={seed:#x} step={step} paths=[{off}, {off + n})"
        out.append(dict(name=f"V7a Philox words, Triton == NumPy, {tag}", kind="exact",
                        n_words=int(ref.size), mismatches=int(np.sum(raw_tr != ref)), passed=bool(np.array_equal(raw_tr, ref))))
        out.append(dict(name=f"V7a Philox words, PyTorch == NumPy, {tag}", kind="exact",
                        n_words=int(ref.size), mismatches=int(np.sum(raw_th != ref)), passed=bool(np.array_equal(raw_th, ref))))
        out.append(dict(name=f"V7b normals, Triton fp32 vs NumPy fp64, {tag}", kind="tol",
                        max_abs=float(dz.max()), mean_abs=float(dz.mean()), tol=1e-5, passed=bool(dz.max() <= 1e-5)))
    return out


def _variants(base: dict) -> dict[str, dict]:
    v = {"base (mortality, dynamic lapse, fee)": copy.deepcopy(base)}
    c = copy.deepcopy(base)
    c["contract"]["resets"] = {"enabled": True, "every_years": 1, "min_years_to_maturity": 3}
    c["contract"]["deposits"] = [{"t": 0.0, "amount": 60000}, {"t": 2.0, "amount": 40000}]
    v["resets + second deposit at t=2"] = c
    c = copy.deepcopy(base)
    c["economy"]["measure"] = "real_world"
    c["economy"]["hull_white"]["market_price_of_risk"] = 0.1
    c["contract"]["maturity_years"] = 20
    v["real-world, 20 years (240 steps)"] = c
    return v


def _compare(name: str, cpu, gpu, n_steps: int, engine: str, gated: bool = True) -> list[dict]:
    recs = []
    tol_state = 1e-6 * n_steps
    worst = {}
    for f in FIELDS_STATE:
        a, b = getattr(gpu, f), getattr(cpu, f)
        worst[f] = float(np.max(np.abs(a - b) / np.abs(b)))
    recs.append(dict(name=f"V8a [{engine}] per-path state rel. error, {name}", kind="tol",
                     max_rel={k: v for k, v in worst.items()}, tol=tol_state,
                     passed=bool(max(worst.values()) <= tol_state), gated=gated))
    for f in FIELDS_PV:
        a, b = getattr(gpu, f), getattr(cpu, f)
        mg, _ = mean_and_se(a)
        mc, se = mean_and_se(b)
        d = np.abs(a - b)
        recs.append(dict(
            name=f"V8b [{engine}] {f}, {name}", kind="mean",
            mean_gpu=mg, mean_cpu=mc, se_cpu=se, diff=mg - mc, diff_over_se=(mg - mc) / se if se > 0 else 0.0,
            path_max_abs=float(d.max()), path_mean_abs=float(d.mean()),
            passed=bool(abs(mg - mc) <= 0.01 * se), gated=gated,
        ))
    return recs


def v8_engines(n_paths: int, seed: int) -> tuple[list[dict], list[dict]]:
    base = _load_cfg()
    recs, timings = [], []
    for name, cfg in _variants(base).items():
        m = model_from_dict(cfg, DEFAULT_CONFIG.parents[1])
        n_steps = m.time_grid().size - 1
        t = time.perf_counter()
        cpu = simulate(m, n_paths, PhiloxNormals(seed))
        t_cpu = time.perf_counter() - t
        eng = FusedEngine(m)
        eng.simulate(1024, seed)  # compile / warm up
        torch.cuda.synchronize()
        t = time.perf_counter()
        tri = eng.simulate(n_paths, seed, antithetic=False)
        t_tri = time.perf_counter() - t
        t = time.perf_counter()
        th32 = simulate_torch(m, n_paths, seed, dtype=torch.float32)
        t_th = time.perf_counter() - t
        th64 = simulate_torch(m, n_paths, seed, dtype=torch.float64)
        fast = FusedEngine(m, fast_math=True).simulate(n_paths, seed, antithetic=False)
        recs += _compare(name, cpu, tri, n_steps, "Triton fused fp32")
        recs += _compare(name, cpu, fast, n_steps, "Triton fused fp32 FAST_MATH, not gated", gated=False)
        recs += _compare(name, cpu, th32, n_steps, "PyTorch unfused fp32")
        # fp64 on the GPU: same algorithm, same numbers -> differences should be ~1e-12.
        # This separates "precision" from "logic" differences.
        d64 = max(float(np.max(np.abs(getattr(th64, f) - getattr(cpu, f)) / np.abs(getattr(cpu, f)))) for f in FIELDS_STATE)
        recs.append(dict(name=f"V8c [PyTorch unfused fp64] per-path state rel. error, {name}", kind="tol",
                         max_rel=d64, tol=1e-10, passed=bool(d64 <= 1e-10)))
        timings.append(dict(variant=name, n_paths=n_paths, n_steps=n_steps,
                            numpy_fp64_s=t_cpu, triton_fused_fp32_s=t_tri, torch_unfused_fp32_s=t_th,
                            note="single wall-clock run incl. host copies; NOT a benchmark (phase 6)"))
    return recs, timings


def v8_phase3(n_paths: int, seed: int) -> list[dict]:
    """Antithetic and Kahan variants of the fused kernel vs the fp64 reference (same tolerances)."""
    base = _load_cfg()
    m = model_from_dict(base, DEFAULT_CONFIG.parents[1])
    n_steps = m.time_grid().size - 1
    recs = []
    cpu_anti = simulate(m, n_paths, PhiloxNormals(seed, antithetic_half=n_paths // 2))
    tri_anti = FusedEngine(m).simulate(n_paths, seed, antithetic=True)
    recs += _compare("base, antithetic", cpu_anti, tri_anti, n_steps, "Triton fused fp32")
    th_anti = simulate_torch(m, n_paths, seed, antithetic_half=n_paths // 2)
    recs += _compare("base, antithetic", cpu_anti, th_anti, n_steps, "PyTorch unfused fp32")
    cpu = simulate(m, n_paths, PhiloxNormals(seed))
    kah = FusedEngine(m, kahan=True).simulate(n_paths, seed, antithetic=False)
    recs += _compare("base", cpu, kah, n_steps, "Triton fused fp32 KAHAN")
    return recs


def v8_phase4(n_paths: int, seed: int) -> list[dict]:
    """Bumped models (fund shock, shifted curve) on GPU vs CPU, plus GPU vs CPU delta/rho (informational)."""
    from segfund.config import bumped
    from segfund.risk import cpu_sim, delta, gpu_sim, rho

    base = _load_cfg()
    m = model_from_dict(base, DEFAULT_CONFIG.parents[1])
    n_steps = m.time_grid().size - 1
    recs = []
    for label, mb in [("fund shock +1%", bumped(m, fund_shock=1.01)), ("curve +10bp", bumped(m, rate_shift=0.001))]:
        recs += _compare(f"base, {label}", simulate(mb, n_paths, PhiloxNormals(seed)),
                         FusedEngine(mb).simulate(n_paths, seed, antithetic=False), n_steps, "Triton fused fp32")
    for name, fn, kw in [("delta h=1%", delta, {"h": 0.01}), ("delta h=0.1%", delta, {"h": 0.001}),
                         ("rho 10bp", rho, {"shift": 0.001}), ("rho 1bp", rho, {"shift": 0.0001})]:
        c = fn(cpu_sim, m, n_paths, seed + 1, **kw)
        g = fn(gpu_sim, m, n_paths, seed + 1, **kw)
        recs.append(dict(name=f"V8d [Triton fused fp32] {name} vs NumPy fp64, base, CRN", kind="mean",
                         mean_gpu=g.value, mean_cpu=c.value, se_cpu=c.se, diff=g.value - c.value,
                         diff_over_se=(g.value - c.value) / c.se, path_max_abs=float("nan"), path_mean_abs=float("nan"),
                         passed=bool(abs(g.value - c.value) <= 0.01 * c.se), gated=False))
    return recs


def v8_phase5(n_outer: int, n_inner: int, seed: int) -> list[dict]:
    """Nested simulation on GPU vs CPU with identical streams: outer states and inner values."""
    from segfund import nested as Nn
    from segfund.config import with_measure

    base = _load_cfg()
    root = DEFAULT_CONFIG.parents[1]
    rn, rw = with_measure(base, "risk_neutral", root), with_measure(base, "real_world", root)
    og = Nn.simulate_outer(rw, n_outer, seed, engine="gpu")
    oc = Nn.simulate_outer(rw, n_outer, seed, engine="cpu")
    tol = 1e-6 * og.k0
    worst = {k: float(np.max(np.abs(getattr(og, k) / getattr(oc, k) - 1))) for k in ("F", "D", "l")}
    worst["x_abs"] = float(np.max(np.abs(og.x - oc.x)))
    recs = [dict(name=f"V8e [Triton fused fp32] nested outer state at t0 (real-world, {n_outer} scenarios)", kind="tol",
                 max_rel=worst, tol=tol, passed=bool(max(worst["F"], worst["D"], worst["l"]) <= tol and worst["x_abs"] < 1e-6))]
    ig = Nn.inner_values(rn, oc, n_inner, seed + 1, engine="gpu")
    ic = Nn.inner_values(rn, oc, n_inner, seed + 1, engine="cpu")
    d = ig.v - ic.v
    recs.append(dict(name=f"V8e [Triton fused fp32] nested inner values ({n_outer} x {n_inner}, antithetic + CV)", kind="mean",
                     mean_gpu=float(ig.v.mean()), mean_cpu=float(ic.v.mean()), se_cpu=float(ic.se.mean()),
                     diff=float(d.mean()), diff_over_se=float(d.mean() / ic.se.mean()),
                     path_max_abs=float(np.abs(d).max()), path_mean_abs=float(np.abs(d).mean()),
                     worst_outer_diff_over_se=float(np.max(np.abs(d) / ic.se)),
                     # gate: every outer scenario's value agrees to 1% of its own inner SE
                     passed=bool(np.max(np.abs(d) / ic.se) <= 0.01)))
    return recs


def format_table(recs: list[dict]) -> str:
    lines = []
    for r in recs:
        flag = "PASS" if r["passed"] else "FAIL"
        if not r.get("gated", True):
            flag = "info: " + ("within tol" if r["passed"] else "OUTSIDE tol")
        lines.append(f"[{flag}] {r['name']}")
        if r["kind"] == "exact":
            lines.append(f"        words={r['n_words']}  mismatches={r['mismatches']}")
        elif r["kind"] == "tol":
            mr = r.get("max_rel")
            if isinstance(mr, dict):
                lines.append("        " + "  ".join(f"{k}={v:.2e}" for k, v in mr.items()) + f"  tol={r['tol']:.1e}")
            elif mr is not None:
                lines.append(f"        max_rel={mr:.2e}  tol={r['tol']:.1e}")
            else:
                lines.append(f"        max_abs={r['max_abs']:.2e}  mean_abs={r['mean_abs']:.2e}  tol={r['tol']:.1e}")
        else:
            lines.append(
                f"        gpu={r['mean_gpu']:.6f}  cpu={r['mean_cpu']:.6f}  diff={r['diff']:+.2e} ({r['diff_over_se']:+.1e} SE)"
                f"  per-path |d| max={r['path_max_abs']:.2e} mean={r['path_mean_abs']:.2e}"
            )
    g = [r for r in recs if r.get("gated", True)]
    lines.append(f"\n{sum(r['passed'] for r in g)}/{len(g)} gated checks passed ({len(recs) - len(g)} informational rows not gated)")
    return "\n".join(lines)
