"""Command line: segfund run | validate | bench | nested."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def _meta() -> dict:
    import numpy as np

    return dict(
        timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
    )


def _pick_engine(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def cmd_run(args) -> int:
    """Value one contract: PV (risk-neutral), CTE(70/95) (real-world), delta and rho (CRN)."""
    from segfund import risk
    from segfund.config import load_config, with_measure
    from segfund.estimators import estimate
    from segfund.reference.engine import mean_and_se

    cfg, base_dir = load_config(args.config)
    m_rn = with_measure(cfg, "risk_neutral", base_dir)
    m_rw = with_measure(cfg, "real_world", base_dir)
    engine = _pick_engine(args.engine)
    sim = risk.gpu_sim if engine == "gpu" else risk.cpu_sim
    n = args.paths or m_rn.sim.n_paths
    seed = args.seed if args.seed is not None else m_rn.sim.seed
    anti = m_rn.sim.antithetic
    if anti and n % 2:
        n += 1
    t0 = time.perf_counter()

    r = sim(m_rn, n, seed, anti)
    pv = estimate(r, m_rn)
    # components: same paths, antithetic pairing respected, no control variate
    comps = {}
    for k in ("pv_maturity_claims", "pv_death_claims", "pv_fees"):
        e = estimate(r, m_rn, field=k, antithetic=anti, control="none")
        comps[k] = {"mean": e.mean, "se": e.se}
    rw = sim(m_rw, n, seed + 1, False)  # CTE: independent paths, no antithetic pairing
    ctes = {f"cte{int(a * 100)}": risk.cte(rw.pv_net, a).as_dict() for a in (0.70, 0.95)}
    rw_mean = dict(zip(("mean", "se"), mean_and_se(rw.pv_net)))
    dl = risk.delta(sim, m_rn, n, seed + 2, h=args.delta_bump, antithetic=anti)
    rh = risk.rho(sim, m_rn, n, seed + 3, shift=args.rho_shift_bp * 1e-4, antithetic=anti)
    secs = time.perf_counter() - t0

    S0 = m_rn.contract.initial_deposit
    method = ("antithetic + " if anti else "") + (m_rn.sim.control_variate if m_rn.sim.control_variate != "none" else "plain")
    print(f"config  : {args.config}   engine: {engine}   paths: {n}   seed: {seed}")
    print(f"deposit : {S0:,.0f}   (educational model; not actuarially approved)")
    print(f"\nRisk-neutral value of the guarantee (PV claims - PV fees), estimator: {method}")
    print(f"  PV net liability     {pv.mean:>12,.2f}   SE {pv.se:,.2f}   95% CI [{pv.mean - 1.96 * pv.se:,.2f}, {pv.mean + 1.96 * pv.se:,.2f}]")
    for k, v in comps.items():
        print(f"  {k:<20} {v['mean']:>12,.2f}   SE {v['se']:,.2f}   ({'antithetic, ' if anti else ''}no control variate)")
    print("\nReal-world tail of the PV net liability (independent paths)")
    print(f"  mean                 {rw_mean['mean']:>12,.2f}   SE {rw_mean['se']:,.2f}")
    for k, v in ctes.items():
        print(f"  {k.upper():<20} {v['cte']:>12,.2f}   SE {v['se']:,.2f}   (VaR {v['var']:,.2f}, {v['k']} tail paths)")
    print("\nSensitivities (risk-neutral, common random numbers)")
    print(f"  delta                {dl.value:>12.5f}   SE {dl.se:.5f}   per $ of fund   "
          f"(a 1% market drop changes PV by {-dl.value * 0.01 * S0:+,.2f})")
    print(f"  rho                  {rh.value:>12.4f}   SE {rh.se:.4f}   per +1bp parallel shift")
    print(f"\nwall time {secs:.2f} s (not a benchmark; see segfund bench)")
    if args.out:
        out = dict(meta=_meta(), config=str(args.config), engine=engine, n_paths=n, seed=seed, wall_s=secs,
                   pv=pv.as_dict(), components=comps, real_world_mean=rw_mean, cte=ctes,
                   delta=dl.as_dict(), rho=rh.as_dict())
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    return 0


def _write(out_dir: Path, stem: str, meta: dict, text: str, **payload) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(dict(meta=meta, **payload), indent=2))
    (out_dir / f"{stem}.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n\n" + text + "\n")
    print(f"wrote {out_dir / stem}.json and .txt")


def cmd_validate(args) -> int:
    out_dir = Path(args.out_dir)
    t0 = time.perf_counter()
    if args.gpu:
        from segfund.gpu import equality as E

        recs = E.v7_rng()
        v8, timings = E.v8_engines(args.paths, args.seed)
        recs += v8
        recs += E.v8_phase3(args.paths, args.seed + 1)
        recs += E.v8_phase4(args.paths, args.seed + 2)
        recs += E.v8_phase5(256, 512, args.seed + 3)
        secs = time.perf_counter() - t0
        text = E.format_table(recs)
        text += "\n\nwall-clock (one run each, includes host copies; not a benchmark):\n" + "\n".join(
            f"  {t['variant']}: numpy {t['numpy_fp64_s']:.2f}s | torch unfused {t['torch_unfused_fp32_s']:.2f}s"
            f" | triton fused {t['triton_fused_fp32_s']:.3f}s" for t in timings
        )
        print(text)
        meta = _meta() | E.gpu_info() | dict(n_paths=args.paths, seed=args.seed, wall_s=secs)
        _write(out_dir, "phase2_gpu", meta, text, checks=recs, timings=timings)
    else:
        from segfund.validate import format_table, run_all

        recs = run_all(n_paths=args.paths, seed=args.seed)
        secs = time.perf_counter() - t0
        text = format_table(recs)
        print(text)
        meta = _meta() | dict(n_paths=args.paths, seed=args.seed, wall_s=secs, rng="Philox4x32-10 counter-based")
        _write(out_dir, "phase1_cpu", meta, text, checks=recs)
    print(f"wall time {secs:.1f} s")
    return 0 if all(r["passed"] for r in recs if r.get("gated", True)) else 1


def cmd_not_yet(phase: int):
    def run(args) -> int:
        print(f"`segfund {args.cmd}` is implemented in phase {phase}; not available yet.")
        return 2

    return run


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="segfund", description="Educational seg fund guarantee Monte Carlo engine.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="value one contract")
    r.add_argument("--config", required=True)
    r.add_argument("--paths", type=int, default=None)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--out", default=None, help="write results JSON here")
    r.add_argument("--engine", choices=["auto", "cpu", "gpu"], default="auto")
    r.add_argument("--delta-bump", type=float, default=0.01, help="relative fund bump for delta (default 1%%)")
    r.add_argument("--rho-shift-bp", type=float, default=10.0, help="curve shift for rho in bp (default 10)")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="run closed-form validations")
    v.add_argument("--paths", type=int, default=200_000)
    v.add_argument("--seed", type=int, default=12345)
    v.add_argument("--gpu", action="store_true", help="GPU vs CPU equality checks (V7, V8) instead of closed forms")
    v.add_argument("--out-dir", default=str(ROOT / "results" / "validation"))
    v.set_defaults(func=cmd_validate)

    sub.add_parser("bench", help="benchmarks (phase 6)").set_defaults(func=cmd_not_yet(6))
    sub.add_parser("nested", help="nested simulation + LSMC (phase 5)").set_defaults(func=cmd_not_yet(5))

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
