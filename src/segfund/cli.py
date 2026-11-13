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


def cmd_run(args) -> int:
    from segfund.config import load_model
    from segfund.reference.engine import NumpyNormals, mean_and_se, simulate

    m = load_model(args.config)
    n = args.paths or m.sim.n_paths
    t0 = time.perf_counter()
    r = simulate(m, n, NumpyNormals(args.seed if args.seed is not None else m.sim.seed))
    secs = time.perf_counter() - t0
    res = {}
    for k in ("pv_net", "pv_maturity_claims", "pv_death_claims", "pv_fees"):
        mu, se = mean_and_se(getattr(r, k))
        res[k] = {"mean": mu, "se": se}
    print(f"config      : {args.config}  ({m.measure}, {n} paths, engine=numpy-fp64)")
    print(f"deposit     : {m.contract.initial_deposit:,.0f}")
    for k, v in res.items():
        print(f"{k:<20}: {v['mean']:>12,.2f}  (se {v['se']:,.2f})")
    print(f"wall time   : {secs:.2f} s")
    if args.out:
        out = dict(meta=_meta(), config=str(args.config), n_paths=n, engine="numpy-fp64", wall_s=secs, results=res)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    return 0


def cmd_validate(args) -> int:
    from segfund.validate import format_table, run_all

    t0 = time.perf_counter()
    recs = run_all(n_paths=args.paths, seed=args.seed)
    secs = time.perf_counter() - t0
    text = format_table(recs)
    print(text)
    print(f"wall time {secs:.1f} s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _meta() | dict(n_paths=args.paths, seed=args.seed, wall_s=secs)
    (out_dir / "phase1_cpu.json").write_text(json.dumps(dict(meta=meta, checks=recs), indent=2))
    (out_dir / "phase1_cpu.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n\n" + text + "\n"
    )
    print(f"wrote {out_dir / 'phase1_cpu.json'} and .txt")
    return 0 if all(r["passed"] for r in recs) else 1


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
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="run closed-form validations")
    v.add_argument("--paths", type=int, default=200_000)
    v.add_argument("--seed", type=int, default=12345)
    v.add_argument("--out-dir", default=str(ROOT / "results" / "validation"))
    v.set_defaults(func=cmd_validate)

    sub.add_parser("bench", help="benchmarks (phase 6)").set_defaults(func=cmd_not_yet(6))
    sub.add_parser("nested", help="nested simulation + LSMC (phase 5)").set_defaults(func=cmd_not_yet(5))

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
