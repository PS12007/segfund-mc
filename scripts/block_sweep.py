"""Phase 6: does the fused kernel's launch shape matter? Writes results/bench/block_sweep.{json,txt}.

Sweeps BLOCK (paths per program) and num_warps at a fixed problem size, base
contract, with the clock gate from segfund.bench.timing. Every configuration is
first checked to produce the same per-path output as the default (BLOCK=256,
num_warps=4): the launch shape must not change any number, only the speed.

Usage: python scripts/block_sweep.py [--paths 1048576]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segfund.bench.timing import ClockMonitor, cooldown, gpu_static, time_gpu  # noqa: E402
from segfund.cli import _meta  # noqa: E402
from segfund.config import load_model  # noqa: E402
from segfund.gpu.equality import gpu_info  # noqa: E402
from segfund.gpu.kernels import FusedEngine  # noqa: E402
from segfund.validate import DEFAULT_CONFIG  # noqa: E402

BLOCKS = (64, 128, 256, 512, 1024)
WARPS = (1, 2, 4, 8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1 << 20)
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()
    m = load_model(DEFAULT_CONFIG)
    n, seed = args.paths, args.seed
    n_steps = m.time_grid().size - 1

    ref_eng = FusedEngine(m)
    ref = ref_eng.alloc(n)
    ref_eng.launch(ref, n, seed)
    rows = []
    with ClockMonitor() as mon:
        for block in BLOCKS:
            for warps in WARPS:
                if warps * 32 > block:  # fewer paths than threads: some threads would idle
                    continue
                eng = FusedEngine(m, block=block, num_warps=warps)
                buf = eng.alloc(n)
                eng.launch(buf, n, seed)
                torch.cuda.synchronize()
                same = bool(torch.equal(buf, ref))
                max_abs = float((buf - ref).abs().max())
                cooldown(mon)
                t = time_gpu(f"BLOCK={block} warps={warps}", lambda: eng.launch(buf, n, seed), mon).as_dict()
                row = dict(block=block, num_warps=warps, bitwise_equal_to_default=same, max_abs_diff=max_abs,
                           path_steps_per_s=n * n_steps / (t["median_ms"] * 1e-3), **t)
                rows.append(row)
                q = "ok" if row["quotable"] else "NOT quotable: " + "; ".join(row["reasons"])
                print(f"BLOCK={block:>5} warps={warps}  {row['median_ms']:8.3f} ms  equal={same}  {q}", flush=True)

    best = min((r for r in rows if r["quotable"]), key=lambda r: r["median_ms"], default=None)
    default = next(r for r in rows if r["block"] == 256 and r["num_warps"] == 4)
    lines = [f"Fused kernel launch-shape sweep: base contract, {n:,} paths x {n_steps} steps, seed {seed}",
             f"{'BLOCK':>6} {'warps':>5} {'median ms':>10} {'95% CI ms':>20} {'vs default':>10}  equal  gate"]
    for r in rows:
        ci = f"[{r['ci95_ms'][0]:.3f}, {r['ci95_ms'][1]:.3f}]"
        rel = r["median_ms"] / default["median_ms"]
        gate = "ok" if r["quotable"] else "NOT quotable (" + "; ".join(r["reasons"]) + ")"
        lines.append(f"{r['block']:>6} {r['num_warps']:>5} {r['median_ms']:>10.3f} {ci:>20} {rel:>10.3f}  "
                     f"{'yes' if r['bitwise_equal_to_default'] else 'NO':>5}  {gate}")
    if best:
        lines.append(f"\nfastest quotable: BLOCK={best['block']} num_warps={best['num_warps']} "
                     f"({best['median_ms']:.3f} ms, {best['median_ms'] / default['median_ms']:.3f}x the default's time)")
    text = "\n".join(lines)
    print("\n" + text)
    out = ROOT / "results" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    meta = _meta() | gpu_info() | dict(gpu_static=gpu_static(), n_paths=n, seed=seed)
    (out / "block_sweep.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=2))
    (out / "block_sweep.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n\n" + text + "\n")
    print(f"wrote {out / 'block_sweep'}.json and .txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
