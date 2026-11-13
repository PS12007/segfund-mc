# segfund-mc

An educational Monte Carlo engine for valuing segregated fund (variable annuity)
guarantees, GMMB and GMDB, under Hull-White rates, a correlated equity fund,
Gompertz-Makeham mortality and dynamic lapses. Written in Python 3.12; GPU
kernels in Triton are planned, on native Windows.

**This is a teaching model. It is not actuarially approved, not regulatory-grade,
and its assumptions are illustrative.**

## Status

Phase 1 of 6 is done: CPU fp64 reference model and closed-form validation.
GPU kernels, variance reduction, risk metrics, nested simulation/LSMC and
benchmarks are not built yet. The Results section stays empty until phase 6,
except for the validation output below.

## Validation (phase 1)

`segfund validate` runs 23 checks against closed forms at 200,000 paths each.
Raw output: [`results/validation/phase1_cpu.txt`](results/validation/phase1_cpu.txt).
Result at commit ecc7e28: 23/23 passed. Monte Carlo checks pass when
|MC − exact| < 4 standard errors.

## Setup (Windows)

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe numpy PyYAML matplotlib pytest
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\segfund.exe validate
.venv\Scripts\segfund.exe run --config configs\base.yaml
```

## Limits (so far)

- Hull-White a and σ are assumptions, not calibrated (no free Canadian swaption vol data).
- One-factor rates: Gaussian, can go negative, all curve moves perfectly correlated.
- Gompertz-Makeham textbook mortality, no improvement, no sex split; expected
  decrements only, so mortality volatility risk is not captured.
- Lapse function and parameters are illustrative; there is no experience study behind them.
- Resets do not restart the maturity date; no withdrawals, no sales charges.
- BoC curve loader is not yet verified on a real download; compounding convention assumed continuous.
