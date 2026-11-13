"""YAML config -> validated model objects.

Unknown keys are an error (a typo like `gmmb_pc` should not silently use a default).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from segfund.curves import FlatCurve, NelsonSiegelCurve, YieldCurve, load_boc_zero_curve
from segfund.models.equity import EquityParams
from segfund.models.hull_white import HullWhite
from segfund.models.lapse import DynamicLapse
from segfund.models.mortality import GompertzMakeham
from segfund.product import Contract, Deposit, ResetRule


@dataclass(frozen=True)
class SimulationSettings:
    n_paths: int
    dt: float
    seed: int
    antithetic: bool = False
    control_variate: str = "none"
    precision: str = "fp64"


@dataclass(frozen=True)
class Model:
    contract: Contract
    hw: HullWhite
    equity: EquityParams
    measure: str  # "risk_neutral" | "real_world"
    lambda_r: float
    mortality: GompertzMakeham | None  # None = mortality off
    lapse: DynamicLapse
    sim: SimulationSettings

    def time_grid(self) -> np.ndarray:
        n = int(round(self.contract.maturity_years / self.sim.dt))
        if abs(n * self.sim.dt - self.contract.maturity_years) > 1e-9:
            raise ValueError("maturity must be a whole number of time steps")
        return np.linspace(0.0, self.contract.maturity_years, n + 1)


def _take(d: dict, allowed: set[str], where: str) -> dict:
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping")
    extra = set(d) - allowed
    if extra:
        raise ValueError(f"unknown key(s) in {where}: {sorted(extra)}")
    return d


def _curve(c: dict, base_dir: Path) -> YieldCurve:
    c = _take(c, {"source", "nelson_siegel", "flat_rate", "boc_csv", "boc_date"}, "economy.curve")
    src = c.get("source", "synthetic")
    if src == "synthetic":
        ns = _take(c.get("nelson_siegel"), {"beta0", "beta1", "beta2", "tau"}, "nelson_siegel")
        return NelsonSiegelCurve(float(ns["beta0"]), float(ns["beta1"]), float(ns["beta2"]), float(ns["tau"]))
    if src == "flat":
        return FlatCurve(float(c["flat_rate"]))
    if src == "boc_csv":
        p = Path(c["boc_csv"])
        if not p.is_absolute():
            p = base_dir / p
        return load_boc_zero_curve(p, c.get("boc_date"))
    raise ValueError(f"unknown curve source {src!r} (synthetic | flat | boc_csv)")


def model_from_dict(cfg: dict[str, Any], base_dir: Path | str = ".") -> Model:
    base_dir = Path(base_dir)
    cfg = _take(cfg, {"contract", "economy", "decrements", "simulation"}, "top level")

    c = _take(
        cfg["contract"],
        {"issue_age", "deposits", "maturity_years", "gmmb_pct", "gmdb_pct", "resets", "fees", "note"},
        "contract",
    )
    fees = _take(c["fees"], {"mer", "guarantee_fee"}, "contract.fees")
    rs = _take(c.get("resets"), {"enabled", "every_years", "min_years_to_maturity"}, "contract.resets")
    contract = Contract(
        issue_age=float(c["issue_age"]),
        deposits=tuple(Deposit(float(d["t"]), float(d["amount"])) for d in c["deposits"]),
        maturity_years=float(c["maturity_years"]),
        gmmb_pct=float(c["gmmb_pct"]),
        gmdb_pct=float(c["gmdb_pct"]),
        mer=float(fees["mer"]),
        guarantee_fee=float(fees["guarantee_fee"]),
        resets=ResetRule(
            enabled=bool(rs.get("enabled", False)),
            every_years=float(rs.get("every_years", 1.0)),
            min_years_to_maturity=float(rs.get("min_years_to_maturity", 0.0)),
        ),
    )

    e = _take(cfg["economy"], {"measure", "curve", "hull_white", "equity"}, "economy")
    measure = e.get("measure", "risk_neutral")
    if measure not in ("risk_neutral", "real_world"):
        raise ValueError("economy.measure must be risk_neutral or real_world")
    hwc = _take(e["hull_white"], {"a", "sigma", "market_price_of_risk"}, "economy.hull_white")
    eqc = _take(e["equity"], {"sigma", "rho_equity_rate", "equity_risk_premium"}, "economy.equity")
    rw = measure == "real_world"
    hw = HullWhite(_curve(e["curve"], base_dir), float(hwc["a"]), float(hwc["sigma"]))
    equity = EquityParams(
        sigma=float(eqc["sigma"]),
        rho=float(eqc["rho_equity_rate"]),
        erp=float(eqc.get("equity_risk_premium", 0.0)) if rw else 0.0,
    )
    lambda_r = float(hwc.get("market_price_of_risk", 0.0)) if rw else 0.0

    dec = _take(cfg.get("decrements"), {"mortality", "lapse"}, "decrements")
    m = _take(dec.get("mortality"), {"model", "A", "B", "c", "enabled", "source"}, "decrements.mortality")
    mortality = None
    if m.get("enabled", True):
        if m.get("model", "gompertz_makeham") != "gompertz_makeham":
            raise ValueError("only gompertz_makeham mortality is implemented")
        mortality = GompertzMakeham(float(m.get("A", 0.00022)), float(m.get("B", 2.7e-6)), float(m.get("c", 1.124)))
    lp = _take(dec.get("lapse"), {"enabled", "base_annual", "k", "lo", "hi"}, "decrements.lapse")
    lapse = DynamicLapse(
        base_annual=float(lp.get("base_annual", 0.0)),
        k=float(lp.get("k", 1.0)),
        lo=float(lp.get("lo", 0.2)),
        hi=float(lp.get("hi", 2.0)),
        enabled=bool(lp.get("enabled", False)),
    )

    s = _take(
        cfg["simulation"],
        {"n_paths", "dt", "seed", "antithetic", "control_variate", "precision"},
        "simulation",
    )
    sim = SimulationSettings(
        n_paths=int(s["n_paths"]),
        dt=float(s["dt"]),
        seed=int(s["seed"]),
        antithetic=bool(s.get("antithetic", False)),
        control_variate=str(s.get("control_variate", "none")),
        precision=str(s.get("precision", "fp64")),
    )
    if sim.n_paths <= 0 or sim.dt <= 0:
        raise ValueError("n_paths and dt must be positive")
    return Model(contract, hw, equity, measure, lambda_r, mortality, lapse, sim)


def load_model(path: str | Path) -> Model:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Relative data paths (e.g. the BoC CSV) resolve against the project root if the
    # config lives in configs/, otherwise against the config's own folder.
    base = path.parent.parent if path.parent.name == "configs" else path.parent
    return model_from_dict(cfg, base)
