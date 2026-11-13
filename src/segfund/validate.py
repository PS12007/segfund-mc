"""Phase 1 validations: fp64 reference model vs closed forms.

Each check returns a dict with the Monte Carlo estimate, its standard error, the
exact value, z = (mc - exact) / se, and pass/fail. Monte Carlo checks pass when
|z| < Z_TOL; deterministic checks pass on an absolute/relative tolerance.

Run with `segfund validate`; raw output goes to results/validation/.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from segfund.config import Model, model_from_dict
from segfund.models.equity import build_step_table
from segfund.reference import closed_form as cf
from segfund.reference.engine import mean_and_se, simulate, simulate_rates
from segfund.rng.philox import PhiloxNormals

Z_TOL = 4.0
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"


def _load_cfg(path=DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _variant(base: dict, **edits) -> Model:
    """Copy the base config and apply dotted-path edits, e.g. economy__hull_white__sigma=0."""
    cfg = copy.deepcopy(base)
    for key, val in edits.items():
        node = cfg
        parts = key.split("__")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return model_from_dict(cfg, DEFAULT_CONFIG.parents[1])


def _plain(base: dict, **edits) -> Model:
    """No mortality, no lapse, no guarantee fee: isolates the market model."""
    e = dict(
        decrements__mortality__enabled=False,
        decrements__lapse__enabled=False,
        contract__fees__guarantee_fee=0.0,
    )
    e.update(edits)
    return _variant(base, **e)


def _mc_record(name, mc, se, exact, **extra):
    if se > 0:
        z = (mc - exact) / se
    else:
        z = 0.0 if mc == exact else float("inf")
    return dict(name=name, kind="mc", mc=mc, se=se, exact=exact, z=z, passed=bool(abs(z) < Z_TOL), **extra)


# ------------------------------------------------------------------------------------
# V1: deterministic - the per-step moment table reproduces the input curve exactly
# ------------------------------------------------------------------------------------


def v1_curve_fit(base: dict, horizon: float = 30.0) -> dict:
    m = _variant(base)
    dt = m.sim.dt
    times = np.arange(int(round(horizon / dt)) + 1) * dt
    tab = build_step_table(m.hw, m.equity, times, m.contract.mer)
    mean = np.zeros(2)  # (x, I)
    cov = np.zeros((2, 2))
    worst = 0.0
    for k in range(tab.n_steps):
        A = np.array([[tab.decay[k], 0.0], [tab.Bh[k], 1.0]])
        C = (tab.L[k] @ tab.L[k].T)[:2, :2]
        mean = A @ mean + np.array([0.0, tab.phi_int[k]])
        cov = A @ cov @ A.T + C
        p_model = np.exp(-mean[1] + 0.5 * cov[1, 1])
        p_curve = float(m.hw.curve.discount(times[k + 1]))
        worst = max(worst, abs(p_model / p_curve - 1.0))
    tol = 1e-12
    return dict(
        name="V1 curve fit: model P(0,T) from step moments vs input curve, T = 1/12 .. 30",
        kind="deterministic",
        max_rel_error=worst,
        tol=tol,
        passed=bool(worst < tol),
    )


# ------------------------------------------------------------------------------------
# V2: Hull-White bond prices by Monte Carlo
# ------------------------------------------------------------------------------------


def v2_zcb(base: dict, n_paths: int, seed: int) -> list[dict]:
    m = _variant(base)
    dt = m.sim.dt
    out = []
    maturities = [1, 2, 5, 10, 20, 30]
    steps = {int(round(T / dt)): T for T in maturities}
    _, rec = simulate_rates(m, n_paths, steps.keys(), PhiloxNormals(seed), horizon=30.0)
    for k, T in steps.items():
        _, I = rec[k]
        mc, se = mean_and_se(np.exp(-I))
        out.append(_mc_record(f"V2a E[exp(-int_0^T r)] vs P(0,T), T={T}", mc, se, float(m.hw.curve.discount(T))))

    # V2b: E[ exp(-int_0^5 r) * P(5, T | x_5) ] = P(0, T) checks the closed-form P(t,T)
    k5 = int(round(5 / dt))
    _, rec = simulate_rates(m, n_paths, [k5], PhiloxNormals(seed + 1), horizon=5.0)
    x5, I5 = rec[k5]
    for T in [10, 20, 30]:
        mc, se = mean_and_se(np.exp(-I5) * m.hw.zcb(5.0, T, x5))
        out.append(_mc_record(f"V2b E[D(5) P(5,T|x5)] vs P(0,T), T={T}", mc, se, float(m.hw.curve.discount(T))))

    # V2c: start at t=5 from a fixed state and price P(5, T | x0) by inner simulation
    for x0 in (-0.01, 0.01):
        for T in [10, 30]:
            kT = int(round((T - 5) / dt))
            _, rec = simulate_rates(m, n_paths, [kT], PhiloxNormals(seed + 2), t0=5.0, x0=x0, horizon=T)
            _, I = rec[kT]
            mc, se = mean_and_se(np.exp(-I))
            out.append(
                _mc_record(f"V2c inner MC P(5,{T}|x5={x0:+.2f}) vs closed form", mc, se, float(m.hw.zcb(5.0, T, x0)))
            )
    return out


# ------------------------------------------------------------------------------------
# V3: martingale and fee checks on the full engine
# ------------------------------------------------------------------------------------


def v3_martingale(base: dict, n_paths: int, seed: int) -> list[dict]:
    m = _plain(base, contract__fees__guarantee_fee=0.0040)
    c = m.contract
    r = simulate(m, n_paths, PhiloxNormals(seed))
    T = c.maturity_years
    F0 = c.initial_deposit
    mc, se = mean_and_se(r.discount_T * r.fund_T)
    out = [_mc_record("V3a E[D(T) F(T)] vs F0 exp(-mer T)", mc, se, F0 * np.exp(-c.mer * T))]
    # With no decrements, PV(fees) = sum_k fee * h * F0 * exp(-mer t_{k+1}) exactly.
    times = m.time_grid()
    exact_fee = float(np.sum(c.guarantee_fee * np.diff(times) * F0 * np.exp(-c.mer * times[1:])))
    mc, se = mean_and_se(r.pv_fees)
    out.append(_mc_record("V3b PV(fee income), no decrements, vs closed form", mc, se, exact_fee))
    return out


# ------------------------------------------------------------------------------------
# V4: constant rate, no decrements -> Black-Scholes put
# ------------------------------------------------------------------------------------


def v4_black_scholes(base: dict, n_paths: int, seed: int, rate: float = 0.03) -> list[dict]:
    out = []
    for pct in (0.75, 1.00, 1.25):
        m = _plain(
            base,
            economy__curve={"source": "flat", "flat_rate": rate},
            economy__hull_white__sigma=0.0,
            contract__gmmb_pct=pct,
        )
        c = m.contract
        r = simulate(m, n_paths, PhiloxNormals(seed))
        mc, se = mean_and_se(r.pv_net)
        exact = float(cf.bs_put(c.initial_deposit, pct * c.initial_deposit, c.maturity_years, rate, c.mer, m.equity.sigma))
        out.append(_mc_record(f"V4 GMMB {int(pct * 100)}% vs Black-Scholes put (r={rate}, q=MER)", mc, se, exact))
    return out


# ------------------------------------------------------------------------------------
# V5: constant rate + mortality (no lapse) -> sum of BS puts weighted by deaths
# ------------------------------------------------------------------------------------


def v5_gmdb(base: dict, n_paths: int, seed: int, rate: float = 0.03) -> dict:
    m = _plain(
        base,
        economy__curve={"source": "flat", "flat_rate": rate},
        economy__hull_white__sigma=0.0,
        decrements__mortality__enabled=True,
        contract__issue_age=70,  # older age so the GMDB part is material
    )
    c = m.contract
    times = m.time_grid()
    q = m.mortality.step_death_probs(c.issue_age, times)
    F0 = c.initial_deposit
    exact = cf.gmdb_gmmb_constant_rate(
        F0, c.gmdb_pct * F0, c.gmmb_pct * F0, c.maturity_years, rate, c.mer, m.equity.sigma, q, times
    )
    r = simulate(m, n_paths, PhiloxNormals(seed))
    mc, se = mean_and_se(r.pv_net)
    return _mc_record("V5 GMDB+GMMB with mortality (age 70) vs semi-analytic sum of BS puts", mc, se, exact)


# ------------------------------------------------------------------------------------
# V6: stochastic rates, no decrements -> closed-form put under Hull-White + GBM
# ------------------------------------------------------------------------------------


def v6_hw_put(base: dict, n_paths: int, seed: int) -> list[dict]:
    out = []
    for rho in (-0.5, 0.0, 0.5):
        m = _plain(base, contract__gmmb_pct=1.0, economy__equity__rho_equity_rate=rho, economy__hull_white__sigma=0.015)
        c = m.contract
        r = simulate(m, n_paths, PhiloxNormals(seed))
        mc, se = mean_and_se(r.pv_net)
        exact = float(cf.hw_gbm_put(m.hw, m.equity, c.initial_deposit, c.initial_deposit, c.maturity_years, c.mer))
        out.append(_mc_record(f"V6 GMMB 100% under HW rates (sigma_r=1.5%, rho={rho:+.1f}) vs closed form", mc, se, exact))
    return out


# ------------------------------------------------------------------------------------
# V10: variance reduction must not bias the answer
# ------------------------------------------------------------------------------------


def v10_variance_reduction(base: dict, n_paths: int, seed: int) -> list[dict]:
    """Antithetic and control-variate estimators vs closed forms.

    Each record also carries the variance-reduction factor (VRF) against the
    plain estimator on the same paths: VRF = (SE_plain / SE_method)^2, i.e. how
    many times more plain paths you'd need for the same error. For antithetic
    the plain SE is taken from the same 2 x (N/2) paths treated as independent.
    """
    from segfund.estimators import estimate

    out = []
    # (a) stochastic rates, no decrements: antithetic only (the control would be the payoff itself)
    m = _plain(base, contract__gmmb_pct=1.0, economy__equity__rho_equity_rate=-0.5, economy__hull_white__sigma=0.015)
    c = m.contract
    exact = float(cf.hw_gbm_put(m.hw, m.equity, c.initial_deposit, c.initial_deposit, c.maturity_years, c.mer))
    r = simulate(m, n_paths, PhiloxNormals(seed, antithetic_half=n_paths // 2))
    plain = estimate(r, m, antithetic=False, control="none")
    anti = estimate(r, m, antithetic=True, control="none")
    out.append(_mc_record("V10a antithetic: GMMB 100% under HW rates (rho=-0.5) vs closed form",
                          anti.mean, anti.se, exact, vrf=(plain.se / anti.se) ** 2))

    # (b-d) constant rate + mortality (V5 setup): control variate, antithetic, both
    rate = 0.03
    m = _plain(
        base,
        economy__curve={"source": "flat", "flat_rate": rate},
        economy__hull_white__sigma=0.0,
        decrements__mortality__enabled=True,
        contract__issue_age=70,
    )
    c = m.contract
    times = m.time_grid()
    q = m.mortality.step_death_probs(c.issue_age, times)
    F0 = c.initial_deposit
    exact = cf.gmdb_gmmb_constant_rate(F0, c.gmdb_pct * F0, c.gmmb_pct * F0, c.maturity_years, rate, c.mer,
                                      m.equity.sigma, q, times)
    r_plain = simulate(m, n_paths, PhiloxNormals(seed + 1))
    r_anti = simulate(m, n_paths, PhiloxNormals(seed + 1, antithetic_half=n_paths // 2))
    base_se = estimate(r_plain, m, antithetic=False, control="none").se
    for label, res, anti_flag, ctrl in [
        ("control variate (HW put)", r_plain, False, "hw_put"),
        ("antithetic", r_anti, True, "none"),
        ("antithetic + control variate", r_anti, True, "hw_put"),
    ]:
        e = estimate(res, m, antithetic=anti_flag, control=ctrl)
        out.append(_mc_record(f"V10 {label}: GMDB+GMMB with mortality (age 70) vs semi-analytic",
                              e.mean, e.se, exact, vrf=(base_se / e.se) ** 2, beta=e.beta))
    return out


def run_all(n_paths: int = 200_000, seed: int = 12345, config=DEFAULT_CONFIG) -> list[dict]:
    base = _load_cfg(config)
    recs = [v1_curve_fit(base)]
    recs += v2_zcb(base, n_paths, seed)
    recs += v3_martingale(base, n_paths, seed + 10)
    recs += v4_black_scholes(base, n_paths, seed + 20)
    recs.append(v5_gmdb(base, n_paths, seed + 30))
    recs += v6_hw_put(base, n_paths, seed + 40)
    recs += v10_variance_reduction(base, n_paths, seed + 50)
    return recs


def format_table(recs: list[dict]) -> str:
    lines = []
    for r in recs:
        flag = "PASS" if r["passed"] else "FAIL"
        if r["kind"] == "mc":
            lines.append(
                f"[{flag}] {r['name']}\n        mc={r['mc']:.8g}  se={r['se']:.3g}  exact={r['exact']:.8g}  z={r['z']:+.2f}"
                + (f"  VRF={r['vrf']:.2f}" if "vrf" in r else "")
                + (f"  beta={r['beta']:.3f}" if r.get("beta") is not None else "")
            )
        else:
            lines.append(f"[{flag}] {r['name']}\n        max_rel_error={r['max_rel_error']:.3g}  tol={r['tol']:.0e}")
    n_pass = sum(r["passed"] for r in recs)
    lines.append(f"\n{n_pass}/{len(recs)} passed (MC checks pass when |z| < {Z_TOL})")
    return "\n".join(lines)
