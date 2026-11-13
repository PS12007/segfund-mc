"""Initial yield curves.

A curve only has to answer one question for the rest of the code: what is the
time-0 discount factor P(0, t)? Everything else (zero rates, instantaneous
forwards) is derived from that.

Two sources:
  * NelsonSiegelCurve - a smooth synthetic curve with documented parameters.
  * InterpolatedZeroCurve - knots of continuously-compounded zero rates, e.g. the
    Bank of Canada zero-coupon curve (see data/boc/README.md and load_boc_zero_curve).

Conventions: time in years, rates continuously compounded, decimals (0.035 = 3.5%).
"""

from __future__ import annotations

import csv
import datetime as _dt
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class YieldCurve:
    """Base class. Subclasses implement zero_rate(t)."""

    def zero_rate(self, t):  # pragma: no cover - abstract
        raise NotImplementedError

    def discount(self, t):
        """P(0, t) = exp(-z(t) * t)."""
        t = np.asarray(t, dtype=np.float64)
        return np.exp(-self.zero_rate(t) * t)

    def log_discount(self, t):
        t = np.asarray(t, dtype=np.float64)
        return -self.zero_rate(t) * t

    def forward_rate(self, t, h: float = 1e-5):
        """Instantaneous forward f(0, t) = -d ln P(0, t) / dt (central difference).

        Only used for reporting r(t); the simulator never needs it (see hull_white.py).
        """
        t = np.asarray(t, dtype=np.float64)
        lo = np.maximum(t - h, 0.0)
        hi = t + h
        return -(self.log_discount(hi) - self.log_discount(lo)) / (hi - lo)


@dataclass(frozen=True)
class NelsonSiegelCurve(YieldCurve):
    """z(t) = b0 + b1 * g(t) + b2 * (g(t) - exp(-t/tau)),  g(t) = (1 - exp(-t/tau)) / (t/tau).

    Nelson & Siegel (1987). b0 is the long-run level, b0 + b1 the short end,
    b2 controls the hump. The default parameters in configs/ are illustrative,
    not fitted to any market.
    """

    beta0: float
    beta1: float
    beta2: float
    tau: float

    def zero_rate(self, t):
        t = np.asarray(t, dtype=np.float64)
        x = t / self.tau
        # g(x) -> 1 as x -> 0; use a series near 0 to avoid 0/0.
        small = x < 1e-8
        xs = np.where(small, 1.0, x)
        g = np.where(small, 1.0 - x / 2.0, -np.expm1(-xs) / xs)
        return self.beta0 + self.beta1 * g + self.beta2 * (g - np.exp(-x))


@dataclass(frozen=True)
class FlatCurve(YieldCurve):
    rate: float

    def zero_rate(self, t):
        return np.full_like(np.asarray(t, dtype=np.float64), self.rate)


class InterpolatedZeroCurve(YieldCurve):
    """Zero rates given at knots; linear interpolation in z(t)*t (i.e. log-linear in P).

    Linear in z*t means piecewise-constant forward rates between knots, which is
    the simplest arbitrage-free-ish interpolation and keeps P(0,t) monotone when
    forwards are positive. Flat zero-rate extrapolation outside the knots.
    """

    def __init__(self, times, zero_rates, label: str = ""):
        t = np.asarray(times, dtype=np.float64)
        z = np.asarray(zero_rates, dtype=np.float64)
        if t.ndim != 1 or t.shape != z.shape or t.size < 2:
            raise ValueError("need matching 1-D arrays with at least 2 knots")
        if np.any(np.diff(t) <= 0) or t[0] <= 0:
            raise ValueError("knot times must be positive and strictly increasing")
        if not np.all(np.isfinite(z)):
            raise ValueError("zero rates must be finite (drop 'na' knots first)")
        self.times = t
        self.zero_rates = z
        self.label = label

    def zero_rate(self, t):
        t = np.asarray(t, dtype=np.float64)
        zt_knots = self.zero_rates * self.times
        tc = np.clip(t, self.times[0], self.times[-1])
        zt = np.interp(tc, self.times, zt_knots)
        z_inside = np.where(tc > 0, zt / np.where(tc > 0, tc, 1.0), self.zero_rates[0])
        z = np.where(t < self.times[0], self.zero_rates[0], z_inside)
        z = np.where(t > self.times[-1], self.zero_rates[-1], z)
        return z


# --------------------------------------------------------------------------------------
# Bank of Canada zero-coupon curve loader
# --------------------------------------------------------------------------------------

BOC_N_TERMS = 120  # 0.25y .. 30.00y in 0.25y steps, per the BoC page description
BOC_TERMS = np.arange(1, BOC_N_TERMS + 1) * 0.25


def _parse_date(s: str):
    s = s.strip().strip('"')
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def load_boc_zero_curve(path: str | Path, date: str | None = None) -> InterpolatedZeroCurve:
    """Load one curve from a Bank of Canada zero-coupon yield curve CSV.

    Source: https://www.bankofcanada.ca/rates/interest-rates/bond-yield-curves/
    ("Yield Curve Data" form). The page says each row is one curve, values are
    decimals (0.0500 = 5%), terms run 0.25y (column 1) to 30y (column 120), and
    missing values are 'na'.

    The parser is deliberately tolerant about headers because the exact file
    layout was not inspected when this was written: it takes the first field of
    each row as a date and the next 120 fields as yields. If the file does not
    look like that, it raises instead of guessing.

    Compounding: this loader treats the yields as continuously compounded. That
    has not been checked against the BoC methodology note; it is listed in the
    README Limits section until it is.
    """
    path = Path(path)
    want = _dt.date.fromisoformat(date) if date else None
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for rec in csv.reader(fh):
            if not rec:
                continue
            d = _parse_date(rec[0])
            if d is None:
                continue  # header or comment line
            vals = rec[1:]
            if len(vals) < BOC_N_TERMS:
                raise ValueError(
                    f"{path}: row for {d} has {len(vals)} yield fields, expected {BOC_N_TERMS}"
                )
            rows.append((d, vals[:BOC_N_TERMS]))
    if not rows:
        raise ValueError(f"{path}: no dated rows found")
    if want is None:
        d, vals = rows[-1]
    else:
        match = [r for r in rows if r[0] == want]
        if not match:
            raise ValueError(f"{path}: no curve for {want}; have {rows[0][0]} .. {rows[-1][0]}")
        d, vals = match[0]

    z = np.array(
        [math.nan if v.strip().lower() in ("na", "", "n/a") else float(v) for v in vals],
        dtype=np.float64,
    )
    ok = np.isfinite(z)
    if ok.sum() < 2:
        raise ValueError(f"{path}: curve for {d} has fewer than 2 usable terms")
    if np.nanmax(np.abs(z)) > 0.5:
        raise ValueError(f"{path}: yields look like percent, not decimals (max {np.nanmax(z)})")
    return InterpolatedZeroCurve(BOC_TERMS[ok], z[ok], label=f"BoC zero curve {d}")
