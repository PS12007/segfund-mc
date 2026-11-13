"""Gompertz-Makeham mortality.

Force of mortality (hazard) at age x:   mu(x) = A + B c^x

  * A: age-independent deaths (accidents)
  * B c^x: exponential ageing (Gompertz's law)

Survival from age x to x + h integrates the hazard exactly:

    p = exp( -A h - B c^x (c^h - 1) / ln c )

Default parameters A = 0.00022, B = 2.7e-6, c = 1.124 are the "Standard Ultimate
Survival Model" from Dickson, Hardy & Waters, "Actuarial Mathematics for Life
Contingent Risks" (Cambridge). That is a textbook model for teaching, not a
Canadian table: it has no sex split, no select period and no mortality
improvement. Real insurers use CIA tables (e.g. CIP2014) with improvement scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GompertzMakeham:
    A: float = 0.00022
    B: float = 2.7e-6
    c: float = 1.124

    def __post_init__(self):
        if self.A < 0 or self.B < 0 or self.c <= 1.0:
            raise ValueError("need A >= 0, B >= 0, c > 1")

    def hazard(self, age):
        return self.A + self.B * np.power(self.c, np.asarray(age, dtype=np.float64))

    def survival(self, age, h):
        """Probability of surviving h years from exact age `age`."""
        age = np.asarray(age, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        lnc = np.log(self.c)
        cum = self.A * h + self.B * np.power(self.c, age) * np.expm1(h * lnc) / lnc
        return np.exp(-cum)

    def step_death_probs(self, issue_age: float, times: np.ndarray) -> np.ndarray:
        """q_k = P(die in [t_k, t_{k+1}) | alive at t_k), one per step."""
        times = np.asarray(times, dtype=np.float64)
        return 1.0 - self.survival(issue_age + times[:-1], np.diff(times))
