"""Dynamic lapse.

Policyholders who hold a valuable guarantee (fund well below the guarantee) are
less likely to walk away; those whose guarantee is worthless lapse more. A simple
and common form scales a base lapse rate by moneyness:

    multiplier = clip( (F / G)^k, lo, hi )
    lapse hazard = -ln(1 - base_annual) * multiplier
    step lapse probability = 1 - exp(-hazard * h)

F/G < 1 means the guarantee is in the money -> multiplier < 1 -> fewer lapses.
k controls how strongly people react; lo/hi stop the rate going to 0 or exploding.
The parameters are illustrative: real lapse assumptions come from experience
studies, and Canadian seg funds have no published industry lapse table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicLapse:
    base_annual: float = 0.05
    k: float = 1.0
    lo: float = 0.2
    hi: float = 2.0
    enabled: bool = True

    def __post_init__(self):
        if not 0.0 <= self.base_annual < 1.0:
            raise ValueError("base_annual lapse must be in [0, 1)")
        if not 0.0 <= self.lo <= self.hi:
            raise ValueError("need 0 <= lo <= hi")

    @property
    def base_hazard(self) -> float:
        return -np.log1p(-self.base_annual)

    def multiplier(self, fund, guarantee):
        ratio = np.asarray(fund, dtype=np.float64) / np.asarray(guarantee, dtype=np.float64)
        return np.clip(np.power(ratio, self.k), self.lo, self.hi)

    def step_prob(self, fund, guarantee, h: float):
        if not self.enabled:
            return np.zeros_like(np.asarray(fund, dtype=np.float64))
        return -np.expm1(-self.base_hazard * self.multiplier(fund, guarantee) * h)
