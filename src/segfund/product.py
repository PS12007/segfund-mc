"""Segregated fund contract definition.

A single policy, normalised per initial policyholder. All amounts in dollars.

Cash flows the insurer sees (all assumptions documented, see LEARNING.md):
  * fee income: the guarantee_fee part of the MER, received at the end of each
    step on the fund value of policies in force at the start of the step
    (fee_k = l_k * guarantee_fee * h * F_{k+1}).
  * death claim (GMDB): for deaths during a step, paid at the end of the step,
    max(G_D - F, 0) per death.
  * maturity claim (GMMB): at maturity, survivors get max(G_M - F_T, 0) each.
  * lapses: the policyholder takes the fund value; the insurer pays nothing and
    loses future fees.

Simplifications versus real Canadian seg fund products: resets do not restart
the 10-year term, there are no partial withdrawals, no deferred sales charges,
and deposits after issue are made by every survivor in the same amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Deposit:
    t: float
    amount: float


@dataclass(frozen=True)
class ResetRule:
    enabled: bool = False
    every_years: float = 1.0
    # A reset at time t is only allowed if at least this many years remain to maturity.
    min_years_to_maturity: float = 0.0


@dataclass(frozen=True)
class Contract:
    issue_age: float
    deposits: tuple[Deposit, ...]
    maturity_years: float
    gmmb_pct: float
    gmdb_pct: float
    mer: float
    guarantee_fee: float
    resets: ResetRule = field(default_factory=ResetRule)

    def __post_init__(self):
        if not self.deposits:
            raise ValueError("contract needs at least one deposit")
        if self.deposits[0].t != 0.0:
            raise ValueError("first deposit must be at t = 0")
        ts = [d.t for d in self.deposits]
        if ts != sorted(ts) or len(set(ts)) != len(ts):
            raise ValueError("deposit times must be strictly increasing")
        if ts[-1] >= self.maturity_years:
            raise ValueError("deposits must be before maturity")
        if any(d.amount <= 0 for d in self.deposits):
            raise ValueError("deposit amounts must be positive")
        if self.maturity_years <= 0:
            raise ValueError("maturity must be positive")
        if not (0 <= self.gmmb_pct <= 2 and 0 <= self.gmdb_pct <= 2):
            raise ValueError("guarantee percentages look wrong (expect e.g. 0.75 or 1.00)")
        if not 0 <= self.guarantee_fee <= self.mer < 1:
            raise ValueError("need 0 <= guarantee_fee <= mer < 1 (the guarantee fee is part of the MER)")
        if self.issue_age < 0 or self.issue_age > 110:
            raise ValueError("issue age out of range")

    @property
    def initial_deposit(self) -> float:
        return self.deposits[0].amount
