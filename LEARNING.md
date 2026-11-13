# LEARNING.md

Notes written to teach the ideas behind segfund-mc, one section per phase.
This is an educational model. Nothing here is actuarially approved or regulatory-grade.

---

## Phase 1: the CPU fp64 reference model and closed-form validation

### 1.1 The finance: what is being valued

**A segregated fund contract** is a Canadian insurance product. The client's money
goes into an investment fund (held "segregated" from the insurer's general assets),
but the insurer adds guarantees:

- **GMMB (guaranteed minimum maturity benefit):** at maturity (often 10 years) the
  client gets at least, say, 75% or 100% of what they deposited.
- **GMDB (guaranteed minimum death benefit):** if the client dies first, their
  beneficiary gets at least, say, 100% of deposits.

Fund value F, guarantee G. At maturity the client gets max(F, G) = F + max(G − F, 0).
The fund pays the F part. The insurer pays **max(G − F, 0)**, which is exactly the
payoff of a **put option** on the fund with strike G. So the insurer is *short a put*.
It gets paid for that put through a fee, part of the fund's MER (management
expense ratio). In our base contract the MER is 2.25% a year, of which 0.40% is the
guarantee fee kept by the insurer.

It is not a plain exchange-traded put because:
- the maturity is long (10 to 30 years), so interest rates matter a lot;
- the payoff only happens if the client is still there, so it depends on
  **mortality** (death triggers the GMDB instead) and **lapses** (clients who
  surrender walk away from the guarantee, and the insurer loses their future fees);
- lapses depend on the market: when the guarantee is valuable (F well below G),
  rational clients keep it, so lapse rates fall exactly when the insurer is losing.
  That's *dynamic lapse*.
- the fee is paid as a percentage of F, so fee income drops just as claims rise.

**Liability per policy** = PV(claims) − PV(fee income). Positive means the
guarantee costs more than the fees bring in.

**Risk-neutral vs real-world measures (preview, used fully in phases 4 and 5).**
- *Risk-neutral (Q):* the measure used for **pricing**. Every asset grows at the
  risk-free rate on average, so discounted asset prices are martingales. The value
  is "what it would cost to hedge", not a forecast.
- *Real-world (P):* the measure used for **risk measurement**. Equities earn a
  risk premium, and the scenarios are meant to look like plausible futures. Capital
  is set from tail outcomes under P.

The config switches between them with `economy.measure`. Under P we add an
equity risk premium and a market price of interest-rate risk.

**CTE (preview, phase 4).** CTE(α) is the average loss in the worst (100 − α)% of
scenarios. CTE(95) is the mean of the worst 5% of outcomes. Canadian seg fund
capital (OSFI's LICAT framework) uses CTE-type tail measures. Unlike a single
percentile (VaR), CTE looks at how bad the tail is, not just where it starts.

### 1.2 The math

**Hull-White one-factor model**

    dr = (θ(t) − a r) dt + σ dW

The short rate r reverts towards a moving target at speed a, with normal shocks
of size σ. θ(t) is chosen so the model reproduces today's yield curve exactly.

We use the equivalent form r(t) = x(t) + φ(t), where x is a zero-mean
Ornstein-Uhlenbeck process (dx = −a x dt + σ dW) and φ(t) is deterministic. This
makes simulation exact. Over one step of length h, both x and its integral
Y = ∫x dt are Gaussian, with closed-form mean and variance:

    x_t = x_s e^(−ah) + ε1,        Var ε1 = σ²(1 − e^(−2ah)) / (2a)
    Y   = x_s B(h)    + ε2,        B(h) = (1 − e^(−ah)) / a

The discount factor is exp(−∫r), and ∫r = Y + ∫φ. We never discretise the
stochastic differential equation, so the bond-price test has no time-step bias.

Bond price formula: P(t,T) = [P(0,T)/P(0,t)] · exp(½[V(t,T) − V(0,T) + V(0,t)] − B(T−t) x(t)).
Higher x(t) means a lower bond price, as expected.

*Why Hull-White?* It's the simplest model that (1) fits today's curve exactly,
(2) has closed-form bond prices, and (3) allows exact simulation. Its downside is
that rates are Gaussian and can go negative. Alternatives: CIR (no negative
rates, but no exact fit to the curve without a shift), two-factor G2++ (better
curve dynamics), LIBOR/SOFR market models (heavier). Insurers' economic scenario
generators often use multi-factor models. One factor is a deliberate
simplification here.

**Equity fund under stochastic rates**

    dF/F = (r − MER) dt + σ_S dW_S,    corr(dW_S, dW) = ρ

Under Q the fund earns the short rate minus fees. The MER acts like a dividend
yield, money leaking out of the fund.

**Cholesky.** Each step needs three *correlated* normals (ε1, ε2, and the equity
shock). We work out their 3×3 covariance matrix C in closed form (formulas in
`models/equity.py`) and factor C = L Lᵀ, with L lower-triangular. If z is three
independent standard normals, then L z has covariance exactly C. That's all
Cholesky does here: it turns independent noise into noise with the right
correlations. The test `test_step_covariance_against_quadrature` checks every
entry of C against numerical integration.

**Mortality: Gompertz-Makeham.** Hazard μ(x) = A + B cˣ: a constant accident
term plus exponential ageing. Survival over a step integrates the hazard exactly.
Parameters A = 0.00022, B = 2.7e-6, c = 1.124 are the *Standard Ultimate Survival
Model* from Dickson, Hardy & Waters' textbook. Under this model a 55-year-old has
a 3.3% chance of dying within 10 years (`1 - GompertzMakeham().survival(55, 10)` = 0.0334).
It's a teaching model, not a Canadian table.

**Monte Carlo error.** The estimate is the average of N independent path values.
By the central limit theorem its standard error is SE = s/√N, where s is the
standard deviation across paths. A 4× increase in paths halves the error. Every
validation reports z = (MC − exact)/SE. If the code is right, z is roughly
standard normal, so |z| < 4 should essentially always hold (false-alarm rate
about 6 in 100,000 per check).

### 1.3 What was built and how it was checked

The NumPy fp64 engine (`reference/engine.py`) loops over monthly time steps and
keeps only the current state of each path. This is the same shape the GPU kernel
will have. It is the ground truth for everything later.

Validation output: `results/validation/phase1_cpu.txt` (200,000 paths per check,
git commit ecc7e28). **23/23 checks passed.** Examples, copied from that file:

| Check | MC | SE | Exact | z |
|---|---|---|---|---|
| V1 model P(0,T) from step moments vs curve, all monthly T to 30y | max rel. error 1.11e-15 | | | |
| V2a P(0,10) by simulation | 0.71207658 | 0.000245 | 0.71196218 | +0.47 |
| V4 GMMB 100% vs Black-Scholes put | 14503.871 | 38.2 | 14505.141 | −0.03 |
| V5 GMDB+GMMB with mortality, age 70 | 6972.9754 | 22.8 | 6944.7949 | +1.24 |
| V6 GMMB 100% under HW rates, ρ = −0.5 | 11240.101 | 33.6 | 11282.999 | −1.28 |

Notice in V6 how correlation changes the put price: 11,283 at ρ = −0.5, 14,179 at
ρ = 0, 16,635 at ρ = +0.5 (closed-form values). With negative correlation, falling
equities come with rising rates, and those claims get discounted more heavily.

First number for the illustrative base contract (`results/runs/base_cpu_200k.json`,
200k paths, risk-neutral): PV(claims) ≈ 3,003 + 269, PV(fees) ≈ 2,551, so the net
liability is **720.82 (SE 14.37) per $100,000 deposited**. Under these made-up
assumptions the 0.40% fee doesn't quite pay for the guarantee. Don't read
more into it than that. The rate parameters aren't calibrated.

### 1.4 Design decisions and alternatives

- **Exact transitions vs Euler.** Euler would be simpler and is what many
  textbooks show, but it adds a time-step bias that would muddy the
  closed-form comparisons. Exact Gaussian steps cost the same per step.
- **Expected in-force instead of simulated deaths.** Each path carries the
  *expected* fraction of policies still in force. Simulating individual deaths
  would add noise and tell us nothing more about a large portfolio. The cost: we
  can't measure mortality *volatility* risk. That's in Limits.
- **Timing conventions** (documented in `product.py`): deaths and claims at the end
  of the step, lapse decision based on moneyness at the start of the step, fee
  income on start-of-step in-force times end-of-step fund. Other conventions are
  equally defensible. What matters is that they're stated and the closed forms
  match them (V5 uses exactly this discrete convention).
- **Our own PSD Cholesky.** `numpy.linalg.cholesky` refuses singular matrices, and
  the Black-Scholes check (σ_rate = 0) makes C singular. A zero pivot just means
  "this factor has no randomness".
- **Config rejects unknown keys**, so a typo can't silently fall back to a default.

### 1.5 What went wrong, or could have

- **Correlated checks look like evidence but aren't independent.** V2b gave
  z ≈ +2.0 at all three maturities, which looks suspicious. They share the same
  simulated paths, so they're one draw, not three. Re-running with four other
  seeds gave z between −1.06 and +0.82, so it was noise. Lesson: within one
  validation group, don't count passes as independent confirmations.
- **The control variate planned in the brief.** "Closed-form constant-rate put" has
  a known mean only if rates are constant. With Hull-White rates its expectation
  isn't the Black-Scholes value, so using it would bias the estimator. Phase 3 will
  use the Hull-White put formula from V6 instead.
- **The BoC loader is untested on a real file.** It was written from the Bank of
  Canada page's description of the format. It refuses to guess when the layout
  differs, but it needs a real download to verify, plus a check of the compounding
  convention.

### 1.6 Interview questions (with model answers)

1. **Why is a seg fund guarantee a put option, and how is it different from a listed put?**
   At maturity the insurer pays max(G − F, 0), a put on the fund struck at the
   guarantee. It differs because it's very long-dated (so rates matter); it's only
   paid if the client survives and hasn't lapsed, and lapses respond to moneyness;
   the GMDB adds a stream of puts with random exercise at death; and the premium is
   a fee proportional to the fund, so income falls when the put is in the money.

2. **Why fit θ(t) to the initial curve, and how did you check it?**
   So the model prices today's zero-coupon bonds exactly. Otherwise every
   valuation carries a mispricing that has nothing to do with the guarantee. I
   checked it two ways: deterministically, by propagating the per-step means and
   variances to recover P(0,T) (max relative error 1.1e-15), and by Monte Carlo
   (E[exp(−∫r)] within 1 SE of P(0,T) out to 30 years).

3. **What does the Cholesky factor do in your simulator?**
   Each step needs the short rate, its time-integral and the equity shock, which
   are correlated. Their covariance has a closed form. Its Cholesky factor maps
   three independent normals to a vector with exactly that covariance. When rate
   volatility is zero the matrix is singular, so I use a PSD-tolerant version.

4. **How do you know your Monte Carlo is right and not just close?**
   Every check compares to an exact answer in units of standard error. If z stays
   within a few units across checks, maturities, strikes and seeds, there's no
   detectable bias at that precision. I also pick tests that isolate one piece:
   rates only (bond prices), equity with constant rates (Black-Scholes), mortality
   with constant rates (sum of puts), and the full market model (Hull-White put).
   And I use different seeds, because checks that share paths aren't independent.

5. **Risk-neutral vs real-world: which do you use for what?**
   Risk-neutral for the value (the cost of hedging, consistent with market prices).
   Real-world for the distribution of future outcomes used in capital and CTE
   measures. Mixing them is a classic error: a real-world expected value is not a
   price, and risk-neutral tails are not forecasts. Nested simulation (phase 5)
   needs both: real-world outer scenarios, risk-neutral valuation inside each.
