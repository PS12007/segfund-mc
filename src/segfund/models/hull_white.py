"""One-factor Hull-White short-rate model, fitted exactly to an initial curve.

    dr(t) = (theta(t) - a r(t)) dt + sigma dW(t)

We use the standard shifted form (Brigo & Mercurio, "Interest Rate Models", sec. 3.3):

    r(t) = x(t) + phi(t),    dx = -a x dt + sigma dW,    x(0) = 0
    phi(t) = f(0,t) + sigma^2 / (2 a^2) * (1 - exp(-a t))^2

Choosing phi this way is the same as choosing theta(t) so that the model's
zero-coupon prices equal the input curve's P(0,T) for every T. The advantage of
the x-form: x is a plain Ornstein-Uhlenbeck process, so over any step its value and
its time-integral are jointly Gaussian with closed-form moments. We simulate those
moments exactly - there is no Euler discretisation error in the rate model.

We never need f(0,t) or theta(t) to simulate. The deterministic part of the
integrated rate only needs ln P(0,t):

    int_s^t phi(u) du = ln P(0,s) - ln P(0,t)
                        + sigma^2/(2a^2) * [ h - 2(e^{-as} - e^{-at})/a + (e^{-2as} - e^{-2at})/(2a) ]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segfund.curves import YieldCurve


def B(a: float, tau):
    """B(tau) = (1 - exp(-a tau)) / a. Sensitivity of P(t, t+tau) to x(t)."""
    tau = np.asarray(tau, dtype=np.float64)
    return -np.expm1(-a * tau) / a


def var_int_x(a: float, sigma: float, tau):
    """Var[ int_t^{t+tau} x(u) du | x(t) ]  (the 'V' function of Brigo-Mercurio)."""
    tau = np.asarray(tau, dtype=np.float64)
    return (sigma**2 / a**2) * (tau - 2.0 * B(a, tau) + B(2.0 * a, tau))


@dataclass(frozen=True)
class HullWhite:
    curve: YieldCurve
    a: float
    sigma: float

    def __post_init__(self):
        if self.a <= 0:
            raise ValueError("Hull-White mean reversion a must be > 0")
        if self.sigma < 0:
            raise ValueError("Hull-White sigma must be >= 0")

    # ---- deterministic pieces -------------------------------------------------------

    def phi(self, t):
        t = np.asarray(t, dtype=np.float64)
        return self.curve.forward_rate(t) + self.sigma**2 / (2 * self.a**2) * np.expm1(-self.a * t) ** 2

    def phi_integral(self, s, t):
        """int_s^t phi(u) du, exactly, from ln P(0,.)."""
        s = np.asarray(s, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        a = self.a
        conv = (t - s) - 2.0 * (np.exp(-a * s) - np.exp(-a * t)) / a + (
            np.exp(-2 * a * s) - np.exp(-2 * a * t)
        ) / (2 * a)
        return self.curve.log_discount(s) - self.curve.log_discount(t) + self.sigma**2 / (2 * a**2) * conv

    def short_rate(self, t, x):
        return np.asarray(x) + self.phi(t)

    # ---- closed-form bond price --------------------------------------------------------

    def zcb(self, t, T, x_t):
        """P(t, T) given x(t) (Brigo-Mercurio, G1++ / Hull-White section, x-form).

        P(t,T) = P(0,T)/P(0,t) * exp( 0.5 [V(t,T) - V(0,T) + V(0,t)] - B(T-t) x(t) )
        """
        t = np.asarray(t, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        a, s = self.a, self.sigma
        lnP = self.curve.log_discount(T) - self.curve.log_discount(t)
        adj = 0.5 * (var_int_x(a, s, T - t) - var_int_x(a, s, T) + var_int_x(a, s, t))
        return np.exp(lnP + adj - B(a, T - t) * np.asarray(x_t, dtype=np.float64))

    # ---- exact one-step moments of (x, int x) ---------------------------------------------

    def step_moments(self, h: float):
        """Moments of (x_t, Y) given x_s, where Y = int_s^t x du and h = t - s.

        x_t = x_s * e^{-ah} + eps1,     Y = x_s * B(h) + eps2
        Returns decay, B(h), Var(eps1), Var(eps2), Cov(eps1, eps2).
        """
        a, s = self.a, self.sigma
        decay = float(np.exp(-a * h))
        Bh = float(B(a, h))
        v11 = s**2 * float(B(2 * a, h))  # sigma^2 (1 - e^{-2ah}) / (2a)
        v22 = float(var_int_x(a, s, h))
        v12 = s**2 / (2 * a**2) * float(np.expm1(-a * h)) ** 2
        return decay, Bh, v11, v22, v12
