"""Kelly-style funded sizing.

For a position of k micros the daily P&L has mean k*mu and variance k^2*sigma^2 (mu, sigma^2 = per-micro daily mean and variance). Maximising expected log growth of the capital at risk C gives
    k* = C * mu / sigma^2.
Here C is the cushion (balance - MLL floor), mu and sigma^2 are estimated CAUSALLY over the previous `window` sessions (strictly before the decision session).
The funded engine sizes as  k = round(a * cushion / unit[t]),  so the Kelly rule is a unit stream  unit = sigma^2 / mu  (a = f = the multiple of full Kelly);
sessions with a non-positive trailing mean (or no history yet) get a huge unit, i.e. 1 micro.
"""
from __future__ import annotations
import numpy as np

HUGE = 1e12


def kelly_unit(pnl: np.ndarray, window: int = 120, min_obs: int = 60) -> np.ndarray:
    x = np.asarray(pnl, dtype=float); n = len(x); out = np.full(n, HUGE)
    cs = np.concatenate([[0.0], np.cumsum(x)]); cq = np.concatenate([[0.0], np.cumsum(x * x)])
    for t in range(n):
        lo = max(0, t - window)
        m = t - lo
        if m < min_obs:
            continue
        mu = (cs[t] - cs[lo]) / m
        var = (cq[t] - cq[lo]) / m - mu * mu
        if mu > 0 and var > 0:
            out[t] = var / mu
    return out
