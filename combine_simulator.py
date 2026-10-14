"""Vectorised Monte Carlo simulation of Combine attempts and XFA extraction paths.

Daily P&L is modelled as Brownian motion over the session: close ~ N(mu, sigma).
The intraday minimum is sampled exactly from the Brownian-bridge law given the
close, so the intraday DLL is triggered by the true intraday low, not just the close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pricing_config import EngineConfig, FeePath


def simulate_single_day(raw_pnl: float, intraday_dll: Optional[float] = None,
                        intraday_min: Optional[float] = None) -> float:
    """Apply the intraday DLL to one session.

    If the intraday low (defaults to the close when not supplied) touches the DLL,
    positions are liquidated at the DLL and the session is locked.
    """
    if intraday_dll is None:
        return float(raw_pnl)
    low = raw_pnl if intraday_min is None else min(intraday_min, raw_pnl)
    return float(intraday_dll) if low <= intraday_dll else float(raw_pnl)


def _draw_days(rng: np.random.Generator, n: int, mu: float, sigma: float,
               intraday_dll: Optional[float], friction: float) -> np.ndarray:
    close = rng.normal(mu, sigma, n)
    if intraday_dll is not None:
        # Minimum of a Brownian bridge from 0 to `close` with variance sigma^2:
        # P(min <= m | close) = exp(-2 m (m - close) / sigma^2), m <= min(0, close).
        u = rng.random(n)
        low = 0.5 * (close - np.sqrt(close ** 2 - 2.0 * sigma ** 2 * np.log(u)))
        close = np.where(low <= intraday_dll, intraday_dll, close)
    return close - friction


@dataclass
class AttemptResults:
    tau: np.ndarray        # trading days to termination (1-based)
    passed: np.ndarray     # bool
    truncated: np.ndarray  # bool: hit max_days without terminating (counted as fail)


def simulate_attempts(cfg: EngineConfig, path: FeePath, n: int,
                      rng: np.random.Generator) -> AttemptResults:
    """Simulate `n` independent Combine attempts, each starting fresh on day 0."""
    rules = cfg.combine_rules
    mu = cfg.market_process.mu_daily_gross
    sigma = cfg.market_process.sigma_daily
    fric = cfg.friction.daily_cost

    pnl = np.zeros(n)
    peak = np.zeros(n)          # highest EOD net P&L (Combine MLL trails it with no lock)
    max_day = np.zeros(n)       # largest single winning day
    tau = np.full(n, rules.max_days, dtype=np.int64)
    passed = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    for day in range(1, rules.max_days + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        x = _draw_days(rng, idx.size, mu, sigma, path.intraday_dll, fric)
        pnl[idx] += x
        max_day[idx] = np.maximum(max_day[idx], x)
        peak[idx] = np.maximum(peak[idx], pnl[idx])

        breached = pnl[idx] <= peak[idx] - rules.trailing_mll
        # Consistency rule: target effectively becomes max(target, max_day / ratio).
        hit = (pnl[idx] >= rules.profit_target) & \
              (max_day[idx] <= rules.consistency_cap_ratio * pnl[idx]) & ~breached
        done = breached | hit
        passed[idx[hit]] = True
        tau[idx[done]] = day
        active[idx[done]] = False

    return AttemptResults(tau=tau, passed=passed, truncated=active.copy())


def simulate_xfa(cfg: EngineConfig, path: FeePath, n: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Return gross withdrawals (before the profit split) for `n` funded accounts.

    XFA MLL trails the EOD high by `trailing_mll` and locks at $0 net P&L.
    """
    x_cfg = cfg.xfa
    mll = cfg.combine_rules.trailing_mll
    mu = cfg.market_process.mu_daily_gross
    sigma = cfg.market_process.sigma_daily
    fric = cfg.friction.daily_cost

    bal = np.zeros(n)
    peak = np.zeros(n)
    withdrawn = np.zeros(n)
    alive = np.ones(n, dtype=bool)

    for day in range(1, x_cfg.horizon_days + 1):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        bal[idx] += _draw_days(rng, idx.size, mu, sigma, path.intraday_dll, fric)
        peak[idx] = np.maximum(peak[idx], bal[idx])
        threshold = np.minimum(peak[idx] - mll, 0.0)
        dead = bal[idx] <= threshold
        alive[idx[dead]] = False

        if day % x_cfg.payout_interval_days == 0:
            live = idx[~dead]
            amt = np.clip((bal[live] - x_cfg.payout_buffer) * x_cfg.payout_fraction,
                          0.0, x_cfg.max_payout)
            bal[live] -= amt
            withdrawn[live] += amt
    return withdrawn
