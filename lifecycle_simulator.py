"""Sequential 'rinse and repeat' lifecycle: one account at a time, over a fixed horizon.

Combine -> pass (pay C_act) -> XFA -> first payout (account closes) -> new Combine -> ...
A breach in either phase cancels the account and a fresh Combine (C_sub) starts the next day.

Cash flows (realised only; an unfinished attempt at the horizon is a sunk cost):
  -C_sub on attempt day 1 and each 21 days after, -C_act on pass, +split * payout on payout.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from combine_simulator import _draw_days
from pricing_config import EngineConfig

HORIZON_DAYS_3Y = 3 * 252


@dataclass(frozen=True)
class PayoutRule:
    """When the funded account takes its first payout and closes (assumption, configurable)."""
    trigger_profit: float = 3000.0   # request a payout once eligible AND net profit reaches this
    payout_fraction: float = 0.50    # share of balance withdrawn
    max_payout: float = 5000.0       # gross cap per payout
    close_after_payout: bool = True
    # Eligibility. Standard path: N winning days of >= winning_day_min.
    # Consistency path: min_trading_days traded and best day <= consistency_ratio of profit.
    min_winning_days: int = 5
    winning_day_min: float = 150.0
    consistency_ratio: Optional[float] = None
    min_trading_days: int = 0


STANDARD_PAYOUT = PayoutRule(max_payout=5000.0, min_winning_days=5, winning_day_min=150.0)
CONSISTENCY_PAYOUT = PayoutRule(max_payout=6000.0, min_winning_days=0, consistency_ratio=0.40,
                                min_trading_days=3)


@dataclass
class LifecycleResult:
    total_pnl: np.ndarray    # net cash over the horizon, per path
    fees: np.ndarray
    payouts: np.ndarray      # trader's share received
    n_combines: np.ndarray
    n_funded: np.ndarray
    n_payouts: np.ndarray
    n_xfa_breaches: np.ndarray


def simulate_lifecycle(cfg: EngineConfig, path_key: str, n_paths: int,
                       rng: np.random.Generator, horizon: int = HORIZON_DAYS_3Y,
                       payout: Optional[PayoutRule] = None) -> LifecycleResult:
    payout = payout or PayoutRule()
    path = cfg.paths[path_key]
    rules = cfg.combine_rules
    mp = cfg.market_process
    fric = cfg.friction.daily_cost
    cycle = cfg.billing_cycle_days
    n = n_paths

    funded = np.zeros(n, dtype=bool)
    age = np.zeros(n, dtype=np.int64)       # Combine day counter for the billing clock
    pnl = np.zeros(n)
    peak = np.zeros(n)
    max_day = np.zeros(n)
    win_days = np.zeros(n, dtype=np.int64)  # XFA days with P&L >= winning_day_min
    xfa_days = np.zeros(n, dtype=np.int64)

    fees = np.zeros(n)
    paid_out = np.zeros(n)
    n_comb = np.zeros(n, dtype=np.int64)
    n_fund = np.zeros(n, dtype=np.int64)
    n_pay = np.zeros(n, dtype=np.int64)
    n_xbr = np.zeros(n, dtype=np.int64)
    n_comb += 1  # first Combine

    def restart(mask: np.ndarray) -> None:
        funded[mask] = False
        age[mask] = 0
        pnl[mask] = 0.0
        peak[mask] = 0.0
        max_day[mask] = 0.0
        win_days[mask] = 0
        xfa_days[mask] = 0
        n_comb[mask] += 1

    for _ in range(horizon):
        comb = ~funded
        age[comb] += 1
        bill = comb & ((age == 1) | ((age > 1) & ((age - 1) % cycle == 0)))
        fees[bill] += path.monthly_cost

        x = _draw_days(rng, n, mp.mu_daily_gross, mp.sigma_daily, path.intraday_dll, fric)
        pnl += x
        peak = np.maximum(peak, pnl)

        # --- Combine ---
        max_day = np.maximum(max_day, x)
        win_days += funded & (x >= payout.winning_day_min)
        xfa_days += funded
        c_breach = comb & (pnl <= peak - rules.trailing_mll)
        c_pass = comb & ~c_breach & (pnl >= rules.profit_target) & \
                 (max_day <= rules.consistency_cap_ratio * pnl)
        # --- XFA ---
        f_breach = funded & (pnl <= np.minimum(peak - rules.trailing_mll, 0.0))
        elig = win_days >= payout.min_winning_days
        if payout.consistency_ratio is not None:
            elig &= (xfa_days >= payout.min_trading_days) & (pnl > 0) & \
                    (max_day <= payout.consistency_ratio * pnl)
        f_pay = funded & ~f_breach & elig & (pnl >= payout.trigger_profit)

        fees[c_pass] += path.activation_fee
        n_fund[c_pass] += 1
        n_xbr += f_breach

        gross = np.clip(pnl * payout.payout_fraction, 0.0, payout.max_payout)
        paid_out[f_pay] += cfg.profit_split * gross[f_pay]
        n_pay[f_pay] += 1

        # transitions (applied after all flags are computed from the same snapshot)
        to_xfa = c_pass
        close = c_breach | f_breach | (f_pay if payout.close_after_payout else np.zeros(n, bool))
        if not payout.close_after_payout:
            pnl[f_pay] -= gross[f_pay]
        funded[to_xfa] = True
        pnl[to_xfa] = 0.0
        peak[to_xfa] = 0.0
        max_day[to_xfa] = 0.0
        win_days[to_xfa] = 0
        xfa_days[to_xfa] = 0
        restart(close)

    return LifecycleResult(total_pnl=paid_out - fees, fees=fees, payouts=paid_out,
                           n_combines=n_comb, n_funded=n_fund, n_payouts=n_pay,
                           n_xfa_breaches=n_xbr)


def breakeven_sharpe_lifecycle(cfg: EngineConfig, path_key: str, n_paths: int = 4000,
                               payout: Optional[PayoutRule] = None, seed: int = 7,
                               lo: float = 0.0, hi: float = 3.0, iters: int = 11) -> float:
    """Smallest gross Sharpe with mean 3-year P&L >= 0 (common random numbers)."""
    def mean_pnl(sr: float) -> float:
        c = cfg.with_market(eval_mode="positive_drift", sharpe_annual=sr)
        r = simulate_lifecycle(c, path_key, n_paths, np.random.default_rng(seed), payout=payout)
        return float(r.total_pnl.mean())

    if mean_pnl(lo) >= 0:
        return lo
    if mean_pnl(hi) < 0:
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if mean_pnl(mid) >= 0 else (mid, hi)
    return hi
