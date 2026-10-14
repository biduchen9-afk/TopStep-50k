"""Replay Topstep Combine and funded-account rules on REAL daily strategy P&L.

Sizing: every day's P&L is expressed in session-ATR units (so 1 unit = the P&L of a position that
gains one trailing-14-session range), then multiplied by a dollar risk unit R. R is chosen so the
daily P&L standard deviation equals a target ("sigma_d"), which is the aggressiveness dial.
Fractional contracts are allowed (micros make that realistic).

Outcomes are measured from EVERY historical start date, not one path, so each probability is an
average over thousands of overlapping real windows. Confidence intervals bootstrap whole years.

Account rules (from the project spec, Topstep 150K):
  Combine: target $9,000, EOD trailing MLL $4,500 (trails continuously), consistency 50%,
           optional intraday DLL -$3,000 (day is liquidated at the DLL when the intraday low hits it).
  Funded:  balance starts at 0; floor = min(peak EOD balance - MLL, 0); withdrawals do not move the
           floor (it depends on peak profit); payouts need N winning days since the last payout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from intraday_strategies import DailySeries
from pricing_config import EngineConfig

WIN_DAY = 150.0   # a "winning day" for payout eligibility (Topstep standard path)


@dataclass(frozen=True)
class Stream:
    name: str
    dates: np.ndarray
    r_pnl: np.ndarray     # P&L in ATR units per session (0 on skipped sessions)
    r_min: np.ndarray     # worst intraday MTM in ATR units (<= 0)
    ok: np.ndarray

    @staticmethod
    def from_series(name: str, s: DailySeries) -> "Stream":
        denom = np.where(s.traded_ok, s.atr * s.mult, np.nan)
        r = np.where(s.traded_ok, s.pnl / denom, 0.0)
        m = np.where(s.traded_ok, s.minpnl / denom, 0.0)
        return Stream(name, s.dates, r, m, s.traded_ok)

    def risk_unit_for_sigma(self, sigma_d: float) -> float:
        sd = self.r_pnl[self.ok].std(ddof=1)
        return sigma_d / sd


def scale(stream: Stream, risk_unit: float) -> Tuple[List[float], List[float]]:
    return (stream.r_pnl * risk_unit).tolist(), (stream.r_min * risk_unit).tolist()


# ----------------------------------------------------------------------------------------------
# Combine
# ----------------------------------------------------------------------------------------------
def run_combine(x: List[float], m: List[float], i0: int, dll: Optional[float], target: float,
                mll: float, cap: float) -> Tuple[int, int]:
    """Return (status, tau): status 1 pass, 0 breach, -1 ran out of data. tau is 1-based days."""
    bal = peak = maxd = 0.0
    n = len(x)
    for t in range(i0, n):
        d = dll if (dll is not None and m[t] <= dll) else x[t]
        bal += d
        if d > maxd:
            maxd = d
        if bal > peak:
            peak = bal
        if bal <= peak - mll:
            return 0, t - i0 + 1
        if bal >= target and maxd <= cap * bal:
            return 1, t - i0 + 1
    return -1, n - i0


def combine_outcomes(x: List[float], m: List[float], cfg: EngineConfig, path_key: str,
                     starts: np.ndarray) -> Dict[str, np.ndarray]:
    p = cfg.paths[path_key]
    r = cfg.combine_rules
    status = np.empty(len(starts), dtype=np.int8)
    tau = np.empty(len(starts), dtype=np.int32)
    for k, i0 in enumerate(starts):
        status[k], tau[k] = run_combine(x, m, int(i0), p.intraday_dll, r.profit_target,
                                        r.trailing_mll, r.consistency_cap_ratio)
    bills = 1 + (np.maximum(tau, 1) - 1) // cfg.billing_cycle_days
    fees = bills * p.monthly_cost + np.where(status == 1, p.activation_fee, 0.0)
    return {"status": status, "tau": tau, "fees": fees}


# ----------------------------------------------------------------------------------------------
# Funded account variants
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FundedVariant:
    """A funded-account plan.

    Stage 1 runs at `size1` until the balance reaches `t1`, then `w1` of the balance is withdrawn
    (e.g. t1=2000, w1=0.5 -> take $1,000, keep $1,000). Stage 2 runs at `size2` until the balance
    reaches `t2`, then `w2` of the balance is withdrawn and the account is closed. If `t2` is None
    the account closes right after the first payout. `cap` is the gross cap per payout request.
    """
    name: str
    t1: float = 2000.0
    w1: float = 0.5
    size1: float = 1.0
    t2: Optional[float] = None
    w2: float = 0.5
    size2: float = 1.0
    cap: float = 5000.0
    need_win_days: int = 5


def run_funded(x: List[float], m: List[float], i0: int, dll: Optional[float], v: FundedVariant,
               mll: float, split: float) -> Tuple[float, float, int, int]:
    """Return (payout1_net, payout2_net, days_alive, end_state) with end_state 1 dead, 2 closed, 0 data end."""
    bal = peak = 0.0
    wins = 0
    stage = 1
    p1 = p2 = 0.0
    n = len(x)
    for t in range(i0, n):
        k = v.size1 if stage == 1 else v.size2
        d = dll if (dll is not None and m[t] * k <= dll) else x[t] * k
        bal += d
        if d >= WIN_DAY:
            wins += 1
        if bal > peak:
            peak = bal
        if bal <= min(peak - mll, 0.0):
            return p1, p2, t - i0 + 1, 1
        if wins >= v.need_win_days:
            trig, frac = (v.t1, v.w1) if stage == 1 else (v.t2, v.w2)
            if bal >= trig:
                gross = min(frac * bal, v.cap)
                net = split * gross
                bal -= gross
                wins = 0
                if stage == 1:
                    p1 = net
                    if v.t2 is None:
                        return p1, p2, t - i0 + 1, 2
                    stage = 2
                else:
                    p2 = net
                    return p1, p2, t - i0 + 1, 2
    return p1, p2, n - i0, 0


def funded_outcomes(x: List[float], m: List[float], cfg: EngineConfig, path_key: str,
                    v: FundedVariant, starts: np.ndarray) -> Dict[str, np.ndarray]:
    p = cfg.paths[path_key]
    out = {k: np.empty(len(starts)) for k in ("p1", "p2", "days", "state")}
    for k, i0 in enumerate(starts):
        a, b, d, s = run_funded(x, m, int(i0), p.intraday_dll, v, cfg.combine_rules.trailing_mll,
                                cfg.profit_split)
        out["p1"][k], out["p2"][k], out["days"][k], out["state"][k] = a, b, d, s
    return out


# ----------------------------------------------------------------------------------------------
# Summary statistics with a year-block bootstrap
# ----------------------------------------------------------------------------------------------
def year_bootstrap(values: np.ndarray, years: np.ndarray, stat, n_boot: int = 300,
                   seed: int = 1) -> Tuple[float, float, float]:
    """(point, lo5, hi95) of stat(values) resampling whole calendar years."""
    rng = np.random.default_rng(seed)
    uy = np.unique(years)
    groups = [values[years == y] for y in uy]
    pt = stat(np.concatenate(groups))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uy), len(uy))
        reps[b] = stat(np.concatenate([groups[i] for i in pick]))
    return float(pt), float(np.percentile(reps, 5)), float(np.percentile(reps, 95))
