"""Replay Topstep 150K Combine and Express Funded Account rules on REAL strategy P&L.

Rules implemented (Topstep help-centre articles, checked October 2026 via web search; see README):
  * Day trading only: every strategy is flat by 15:55 ET, before the 3:10 PM CT (4:10 PM ET) cutoff.
  * Maximum Loss Limit (MLL) $4,500: set from the highest END-OF-DAY balance, trails up, and LOCKS
    at the starting balance ($0) once reached. A breach is enforced in REAL TIME against realised +
    unrealised P&L (intraday low), not just at the close. (`intraday_mll=False` gives the old EOD rule.)
  * Combine: +$9,000 target; consistency = best day <= 55% of total profit (target rises to
    best_day / 0.55 otherwise); max 15 minis (150 micros).
  * Optional Daily Loss Limit (DLL): when the intraday low reaches it, positions are liquidated at the
    DLL and the session is over (not a breach unless the MLL is closer).
  * Express Funded Account: balance starts at $0; contract limit follows a balance ladder
    (3/4/5/10/15 lots); payout paths:
        standard    - 5 winning days of >= $150, up to 50% of balance, cap $5,000
        consistency - >= 3 trading days and best day <= 40% of net profit, up to 50%, cap $6,000
      minimum request $125. After EVERY payout the MLL resets to $0 permanently and the day counts restart.
  * 90/10 profit split. Fees: see config.yaml (billing 21 trading days, activation only on pass).

Sizing: integer micro contracts (10 micros = 1 mini). A "risk unit" R is the dollar value of one
trailing-14-session range; contracts = max(1, round(R / dollars-per-ATR-per-micro)), then capped.
Outcomes are measured from EVERY historical start date; CIs resample whole calendar years.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from intraday_strategies import DailySeries
from pricing_config import EngineConfig

MICRO_PER_MINI = 10
# (minimum balance, lots) ladder for a 150K XFA, from third-party summaries of Topstep's scaling plan
XFA_LADDER = ((0.0, 3), (1500.0, 4), (2000.0, 5), (3000.0, 10), (4500.0, 15))


@dataclass(frozen=True)
class RuleSet:
    profit_target: float = 9000.0
    mll: float = 4500.0
    consistency_ratio: float = 0.55           # official Combine figure (spec said 0.50)
    combine_mll_locks: bool = True            # MLL locks at $0 once reached (official), spec: trails forever
    intraday_mll: bool = True                 # breach is real-time incl. open P&L
    dll: Optional[float] = None
    split: float = 0.90
    win_day: float = 150.0
    payout_min: float = 125.0
    cap_standard: float = 5000.0
    cap_consistency: float = 6000.0
    consistency_xfa: float = 0.40
    consistency_days: int = 3
    standard_win_days: int = 5
    combine_micros: int = 150
    max_combine_days: int = 252               # an unresolved attempt is abandoned (counted as a failure) after a year
    ladder: tuple = XFA_LADDER


def rules_from_config(cfg: EngineConfig, path_key: str, **overrides) -> RuleSet:
    o = cfg.official
    r = RuleSet(profit_target=cfg.combine_rules.profit_target, mll=cfg.combine_rules.trailing_mll,
                consistency_ratio=o.consistency_ratio, combine_mll_locks=o.combine_mll_locks,
                intraday_mll=o.intraday_mll, dll=cfg.paths[path_key].intraday_dll, split=cfg.profit_split,
                win_day=o.win_day, payout_min=o.payout_min, cap_standard=o.cap_standard,
                cap_consistency=o.cap_consistency, consistency_xfa=o.consistency_xfa,
                consistency_days=o.consistency_days, standard_win_days=o.standard_win_days,
                combine_micros=o.combine_micros, max_combine_days=o.max_combine_days, ladder=o.xfa_ladder)
    return replace(r, **overrides) if overrides else r


def costs_from_config(cfg: EngineConfig):
    from intraday_strategies import Costs
    # sleeves are priced per MINI-equivalent and then divided by 10 to get one micro, so a micro's all-in fee is x10 here
    return Costs(commission_rt=cfg.friction.micro_round_turn_commission * MICRO_PER_MINI, slippage_ticks_rt=cfg.friction.slippage_ticks_rt)


def xfa_micro_cap(balance: float, ladder: tuple = XFA_LADDER) -> int:
    lots = ladder[0][1]
    for floor_bal, l in ladder:
        if balance >= floor_bal:
            lots = l
    return lots * MICRO_PER_MINI


@dataclass(frozen=True)
class Stream:
    """Per-session P&L for ONE micro contract plus the dollar value of one ATR per micro."""
    name: str
    dates: np.ndarray
    pnl: np.ndarray          # net $ per micro
    low: np.ndarray          # worst intraday MTM $ per micro (<= 0)
    atr_usd: np.ndarray      # $ value of the trailing-14-session range per micro (NaN if unavailable)
    ok: np.ndarray

    @staticmethod
    def from_series(name: str, s: DailySeries) -> "Stream":
        k = 1.0 / MICRO_PER_MINI
        atr_usd = np.where(s.traded_ok, s.atr * s.mult * k, np.nan)
        return Stream(name, s.dates, np.where(s.traded_ok, s.pnl * k, 0.0),
                      np.where(s.traded_ok, s.minpnl * k, 0.0), atr_usd, s.traded_ok)

    def desired_micros(self, risk_unit: float) -> List[int]:
        """Contracts a volatility-targeting bot would hold on each session (>= 1)."""
        n = np.where(self.ok, np.maximum(1, np.rint(risk_unit / np.where(self.ok, self.atr_usd, 1.0))), 1)
        return n.astype(int).tolist()


def _lists(s: Stream) -> Tuple[List[float], List[float]]:
    return s.pnl.tolist(), s.low.tolist()


# ----------------------------------------------------------------------------------------------
# Combine
# ----------------------------------------------------------------------------------------------
def run_combine(x: List[float], lo: List[float], nd: List[int], i0: int, r: RuleSet,
                dyn_a: Optional[float] = None, atr: Optional[List[float]] = None) -> Tuple[int, int]:
    """Return (status, tau): 1 pass, 0 breach, -1 data ran out. tau is a 1-based day count."""
    bal = peak = best = 0.0
    floor = -r.mll
    n = len(x)
    dll = r.dll
    for t in range(i0, min(n, i0 + r.max_combine_days)):
        if dyn_a is None:
            k = min(nd[t], r.combine_micros)
        else:                                   # risk = dyn_a x cushion above the MLL, in $ per ATR move
            a_t = atr[t]
            k = 1 if not (a_t > 0) else min(max(1, int(round(dyn_a * max(bal - floor, 500.0) / a_t))), r.combine_micros)
        p, w = x[t] * k, lo[t] * k
        lim = floor - bal                       # negative distance to the MLL
        if r.intraday_mll or dll is not None:
            first = lim if dll is None else max(lim, dll)
            if r.intraday_mll and w <= first:
                if dll is not None and dll > lim:
                    p = dll                     # DLL reached first: liquidate, day over
                else:
                    return 0, t - i0 + 1        # MLL touched in real time
            elif dll is not None and w <= dll:
                p = dll
        bal += p
        if not r.intraday_mll and bal <= floor:
            return 0, t - i0 + 1
        if p > best:
            best = p
        if bal > peak:
            peak = bal
            floor = peak - r.mll
            if r.combine_mll_locks and floor > 0.0:
                floor = 0.0
        if bal >= r.profit_target and best <= r.consistency_ratio * bal:
            return 1, t - i0 + 1
    return -1, min(n, i0 + r.max_combine_days) - i0


def combine_outcomes(x, lo, nd, cfg: EngineConfig, path_key: str, r: RuleSet,
                     starts: np.ndarray, dyn_a: Optional[float] = None, atr=None) -> Dict[str, np.ndarray]:
    p = cfg.paths[path_key]
    status = np.empty(len(starts), dtype=np.int8)
    tau = np.empty(len(starts), dtype=np.int32)
    for k, i0 in enumerate(starts):
        status[k], tau[k] = run_combine(x, lo, nd, int(i0), r, dyn_a, atr)
    bills = 1 + (np.maximum(tau, 1) - 1) // cfg.billing_cycle_days
    fees = bills * p.monthly_cost + np.where(status == 1, p.activation_fee, 0.0)
    return {"status": status, "tau": tau, "fees": fees}


# ----------------------------------------------------------------------------------------------
# Funded account
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FundedPlan:
    """Funded-account plan.

    Stage 1 trades at risk multiplier `r1` until a payout is possible and the balance >= `t1`; then
    `w1` (<= 0.5 under Topstep's rule) of the balance is withdrawn. The MLL then resets to $0, so the
    money left in the account is the only cushion. Stage 2 trades at multiplier `r2` until the balance
    >= `t2` again, takes `w2` of it and the account is closed. `t2=None` closes after the first payout.
    """
    name: str
    path: str = "standard"            # "standard" or "consistency"
    t1: float = 2000.0
    w1: float = 0.5
    r1: float = 1.0
    t2: Optional[float] = None
    w2: float = 0.5
    r2: float = 1.0
    max_payouts: int = 2               # total payouts before the account is closed (>=2 needs t2)
    a1: Optional[float] = None         # dynamic sizing: risk = a1 x cushion (balance - MLL floor) before the first payout
    a2: Optional[float] = None         # ... and a2 x cushion after it (None -> a1). None = use the fixed-size `nd` list
    scale: Optional[tuple] = None      # scaling of `a1` BEFORE the first payout (firm_engine.run_funded_firm only): ("ladder", ((balance, a), ...)) | ("perf", L, f_up, f_down) | ("ramp", d, a_start); None = constant a1


def run_funded_detail(x: List[float], lo: List[float], nd: List[int], i0: int, r: RuleSet,
                      plan: FundedPlan, atr: Optional[List[float]] = None, times: Optional[list] = None) -> Tuple[List[float], int, int, float]:
    """Return (net payouts in order, days, end_state, balance when it ended): end_state 1 dead, 2 closed by plan, 0 data end."""
    bal = peak = 0.0
    floor = -r.mll
    reset = False                       # True after the first payout: MLL fixed at $0
    stage = 1
    wins = tdays = 0
    prof = best = 0.0
    pays: List[float] = []
    n = len(x)
    dll = r.dll
    cap_pay = r.cap_standard if plan.path == "standard" else r.cap_consistency
    for t in range(i0, n):
        if plan.a1 is None:
            mult = plan.r1 if stage == 1 else plan.r2
            k = max(1, int(round(nd[t] * mult)))
        else:
            a_t = atr[t]
            a = plan.a1 if (stage == 1 or plan.a2 is None) else plan.a2
            k = 1 if not (a_t > 0) else max(1, int(round(a * max(bal - floor, 250.0) / a_t)))
        k = min(k, xfa_micro_cap(bal, r.ladder))
        p, w = x[t] * k, lo[t] * k
        lim = floor - bal
        first = lim if dll is None else max(lim, dll)
        if w <= first:
            if dll is not None and dll > lim:
                p = dll
            elif r.intraday_mll:
                return pays, t - i0 + 1, 1, floor
        bal += p
        if not r.intraday_mll and bal <= floor:
            return pays, t - i0 + 1, 1, bal
        tdays += 1
        prof += p
        if p >= r.win_day:
            wins += 1
        if p > best:
            best = p
        if not reset and bal > peak:
            peak = bal
            floor = min(peak - r.mll, 0.0)
        if plan.path == "standard":
            eligible = wins >= r.standard_win_days
        else:
            eligible = tdays >= r.consistency_days and prof > 0 and best <= r.consistency_xfa * prof
        trig, frac = (plan.t1, plan.w1) if stage == 1 else (plan.t2, plan.w2)
        if eligible and bal >= trig:
            gross = min(min(frac, 0.5) * bal, cap_pay)
            if gross >= r.payout_min:
                bal -= gross
                floor, reset = 0.0, True
                wins = tdays = 0
                prof = best = 0.0
                pays.append(r.split * gross)
                if times is not None:
                    times.append(t)
                stage = 2
                if plan.t2 is None or len(pays) >= plan.max_payouts:
                    return pays, t - i0 + 1, 2, bal
    return pays, n - i0, 0, bal


def run_funded(x: List[float], lo: List[float], nd: List[int], i0: int, r: RuleSet,
               plan: FundedPlan, atr: Optional[List[float]] = None) -> Tuple[float, float, int, int]:
    """Compact form: (first payout, SUM of later payouts, days, end_state)."""
    pays, days, state, _ = run_funded_detail(x, lo, nd, i0, r, plan, atr)
    return (pays[0] if pays else 0.0), float(sum(pays[1:])), days, state


def year_bootstrap(values: np.ndarray, years: np.ndarray, stat, n_boot: int = 300,
                   seed: int = 1) -> Tuple[float, float, float]:
    """(point, lo5, hi95) of stat(values), resampling whole calendar years."""
    rng = np.random.default_rng(seed)
    uy = np.unique(years)
    groups = [values[years == y] for y in uy]
    pt = stat(np.concatenate(groups))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uy), len(uy))
        reps[b] = stat(np.concatenate([groups[i] for i in pick]))
    return float(pt), float(np.percentile(reps, 5)), float(np.percentile(reps, 95))
