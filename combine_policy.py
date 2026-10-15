"""Combine sizing policies that use only the account's own state (mechanics, no new signals).

size (micros) on day t = min( round(R * m / ATR_t) , round(c * max(balance - MLL floor, 500) / ATR_t) ), at least 1, at most the Combine cap, where
  m = s0     while the account has not yet made `ramp` x target in profit (peak balance below ramp x target): early-phase multiplier (1 = neutral)
  m = g      once balance >= lock_at x target: de-risk to protect a nearly finished Combine (lock_at=None disables)
  c          cushion cap: never risk more than c ATR-units of the remaining distance to the loss floor (cap_c=None disables)
With s0=1, lock_at=None, cap_c=None this is exactly the fixed-risk Combine in real_engine.run_combine (checked in test_combine_policy.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from real_engine import RuleSet


@dataclass(frozen=True)
class CombinePolicy:
    R: float = 9600.0
    s0: float = 1.0
    ramp: float = 0.2
    lock_at: Optional[float] = None
    g: float = 1.0
    cap_c: Optional[float] = None


def run_combine_policy(x: List[float], lo: List[float], atr: List[float], i0: int, r: RuleSet, pol: CombinePolicy) -> Tuple[int, int]:
    """Return (status, tau) exactly as run_combine: 1 pass, 0 breach, -1 data ran out; tau is a 1-based day count."""
    bal = peak = best = 0.0
    floor = -r.mll
    n = len(x)
    dll = r.dll
    tgt = r.profit_target
    for t in range(i0, min(n, i0 + r.max_combine_days)):
        a_t = atr[t]
        if not (a_t > 0):
            k = 1
        else:
            m = 1.0
            if pol.lock_at is not None and bal >= pol.lock_at * tgt:
                m = pol.g
            elif peak < pol.ramp * tgt:
                m = pol.s0
            k = max(1, int(round(pol.R * m / a_t)))
            if pol.cap_c is not None:
                k = min(k, max(1, int(round(pol.cap_c * max(bal - floor, 500.0) / a_t))))
        k = min(k, r.combine_micros)
        p, w = x[t] * k, lo[t] * k
        lim = floor - bal
        if r.intraday_mll or dll is not None:
            first = lim if dll is None else max(lim, dll)
            if r.intraday_mll and w <= first:
                if dll is not None and dll > lim:
                    p = dll
                else:
                    return 0, t - i0 + 1
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
