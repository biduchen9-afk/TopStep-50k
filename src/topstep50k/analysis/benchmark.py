"""Naive buy-and-hold benchmark daily PnL.

Checklist item from the 2026-08-26 model-validation audit (see
docs/rules_sources.md / session notes). Important framing correction
from the same day: the goal here is passing a prop-firm Combine under a
hard $2,000 trailing MLL and a fixed profit target, NOT maximizing
unconstrained dollar return -- so "does the strategy out-earn
buy-and-hold in $" is the wrong question and is not computed anywhere
downstream. What evaluation/harness.py's Gate 4 actually compares is
Combine PASS RATE: this benchmark's daily series run through the same
simulate_combine_window/realized_pass_rate machinery as any strategy.
A naked buy-and-hold position is expected to show a near-zero pass rate
(nothing bounds its drawdown against the $2,000 MLL) -- that's not a
bug in the comparison, it's the actual point: it isolates how much of
the strategy's Combine pass rate comes from risk-bounding (stops, gates,
flat-by-close) versus just riding a directional move a naive position
would also have caught.

Uses the same 16:00 CT trading-day bucketing as the rest of the engine
(engine.ledger.trading_day) so the resulting daily series aligns exactly
with a strategy's own daily_pnl day index -- no off-by-one at session
boundaries.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

from topstep50k.engine.ledger import trading_day

if TYPE_CHECKING:
    from topstep50k.engine.types import Bar


def buy_and_hold_daily_pnl(
    bars: list["Bar"],
    point_value: Decimal,
    *,
    qty: int = 1,
) -> dict[date, Decimal]:
    """Dollar PnL of holding `qty` contracts long throughout, marked at
    each trading day's final bar close vs. the prior trading day's final
    close. First day in the series is 0 (no prior mark to diff against).

    `bars` must be chronologically sorted (as returned by the project's
    loaders) and tz-aware UTC.
    """
    last_close_by_day: dict[date, float] = {}
    for b in bars:
        last_close_by_day[trading_day(b.ts)] = b.close  # last write per day wins
    days = sorted(last_close_by_day)
    pv = float(point_value)
    out: dict[date, Decimal] = {}
    prev_close = None
    for d in days:
        close = last_close_by_day[d]
        if prev_close is None:
            out[d] = Decimal("0")
        else:
            out[d] = Decimal(str(round((close - prev_close) * pv * qty, 2)))
        prev_close = close
    return out


def to_aligned_array(daily: dict[date, Decimal], day_index: list[date]) -> np.ndarray:
    """Same convention as the *_databento eval/sweep scripts' `to_series`
    helpers: zero-filled outside the benchmark's own day index."""
    out = np.zeros(len(day_index), dtype=float)
    for i, d in enumerate(day_index):
        if d in daily:
            out[i] = float(daily[d])
    return out
