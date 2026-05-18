"""Rule-based regime conditioners derived from the academic literature.

Each conditioner is a pure function of bar history. Features are
computed end-of-day; the conditioner exposed to a strategy applies
the feature from PRIOR EOD to today's session decision so the gate
is strictly causal.

The three conditioners and their literature grounding:

  1. `orb_expansion_gate`
     ORB only trades when realized-vol is NOT in deep compression.
     Rule: `rv_5 / rv_20 >= 0.8` where rv_K is the stdev of the last
     K daily log returns. Source: the ATR-breakout / volatility-
     expansion literature -- "periods of low volatility are followed
     by periods of high volatility; trade the transition." We invert
     the chop filter direction (which we already tried and which
     hurt) and instead require the SHORT-WINDOW vol to be roughly in
     line with or above the LONG-WINDOW vol. Deep compression (rv_5
     << rv_20) tends to either keep being compressed or expand in
     unpredictable directions.

  2. `meanrev_low_vol_gate`
     Mean reversion only trades on days when realized vol is below
     its trailing median. Rule: `rv_20[t-1] < median(rv_20 over the
     last 60 days)`. Source: the mean-reversion literature is clear
     that fading bands fails during high-vol trending regimes; the
     "Bollinger fade" works in normal/low-vol regimes when prices
     mean-revert to the local average. We use a trailing-median
     threshold (no fitted parameter) for robustness.

  3. `overnight_drift_post_selloff_gate`
     Overnight long only after a DOWN RTH session. Rule: prior RTH
     close < prior RTH open. Source: Boyarchenko, Larsen, and Whelan
     (NY Fed Staff Report #917, 2020), "The Overnight Drift". Quote:
     "market selloffs generate robust positive overnight reversals,
     while reversals following market rallies are much more modest."
     The strategy is a direct implementation of that empirical
     finding: trade the inventory-risk-driven overnight reversal
     only when there is excess sell pressure to be unwound.

All three rules have ZERO free parameters tuned to our data. The
trailing-median is a function of the data but is not a fitted
parameter -- it is a self-normalising threshold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, time, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np

from topstep50k.engine.types import Bar


EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class DailySessionStats:
    """Per-US/Eastern-trading-date summary.

    Fields are computed from bars whose `ts.astimezone(EASTERN).date()`
    equals `d`. RTH-only fields use only bars whose Eastern time-of-day
    is in [09:30, 16:00).
    """
    d: date
    rth_open: float | None   # close of bar at 09:30 ET (approx)
    rth_close: float | None  # close of last bar before 16:00 ET
    log_return_close_to_close: float
    rth_return: float  # (rth_close / rth_open) - 1


def per_day_session_stats(bars: list[Bar]) -> dict[date, DailySessionStats]:
    """Aggregate per-US/Eastern-trading-day stats from bar history.

    Causal: each entry uses only bars whose ts is on day `d`.
    """
    by_day: dict[date, list[Bar]] = defaultdict(list)
    for b in bars:
        by_day[b.ts.astimezone(EASTERN).date()].append(b)
    out: dict[date, DailySessionStats] = {}
    prev_close: float | None = None
    rth_open_t = time(9, 30)
    rth_close_t = time(16, 0)
    for d in sorted(by_day.keys()):
        day_bars = sorted(by_day[d], key=lambda b: b.ts)
        rth_bars = [b for b in day_bars
                    if rth_open_t <= b.ts.astimezone(EASTERN).time() < rth_close_t]
        rth_open = rth_bars[0].open if rth_bars else None
        rth_close = rth_bars[-1].close if rth_bars else None
        close_eod = day_bars[-1].close
        if prev_close is not None and prev_close > 0 and close_eod > 0:
            log_r = float(np.log(close_eod / prev_close))
        else:
            log_r = 0.0
        if rth_open and rth_close and rth_open > 0:
            rth_r = (rth_close - rth_open) / rth_open
        else:
            rth_r = 0.0
        out[d] = DailySessionStats(
            d=d, rth_open=rth_open, rth_close=rth_close,
            log_return_close_to_close=log_r, rth_return=rth_r,
        )
        prev_close = close_eod
    return out


def rolling_vol(stats: dict[date, DailySessionStats],
                window: int) -> dict[date, float]:
    """For each date, the stdev of the LAST `window` daily close-to-close
    log returns ending at-or-before that date. Returns 0.0 until the
    history is long enough.

    Strict causality: the value at date d uses returns from days <= d.
    """
    days = sorted(stats.keys())
    buf: list[float] = []
    out: dict[date, float] = {}
    for d in days:
        buf.append(stats[d].log_return_close_to_close)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) >= 2:
            out[d] = float(np.std(buf, ddof=1))
        else:
            out[d] = 0.0
    return out


def trailing_median(series: dict[date, float], window: int
                     ) -> dict[date, float]:
    """For each date, the median of the LAST `window` values ending at
    that date (inclusive). Returns the value itself until the window is
    populated."""
    days = sorted(series.keys())
    buf: list[float] = []
    out: dict[date, float] = {}
    for d in days:
        buf.append(series[d])
        if len(buf) > window:
            buf.pop(0)
        out[d] = float(np.median(buf))
    return out


# ---------- the three gates ------------------------------------------------


def orb_expansion_gate(stats: dict[date, DailySessionStats]
                       ) -> Callable[[date], bool]:
    """Gate factory for ORB. Returns a callable mapping a trading date
    to True (trade allowed) / False (skip).

    Rule: rv_5_prior / rv_20_prior >= 0.8, where rv_K_prior is the
    realized-vol over the K days ENDING THE DAY BEFORE the decision
    date. Returns False until both series are warm.
    """
    rv5 = rolling_vol(stats, 5)
    rv20 = rolling_vol(stats, 20)
    days = sorted(stats.keys())
    prior_of = {d: days[i - 1] if i > 0 else None
                for i, d in enumerate(days)}

    def gate(d: date) -> bool:
        prior = prior_of.get(d)
        if prior is None:
            return False
        v5 = rv5.get(prior, 0.0)
        v20 = rv20.get(prior, 0.0)
        if v5 <= 0 or v20 <= 0:
            return False
        return bool((v5 / v20) >= 0.8)

    return gate


def meanrev_low_vol_gate(stats: dict[date, DailySessionStats],
                          *, median_window: int = 60
                          ) -> Callable[[date], bool]:
    """Gate factory for MeanRev. Rule: rv_20_prior < median(rv_20 over
    last `median_window` days)."""
    rv20 = rolling_vol(stats, 20)
    median = trailing_median(rv20, median_window)
    days = sorted(stats.keys())
    prior_of = {d: days[i - 1] if i > 0 else None
                for i, d in enumerate(days)}

    def gate(d: date) -> bool:
        prior = prior_of.get(d)
        if prior is None:
            return False
        v = rv20.get(prior, 0.0)
        m = median.get(prior, 0.0)
        if v <= 0 or m <= 0:
            return False
        return bool(v < m)

    return gate


def overnight_drift_post_selloff_gate(
    stats: dict[date, DailySessionStats],
) -> Callable[[date], bool]:
    """Gate factory for OvernightDrift. Rule: the PRIOR day's RTH
    return was negative (close < open in RTH session).

    Source: Boyarchenko, Larsen, Whelan (NY Fed SR #917, 2020).
    """
    days = sorted(stats.keys())
    prior_of = {d: days[i - 1] if i > 0 else None
                for i, d in enumerate(days)}

    def gate(d: date) -> bool:
        prior = prior_of.get(d)
        if prior is None:
            return False
        s = stats.get(prior)
        if s is None:
            return False
        return bool(s.rth_return < 0.0)

    return gate
