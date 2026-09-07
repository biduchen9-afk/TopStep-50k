"""Daily-resolution feature extraction for regime detection on bar data.

For each US/Eastern trading date, computes end-of-day features that an
HMM (or any sequence model) can consume:

  * log_return  -- log(close_eod / close_prev_eod)
  * abs_return  -- |log_return| (instantaneous vol proxy)
  * range_pct   -- (high_today - low_today) / close_prev (true-range proxy)
  * vol_5       -- stdev of last 5 daily log returns
  * vol_20      -- stdev of last 20 daily log returns

Causality: each row at index i is computed from bars on or before that
day; features at day i are observable at end-of-day i. The downstream
HMM filter MUST use feature[i] to make a decision applied to day i+1
(see regime.hmm.HMMRegimeFilter for the forward-recurrence guarantee).

This module does NOT call the clock; it is offline pre-compute. The
clock guard lives in the filter where the features are consumed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from topstep50k.engine.types import Bar


EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class DailyFeatures:
    dates: list[date]
    ts_eod_utc: list[datetime]  # last-bar timestamp of each date, UTC
    matrix: np.ndarray          # shape (n_days, 5)
    feature_names: tuple[str, ...] = (
        "log_return", "abs_return", "range_pct", "vol_5", "vol_20"
    )

    def __len__(self) -> int:
        return len(self.dates)


def daily_features_from_bars(bars: list[Bar]) -> DailyFeatures:
    """Aggregate bars into end-of-day features keyed by US/Eastern date."""
    by_day: dict[date, list[Bar]] = defaultdict(list)
    for b in bars:
        eastern_date = b.ts.astimezone(EASTERN).date()
        by_day[eastern_date].append(b)
    dates = sorted(by_day.keys())
    if len(dates) < 21:
        raise ValueError(
            f"Need at least 21 trading days for the rolling-vol features; "
            f"got {len(dates)}"
        )
    log_returns_buffer: list[float] = []
    rows: list[list[float]] = []
    ts_eods: list[datetime] = []
    prev_close: float | None = None
    for d in dates:
        day_bars = sorted(by_day[d], key=lambda b: b.ts)
        last = day_bars[-1]
        close = last.close
        high = max(b.high for b in day_bars)
        low = min(b.low for b in day_bars)
        if prev_close is None:
            log_r = 0.0
            range_pct = (high - low) / close if close > 0 else 0.0
        else:
            log_r = float(np.log(close / prev_close)) if prev_close > 0 else 0.0
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            range_pct = tr / prev_close if prev_close > 0 else 0.0
        log_returns_buffer.append(log_r)
        if len(log_returns_buffer) > 20:
            log_returns_buffer.pop(0)
        if len(log_returns_buffer) >= 5:
            vol_5 = float(np.std(log_returns_buffer[-5:], ddof=1))
        else:
            vol_5 = 0.0
        if len(log_returns_buffer) >= 20:
            vol_20 = float(np.std(log_returns_buffer[-20:], ddof=1))
        else:
            vol_20 = 0.0
        rows.append([log_r, abs(log_r), range_pct, vol_5, vol_20])
        ts_eods.append(last.ts)
        prev_close = close
    return DailyFeatures(
        dates=dates,
        ts_eod_utc=ts_eods,
        matrix=np.array(rows, dtype=np.float64),
    )
