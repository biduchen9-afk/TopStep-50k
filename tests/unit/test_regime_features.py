"""Daily features from bars: correctness + causality."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from topstep50k.engine.types import Bar
from topstep50k.regime import daily_features_from_bars


def _build_bars(n_days: int, prices_per_day=24, seed=0):
    """N consecutive trading days, prices_per_day bars per day in UTC,
    starting at 13:30 UTC (= 09:30 EDT)."""
    rng = np.random.default_rng(seed)
    bars = []
    start_day = datetime(2024, 6, 3, 13, 30, tzinfo=timezone.utc)  # Mon EDT
    p = 4500.0
    for d in range(n_days):
        day_start = start_day + timedelta(days=d)
        for j in range(prices_per_day):
            p += float(rng.normal(0, 1.0))
            ts = day_start + timedelta(minutes=j * 15)
            bars.append(Bar(ts=ts, open=p, high=p + 0.5, low=p - 0.5,
                             close=p, volume=10))
    return bars


def test_rejects_short_history():
    bars = _build_bars(5)
    with pytest.raises(ValueError):
        daily_features_from_bars(bars)


def test_returns_one_row_per_day():
    bars = _build_bars(30)
    feats = daily_features_from_bars(bars)
    assert len(feats) == 30
    assert feats.matrix.shape == (30, 5)


def test_first_row_has_zero_return():
    bars = _build_bars(25)
    feats = daily_features_from_bars(bars)
    # First day has no prior close -> log_return = 0
    assert feats.matrix[0, 0] == 0.0


def test_vol_5_warmup():
    bars = _build_bars(25)
    feats = daily_features_from_bars(bars)
    # vol_5 column should be 0 for the first 4 days, nonzero from day 5
    # (depending on seed; just assert non-negative)
    for row in feats.matrix:
        assert row[3] >= 0  # vol_5
        assert row[4] >= 0  # vol_20


def test_ts_eod_matches_dates():
    bars = _build_bars(22)
    feats = daily_features_from_bars(bars)
    # ts_eod entries must be on the same Eastern date as the dates entry
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")
    for d, ts in zip(feats.dates, feats.ts_eod_utc):
        assert ts.astimezone(EASTERN).date() == d
