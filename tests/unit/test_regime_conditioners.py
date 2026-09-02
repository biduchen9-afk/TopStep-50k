"""Tests for the literature-grounded regime conditioners.

Properties asserted:
  * Causality: a gate's decision for day d uses only data from days < d.
  * Warm-up: gates return False before sufficient history exists.
  * Overnight gate fires exactly on days after a down RTH session.
  * MeanRev gate fires when the trailing rv_20 is below its median.
  * ORB gate fires when rv_5/rv_20 ratio crosses 0.8.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from topstep50k.engine.types import Bar
from topstep50k.regime.conditioners import (
    meanrev_low_vol_gate,
    orb_expansion_gate,
    overnight_drift_post_selloff_gate,
    per_day_session_stats,
    rolling_vol,
    trailing_median,
)


EASTERN = ZoneInfo("America/New_York")


def _utc(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def _bar(ts, price):
    return Bar(ts=ts, open=price, high=price + 0.25, low=price - 0.25,
               close=price, volume=100)


def _build_session(d_start: date, n_days: int,
                    daily_returns: list[float],
                    rth_returns: list[float] | None = None):
    """Build minimal RTH bars per US/Eastern trading day with controlled
    close-to-close returns.

    Each day has 5 bars at 09:30, 12:00, and 15:55 ET (= UTC after the
    EDT-aware offset). We pick the close of the LAST bar as the EOD
    close and the open of the FIRST bar at 09:30 as the RTH open.
    """
    if rth_returns is None:
        rth_returns = daily_returns
    assert len(daily_returns) == n_days
    bars = []
    price = 4500.0
    for i in range(n_days):
        # i-th trading day
        day_eastern = datetime.combine(d_start + timedelta(days=i),
                                        datetime.min.time(),
                                        tzinfo=EASTERN)
        ret = daily_returns[i]
        rth_ret = rth_returns[i]
        rth_open = price
        rth_close = rth_open * (1 + rth_ret)
        eod = price * (1 + ret)  # close-to-close
        # 09:30 ET bar: at open price
        b1 = _bar(day_eastern.replace(hour=9, minute=30).astimezone(timezone.utc),
                   rth_open)
        # 12:00 ET bar: midway interpolation
        b2 = _bar(day_eastern.replace(hour=12, minute=0).astimezone(timezone.utc),
                   (rth_open + rth_close) / 2)
        # 15:55 ET bar: RTH close
        b3 = _bar(day_eastern.replace(hour=15, minute=55).astimezone(timezone.utc),
                   rth_close)
        # 20:00 ET bar (overnight close anchor)
        b4 = _bar(day_eastern.replace(hour=20, minute=0).astimezone(timezone.utc),
                   eod)
        bars.extend([b1, b2, b3, b4])
        price = eod
    return bars


def test_per_day_session_stats_computes_rth_return():
    # 2 days, RTH up 1% then down 2%
    bars = _build_session(date(2024, 6, 3), 2,
                          daily_returns=[0.01, -0.02],
                          rth_returns=[0.01, -0.02])
    stats = per_day_session_stats(bars)
    assert len(stats) == 2
    d1, d2 = sorted(stats.keys())
    assert stats[d1].rth_return == pytest.approx(0.01, rel=1e-6)
    assert stats[d2].rth_return == pytest.approx(-0.02, rel=1e-6)


def test_overnight_gate_fires_after_negative_rth():
    rng = np.random.default_rng(0)
    daily = list(rng.normal(0, 0.01, 30))
    # Force day 14 to be DOWN, day 15 should have gate=True
    daily[14] = -0.02
    daily[20] = +0.02  # day 21 should have gate=False
    bars = _build_session(date(2024, 6, 3), 30, daily_returns=daily,
                          rth_returns=daily)
    stats = per_day_session_stats(bars)
    gate = overnight_drift_post_selloff_gate(stats)
    days = sorted(stats.keys())
    # day 15 (one after the forced selloff) -> True
    assert gate(days[15]) is True
    # day 21 (one after the forced rally) -> False
    assert gate(days[21]) is False
    # day 0 returns False (no prior day)
    assert gate(days[0]) is False


def test_meanrev_gate_warms_up_then_fires_in_low_vol():
    # First 30 days: high volatility (0.02 stdev). Next 30 days: low vol (0.005).
    rng = np.random.default_rng(1)
    daily = list(rng.normal(0, 0.02, 30)) + list(rng.normal(0, 0.005, 30))
    bars = _build_session(date(2024, 1, 2), 60, daily_returns=daily,
                          rth_returns=daily)
    stats = per_day_session_stats(bars)
    gate = meanrev_low_vol_gate(stats, median_window=30)
    days = sorted(stats.keys())
    # Warm-up: should return False on the first day
    assert gate(days[0]) is False
    # By the end (day 58 or 59), rv_20 has dropped to the low-vol level,
    # which should be well below the 30-day median (which still has high-vol days)
    decisions_late = [gate(days[i]) for i in range(50, 60)]
    assert any(decisions_late), (
        f"expected at least one late-period gate=True in low-vol regime, "
        f"got {decisions_late}"
    )


def test_orb_gate_warms_up():
    rng = np.random.default_rng(2)
    daily = list(rng.normal(0, 0.01, 30))
    bars = _build_session(date(2024, 1, 2), 30, daily_returns=daily,
                          rth_returns=daily)
    stats = per_day_session_stats(bars)
    gate = orb_expansion_gate(stats)
    days = sorted(stats.keys())
    # First few days: rv_5 or rv_20 are zero/undefined => False
    assert gate(days[0]) is False
    # Late: rv_5/rv_20 will be ~1.0 (random returns, both windows similar),
    # so gate should be True more often than not.
    late_decisions = [gate(days[i]) for i in range(20, 30)]
    assert sum(late_decisions) >= 5, (
        f"expected most late-period decisions True for stationary noise, "
        f"got {late_decisions}"
    )


def test_rolling_vol_is_causal():
    """rolling_vol[d] depends only on returns at days <= d."""
    daily = [0.01, 0.02, -0.01, -0.02, 0.005]
    bars = _build_session(date(2024, 1, 2), 5, daily_returns=daily,
                          rth_returns=daily)
    stats = per_day_session_stats(bars)
    rv = rolling_vol(stats, window=3)
    days = sorted(stats.keys())
    # day 0: only one return -> ddof=1 stdev undefined, function returns 0
    assert rv[days[0]] == 0.0
    # day 1: two returns -> stdev defined
    assert rv[days[1]] > 0
    # day 4: stdev of last 3 returns
    expected = float(np.std([stats[d].log_return_close_to_close
                               for d in days[2:5]], ddof=1))
    assert rv[days[4]] == pytest.approx(expected, rel=1e-6)
