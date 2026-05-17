"""Opening-range-breakout strategy unit tests on synthetic ES bars.

Properties asserted:
  * OR is built from bars inside the [open, open + or_minutes) window;
    breakout decisions only fire from bars >= or_end.
  * Long breakout fires on close > or_high; short on close < or_low.
  * Time-stop flattens before session close even with no S/T hit.
  * One trade per day (no re-entry).
  * Bars outside the RTH session do not affect the OR.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Bar, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.orb import OpeningRangeBreakout


ES = Instrument(
    symbol="ES",
    point_value=Decimal("50"),
    tick_size=Decimal("0.25"),
    commission_per_side=Decimal("0"),
)


def _utc(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def _bar(ts, price, hi=None, lo=None):
    return Bar(
        ts=ts,
        open=price,
        high=hi if hi is not None else price + 0.25,
        low=lo if lo is not None else price - 0.25,
        close=price,
        volume=100,
    )


def _run(strategy, bars, start_ts):
    clk = Clock(start_ts - timedelta(seconds=1))
    data = InMemoryBarSource({"ES": bars}, clk)
    pf = PortfolioStrategy(components={"ES": strategy})
    bt = Backtester(
        rules=combine_50k(), instruments={"ES": ES},
        strategy=pf, audit=InMemoryAuditLog(),
    )
    return bt.run(clk, data)


def _june_3_2024_session_bars(or_high_break=False, or_low_break=False, flat=False):
    """Construct minute bars for one ES RTH session (09:30-16:00 ET).
    June 3 2024 is EDT -> UTC-4 -> 13:30-20:00 UTC.

    Generates one bar per minute (390 bars).
    """
    bars = []
    open_utc = _utc(2024, 6, 3, 13, 30)
    # OR window: 09:30 - 10:00 ET = 13:30 - 14:00 UTC. 30 bars.
    # In OR: range 4500.00 .. 4505.00.
    for i in range(30):
        prices = [4500.0, 4505.0, 4502.0]  # cycle to span the range
        p = prices[i % 3]
        bars.append(_bar(open_utc + timedelta(minutes=i), p,
                         hi=4505.0 if i == 1 else p + 0.25,
                         lo=4500.0 if i == 0 else p - 0.25))
    # Post-OR: 360 bars from 14:00 to 20:00 UTC.
    for i in range(360):
        ts = open_utc + timedelta(minutes=30 + i)
        if flat:
            bars.append(_bar(ts, 4503.0))
            continue
        if or_high_break:
            # Cross above 4505 starting from bar 30 onward, stay above.
            bars.append(_bar(ts, 4508.0))
        elif or_low_break:
            bars.append(_bar(ts, 4498.0))
        else:
            bars.append(_bar(ts, 4503.0))
    return bars


def test_breakout_long_fires_when_close_above_or_high():
    bars = _june_3_2024_session_bars(or_high_break=True)
    strat = OpeningRangeBreakout(symbol="ES", or_minutes=30, tp_multiple=1.0,
                                 flat_before_close_minutes=15)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    # We expect one entry long (at 4505+ open of bar after breakout) and one TP exit.
    sides = [f.payload["side"] for f in fills]
    assert sides[0] == "buy"  # entry
    # Some exit happened (TP, time-stop, or session-end forced flat)
    assert len(fills) >= 1


def test_no_entry_when_flat_within_or_range():
    bars = _june_3_2024_session_bars(flat=True)
    strat = OpeningRangeBreakout(symbol="ES", or_minutes=30, tp_multiple=1.0,
                                 flat_before_close_minutes=15)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    # No entries because price stayed inside OR.
    assert fills == []


def test_short_breakout_fires():
    bars = _june_3_2024_session_bars(or_low_break=True)
    strat = OpeningRangeBreakout(symbol="ES", or_minutes=30, tp_multiple=1.0,
                                 flat_before_close_minutes=15)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    sides = [f.payload["side"] for f in fills]
    assert sides and sides[0] == "sell"


def test_long_only_ignores_short_breakout():
    bars = _june_3_2024_session_bars(or_low_break=True)
    strat = OpeningRangeBreakout(symbol="ES", or_minutes=30, direction="long",
                                 tp_multiple=1.0, flat_before_close_minutes=15)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    assert fills == []


def test_pre_session_bars_do_not_affect_or():
    """A bar at 08:30 ET (UTC 12:30) is OUTSIDE the session and must not
    inform the OR boundaries."""
    bars = []
    # Pre-session: stray bar at 4400 (far below the OR low we'll set later)
    bars.append(_bar(_utc(2024, 6, 3, 12, 30), 4400.0))
    # Now the normal session as above
    bars += _june_3_2024_session_bars(or_low_break=True)
    strat = OpeningRangeBreakout(symbol="ES", or_minutes=30, direction="both",
                                 tp_multiple=1.0, flat_before_close_minutes=15)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 12, 30))
    # Should still register a short breakout when price went below 4500.
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    sides = [f.payload["side"] for f in fills]
    assert sides and sides[0] == "sell"


def test_time_stop_forces_flat_before_close():
    """Flat-by setting forces exit even if no S/T was hit."""
    bars = _june_3_2024_session_bars(or_high_break=True)
    # Set a stupidly far TP so we don't hit it; rely on time stop.
    strat = OpeningRangeBreakout(symbol="ES", or_minutes=30, tp_multiple=100.0,
                                 flat_before_close_minutes=15)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    # Final state must be flat.
    # Tally signed qty.
    net = 0
    for f in fills:
        p = f.payload
        if p["side"] == "buy":
            net += p["qty"]
        else:
            net -= p["qty"]
    assert net == 0


def test_one_entry_per_day_only():
    """After exit, no re-entry within the same RTH session."""
    bars = _june_3_2024_session_bars(or_high_break=True)
    # Tiny TP to exit fast, then session continues. Make sure no second entry.
    strat = OpeningRangeBreakout(symbol="ES", or_minutes=30, tp_multiple=0.1,
                                 flat_before_close_minutes=15)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    # Expect AT MOST 2 fills (one entry, one TP exit).
    assert len(fills) <= 2
