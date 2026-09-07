"""Overnight-drift strategy unit tests.

Properties asserted:
  * Enters long ~entry_offset_minutes before RTH close.
  * Exits long ~exit_offset_minutes after next RTH open.
  * Holds position continuously overnight (no extra entries/exits).
  * Defaults to long-only (qty=+1).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Bar, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.overnight_drift import OvernightDrift


ES = Instrument(symbol="ES", point_value=Decimal("50"),
                tick_size=Decimal("0.25"),
                commission_per_side=Decimal("0"))


def _utc(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def _bar(ts, price):
    return Bar(ts=ts, open=price, high=price + 0.25, low=price - 0.25,
               close=price, volume=100)


def _run(strategy, bars, start_ts):
    clk = Clock(start_ts - timedelta(seconds=1))
    data = InMemoryBarSource({"ES": bars}, clk)
    pf = PortfolioStrategy(components={"ES": strategy})
    bt = Backtester(rules=combine_50k(), instruments={"ES": ES},
                    strategy=pf, audit=InMemoryAuditLog(),
                    combine_enforcement=False)
    return bt.run(clk, data)


def _two_day_bars(price_close_day1=4500.0, price_open_day2=4510.0):
    """Two consecutive RTH days with overnight gap.

    Day 1: June 3 2024 (Mon EDT). 13:30 - 20:00 UTC = 09:30 - 16:00 ET.
    Overnight: 20:00 UTC day 1 -> 13:30 UTC day 2. We emit a few bars
    overnight at the day-1 close price (no price moves overnight).
    Day 2: June 4 2024. Opens at 13:30 UTC at price_open_day2.

    Per-minute bars.
    """
    bars = []
    d1_open = _utc(2024, 6, 3, 13, 30)
    # Day 1 RTH: 390 bars at 4500
    for i in range(390):
        bars.append(_bar(d1_open + timedelta(minutes=i), price_close_day1))
    # Overnight gap: bars from 20:00 day 1 to 13:30 day 2 (17.5 hours)
    overnight_start = _utc(2024, 6, 3, 20, 1)
    overnight_minutes = (17 * 60) + 30
    for i in range(overnight_minutes):
        ts = overnight_start + timedelta(minutes=i)
        # Hold flat overnight at the day-1 close until we approach day-2 open,
        # then snap to day-2 open price for the last bar.
        p = (price_open_day2
             if i >= overnight_minutes - 1 else price_close_day1)
        bars.append(_bar(ts, p))
    # Day 2 RTH: 390 bars at the day-2 open price
    d2_open = _utc(2024, 6, 4, 13, 30)
    for i in range(390):
        bars.append(_bar(d2_open + timedelta(minutes=i), price_open_day2))
    return bars


def test_enters_long_near_close_exits_after_next_open():
    bars = _two_day_bars()
    strat = OvernightDrift(symbol="ES", entry_offset_minutes=5,
                            exit_offset_minutes=5)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    sides = [f.payload["side"] for f in fills]
    # First fill should be the entry (buy)
    assert sides and sides[0] == "buy"
    # Last (or near-last) fill should be the exit (sell)
    assert "sell" in sides


def test_holds_through_overnight():
    bars = _two_day_bars()
    strat = OvernightDrift(symbol="ES", entry_offset_minutes=5,
                            exit_offset_minutes=5)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    sides = [f.payload["side"] for f in fills]
    # Day-1 close: BUY. Day-2 open: SELL. Day-2 close: BUY again.
    assert sides == ["buy", "sell", "buy"], (
        f"expected one full overnight cycle plus a re-entry, got {sides}"
    )


def test_qty_minus_one_goes_short():
    bars = _two_day_bars()
    strat = OvernightDrift(symbol="ES", qty=-1, entry_offset_minutes=5,
                            exit_offset_minutes=5)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    sides = [f.payload["side"] for f in fills]
    assert sides and sides[0] == "sell"  # short entry
    assert "buy" in sides  # cover exit


def test_exit_offset_zero_still_exits_at_open():
    # Regression test: exit_offset_minutes=0 used to make the exit
    # window `[session_open, session_open + 0)` -- an EMPTY interval --
    # so the position never flattened at all and rode forever. It must
    # exit at exactly session_open (offset 0) instead.
    bars = _two_day_bars()
    strat = OvernightDrift(symbol="ES", entry_offset_minutes=5,
                            exit_offset_minutes=0)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    sides = [f.payload["side"] for f in fills]
    assert sides == ["buy", "sell", "buy"], (
        f"exit_offset_minutes=0 must still produce a full overnight "
        f"cycle plus re-entry, got {sides}"
    )
    exit_fill = fills[1]
    assert exit_fill.payload["side"] == "sell"
    assert exit_fill.ts == _utc(2024, 6, 4, 13, 30)  # exactly at RTH open


def test_exit_does_not_flap_on_the_afternoon_reentry():
    # Regression test for a bug introduced while fixing the above: an
    # unbounded `bar.ts >= cur_exit_utc` check (no upper bound at all)
    # stays true for the rest of the day once the morning's exit trigger
    # passes, so the SAME afternoon's fresh entry (for the next overnight
    # cycle) would get immediately flattened again on its very next bar.
    # Exactly 3 fills total (buy/sell/buy) -- no extra flap on day 1.
    bars = _two_day_bars()
    for exit_offset in (0, 5, 15):
        strat = OvernightDrift(symbol="ES", entry_offset_minutes=5,
                                exit_offset_minutes=exit_offset)
        result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
        fills = result.audit.of_kind("fill")
        sides = [f.payload["side"] for f in fills]
        assert sides == ["buy", "sell", "buy"], (
            f"exit_offset_minutes={exit_offset}: expected no flap, got {sides}"
        )
