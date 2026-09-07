"""Power Hour Continuation: entry timing, direction, stop, and
session-boundary correctness."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Bar, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.power_hour import PowerHourContinuation


ES = Instrument(symbol="ES", point_value=Decimal("50"),
                 tick_size=Decimal("0.25"), commission_per_side=Decimal("0"))


def _utc(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def _bar(ts, price, hi=None, lo=None):
    return Bar(ts=ts, open=price, high=hi if hi is not None else price + 0.25,
               low=lo if lo is not None else price - 0.25, close=price, volume=100)


def _run(strategy, bars, start_ts):
    clk = Clock(start_ts - timedelta(seconds=1))
    data = InMemoryBarSource({"ES": bars}, clk)
    pf = PortfolioStrategy(components={"ES": strategy})
    bt = Backtester(rules=combine_50k(), instruments={"ES": ES},
                    strategy=pf, audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, data)


def _one_rth_day(prices: list[float]):
    """June 3 2024 EDT RTH: 09:30-16:00 ET = 13:30-20:00 UTC, 390 1-min bars."""
    assert len(prices) == 390
    open_utc = _utc(2024, 6, 3, 13, 30)
    return [_bar(open_utc + timedelta(minutes=i), prices[i]) for i in range(390)]


def test_no_entry_before_power_hour_window():
    # Big move by minute 100, but power_hour_minutes=60 means the signal
    # isn't measured until 330 minutes in (390 - 60). Flat rest of day.
    prices = [5000.0] * 100 + [5020.0] * 290
    bars = _one_rth_day(prices)
    strat = PowerHourContinuation(symbol="ES", power_hour_minutes=60)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    # First fill (if any) must occur at/after the 330-min mark (15:00 UTC on this day... 13:30+330min=19:00 UTC)
    assert fills, "expected an entry once the power-hour signal fires"
    signal_boundary = _utc(2024, 6, 3, 13, 30) + timedelta(minutes=330)
    assert fills[0].ts >= signal_boundary


def test_long_direction_on_up_day():
    prices = [5000.0] * 330 + [5020.0] * 60
    bars = _one_rth_day(prices)
    strat = PowerHourContinuation(symbol="ES", power_hour_minutes=60, min_signal_pct=0.0010)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    sides = [f.payload["side"] for f in fills]
    assert sides and sides[0] == "buy"


def test_short_direction_on_down_day():
    prices = [5000.0] * 330 + [4980.0] * 60
    bars = _one_rth_day(prices)
    strat = PowerHourContinuation(symbol="ES", power_hour_minutes=60, min_signal_pct=0.0010)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    sides = [f.payload["side"] for f in fills]
    assert sides and sides[0] == "sell"


def test_no_entry_below_signal_threshold():
    # Tiny move -- below min_signal_pct -- should not trade.
    prices = [5000.0] * 389 + [5000.5]
    bars = _one_rth_day(prices)
    strat = PowerHourContinuation(symbol="ES", power_hour_minutes=60, min_signal_pct=0.01)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    assert fills == []


def test_flat_before_close():
    # Big up move into the power hour, never hits a stop -- must still be
    # flat by flat_before_close_minutes.
    prices = [5000.0] * 330 + [5020.0] * 60
    bars = _one_rth_day(prices)
    strat = PowerHourContinuation(symbol="ES", power_hour_minutes=60,
                                   min_signal_pct=0.0010, stop_multiple=100.0,
                                   flat_before_close_minutes=5)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    net = 0
    for f in fills:
        net += f.payload["qty"] if f.payload["side"] == "buy" else -f.payload["qty"]
    assert net == 0, "must end the session flat"


def test_daily_filter_skips_the_day():
    prices = [5000.0] * 330 + [5020.0] * 60
    bars = _one_rth_day(prices)
    strat = PowerHourContinuation(symbol="ES", power_hour_minutes=60,
                                   min_signal_pct=0.0010,
                                   daily_filter=lambda d: False)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    assert fills == []
