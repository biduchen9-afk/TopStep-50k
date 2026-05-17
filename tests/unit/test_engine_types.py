from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from topstep50k.engine import Bar, Clock, Instrument, LookAheadError, OrderSide


ES = Instrument(
    symbol="ES",
    point_value=Decimal("50"),
    tick_size=Decimal("0.25"),
    commission_per_side=Decimal("1.50"),
)


def utc(y, m, d, h=0, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


class TestInstrumentPnL:
    def test_long_winner_one_tick(self):
        pnl = ES.pnl(OrderSide.BUY, qty=1, entry=4500.00, exit=4500.25)
        assert pnl == Decimal("12.50")  # 1 tick * $12.50/tick

    def test_short_winner(self):
        pnl = ES.pnl(OrderSide.SELL, qty=2, entry=4500.00, exit=4499.50)
        # 2 ticks favorable * 2 contracts * $12.50 = $50
        assert pnl == Decimal("50.00")

    def test_loser_is_negative(self):
        pnl = ES.pnl(OrderSide.BUY, qty=1, entry=4500.00, exit=4499.00)
        assert pnl == Decimal("-50.00")


class TestBar:
    def test_requires_tzaware_ts(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            Bar(ts=datetime(2026, 1, 1), open=1, high=1, low=1, close=1, volume=0)


class TestClock:
    def test_initialises_at_start(self):
        clk = Clock(utc(2026, 1, 1, 9, 30))
        assert clk.now() == utc(2026, 1, 1, 9, 30)

    def test_advance_forward_ok(self):
        clk = Clock(utc(2026, 1, 1, 9, 30))
        clk.advance_to(utc(2026, 1, 1, 9, 31))
        assert clk.now() == utc(2026, 1, 1, 9, 31)

    def test_advance_backward_blocked(self):
        clk = Clock(utc(2026, 1, 1, 9, 30))
        with pytest.raises(LookAheadError):
            clk.advance_to(utc(2026, 1, 1, 9, 29))

    def test_visible_at_now(self):
        clk = Clock(utc(2026, 1, 1, 9, 30))
        clk.assert_visible(utc(2026, 1, 1, 9, 30), "bar")

    def test_visible_in_past(self):
        clk = Clock(utc(2026, 1, 1, 9, 30))
        clk.assert_visible(utc(2026, 1, 1, 9, 29), "bar")

    def test_future_blocks(self):
        clk = Clock(utc(2026, 1, 1, 9, 30))
        with pytest.raises(LookAheadError, match="Look-ahead"):
            clk.assert_visible(utc(2026, 1, 1, 9, 31), "bar")

    def test_naive_ts_rejected(self):
        clk = Clock(utc(2026, 1, 1))
        with pytest.raises(ValueError):
            clk.assert_visible(datetime(2026, 1, 1), "bar")
