"""End-to-end backtest tests on synthetic deterministic data.

The synthetic price paths are constructed so each test produces an
EXACT, predictable outcome — pass, MLL hard breach, DLL soft halt,
or consistency fail. No randomness; no float drift in the assertions
(the rules engine uses Decimal).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Bar, Clock, Instrument
from topstep50k.rules import BreachReason, combine_50k
from topstep50k.strategy.base import Strategy, TargetPosition


ES = Instrument(
    symbol="ES",
    point_value=Decimal("50"),
    tick_size=Decimal("0.25"),
    commission_per_side=Decimal("0"),  # zero for predictable assertions
)


def utc(y, m, d, h=14):  # 14:00 UTC = active US session
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def make_bars(prices, start=utc(2026, 1, 5), step=timedelta(hours=1)):
    """Each price becomes (open=close=price, high=price+0.25, low=price-0.25)."""
    return [
        Bar(
            ts=start + i * step,
            open=p,
            high=p + 0.25,
            low=p - 0.25,
            close=p,
            volume=100,
        )
        for i, p in enumerate(prices)
    ]


class HoldLong(Strategy):
    """Buy `qty` contracts on the first bar, hold forever."""

    def __init__(self, qty: int = 1):
        self.qty = qty
        self.fired = False

    def on_bar(self, symbol, bar, ctx):
        if not self.fired:
            self.fired = True
            return [TargetPosition(symbol=symbol, qty=self.qty, tag="open")]
        return []


HoldOneLong = HoldLong  # back-compat alias


def run(strategy, bars_by_sym, start_ts=utc(2026, 1, 5)):
    clock = Clock(start_ts - timedelta(seconds=1))
    data = InMemoryBarSource(bars_by_sym, clock)
    bt = Backtester(
        rules=combine_50k(),
        instruments={"ES": ES},
        strategy=strategy,
        audit=InMemoryAuditLog(),
    )
    return bt.run(clock, data)


class TestPassPath:
    def test_strategy_that_makes_target_passes(self):
        # Buy at 4500, ride to 4565 over a single day (+65 points * $50 = +$3,250)
        # Single bar per "day" would skip end_day; spread across 6 days so
        # consistency check has multiple positive days.
        bars = []
        start = utc(2026, 1, 5, 14)
        # 6 trading days; each day price climbs +12 points (+$600/contract)
        # Need profit >= $3,000. 6 days * $600 = $3,600 total.
        # Also bars need to span midnight UTC for day rollover.
        for d in range(6):
            for h in range(3):
                price = 4500 + d * 12 + h * 4
                bars.append(Bar(
                    ts=start + timedelta(days=d, hours=h),
                    open=price, high=price + 0.25, low=price - 0.25,
                    close=price, volume=100,
                ))
        result = run(HoldOneLong(), {"ES": bars})
        # Strategy bought at bar[1].open (= 4504 on day 0 hour 1).
        # End-of-data close = 4500 + 5*12 + 2*4 = 4568.
        # PnL = (4568 - 4504) * $50 = $3,200. >= $3,000.
        assert result.passed
        assert result.breach is None
        assert result.final_equity == Decimal("53200")
        assert len(result.audit.of_kind("day_close")) == 6


class TestMaxLossHardBreach:
    def test_strategy_hits_mll_and_halts(self):
        # 5 contracts long. One bar with a $5+ adverse move = $1,250+ per
        # contract drop, easily skipping past DLL ($1,000) and into MLL
        # ($2,000) on the same bar — engine evaluates MLL before DLL on
        # the same equity reading, so MLL wins.
        start = utc(2026, 1, 5, 14)
        bars = []
        prices = [4500.0, 4500.0, 4490.0, 4480.0]  # -$5,000 across 2 bars
        for i, p in enumerate(prices):
            bars.append(Bar(
                ts=start + timedelta(hours=i),
                open=p, high=p + 0.25, low=p - 0.25,
                close=p, volume=100,
            ))
        result = run(HoldLong(qty=5), {"ES": bars})
        assert not result.passed
        assert result.breach is not None
        assert result.breach.reason is BreachReason.MAX_LOSS_LIMIT
        # Audit must have a fill that flattens after the breach.
        flatten_fills = [
            e for e in result.audit.of_kind("fill")
            if e.payload.get("forced") == "max_loss_limit"
        ]
        assert len(flatten_fills) == 1


class TestDailyLossOptOut:
    """The Combine doesn't enforce a DLL, so the default config never
    halts on intraday loss. We assert both branches: (a) default rules
    let a -$1,050 day pass with no breach; (b) explicitly configuring a
    DLL via dataclasses.replace() restores the soft-halt behaviour."""

    def _losing_day_bars(self):
        start = utc(2026, 1, 5, 14)
        bars = []
        # Day 1: 4500 -> 4479 = -21 pts * $50 = -$1,050
        for i, p in enumerate([4500, 4500, 4490, 4479]):
            bars.append(Bar(
                ts=start + timedelta(hours=i),
                open=p, high=p + 0.25, low=p - 0.25,
                close=p, volume=100,
            ))
        # Day 2 stays flat to let the engine see a day rollover
        for i, p in enumerate([4480, 4480, 4480]):
            bars.append(Bar(
                ts=start + timedelta(days=1, hours=i),
                open=p, high=p + 0.25, low=p - 0.25,
                close=p, volume=100,
            ))
        return bars

    def test_default_combine_does_not_halt_on_dll_sized_loss(self):
        result = run(HoldOneLong(), {"ES": self._losing_day_bars()})
        assert not result.passed
        assert result.breach is None
        dll_breaches = [
            e for e in result.audit.of_kind("breach")
            if e.payload["reason"] == "daily_loss_limit"
        ]
        assert dll_breaches == []

    def test_with_configured_dll_soft_halts(self):
        from dataclasses import replace
        bars = self._losing_day_bars()
        clock = Clock(utc(2026, 1, 5) - timedelta(seconds=1))
        data = InMemoryBarSource({"ES": bars}, clock)
        rules = replace(combine_50k(), daily_loss_limit=Decimal("1000"))
        bt = Backtester(rules=rules, instruments={"ES": ES},
                        strategy=HoldOneLong(), audit=InMemoryAuditLog())
        result = bt.run(clock, data)
        dll_breaches = [
            e for e in result.audit.of_kind("breach")
            if e.payload["reason"] == "daily_loss_limit"
        ]
        assert len(dll_breaches) == 1
        assert result.breach is None  # soft, not a Combine fail
