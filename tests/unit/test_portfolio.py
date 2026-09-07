"""PortfolioStrategy: routes per-symbol bars and applies risk gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from topstep50k.engine.backtest import Backtester
from topstep50k.engine.clock import Clock
from topstep50k.engine.types import Bar, Instrument
from topstep50k.data.source import InMemoryBarSource
from topstep50k.portfolio import PerSymbolStrategy, PortfolioStrategy
from topstep50k.rules.topstep import combine_50k
from topstep50k.strategy.base import StrategyContext, TargetPosition


def _ts(minute):
    return datetime(2025, 1, 1, 13, 30, tzinfo=timezone.utc) + timedelta(minutes=minute)


def _bar(minute, price):
    return Bar(ts=_ts(minute), open=price, high=price + 0.25, low=price - 0.25,
               close=price, volume=10)


class _BuyOnceThenHold:
    """Open 1 long on bar 1, hold."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._opened = False

    def on_bar(self, bar, ctx):
        if not self._opened:
            self._opened = True
            return TargetPosition(symbol=self.symbol, qty=1, tag=f"open-{self.symbol}")
        return None


def _es_inst():
    return Instrument(symbol="ES", point_value=Decimal("50"),
                      tick_size=Decimal("0.25"), commission_per_side=Decimal("2.00"))


def _nq_inst():
    return Instrument(symbol="NQ", point_value=Decimal("20"),
                      tick_size=Decimal("0.25"), commission_per_side=Decimal("2.00"))


def test_portfolio_routes_per_symbol():
    """Each component sees only its own symbol's bars; orders for both
    fire on the appropriate bars."""
    es_bars = [_bar(i, 4500 + i * 0.25) for i in range(10)]
    nq_bars = [_bar(i, 15000 + i * 0.5) for i in range(10)]
    clk = Clock(_ts(0))
    src = InMemoryBarSource({"ES": es_bars, "NQ": nq_bars}, clk)
    rules = combine_50k()
    portfolio = PortfolioStrategy(components={
        "ES": _BuyOnceThenHold("ES"),
        "NQ": _BuyOnceThenHold("NQ"),
    })
    bt = Backtester(rules=rules, instruments={"ES": _es_inst(), "NQ": _nq_inst()},
                    strategy=portfolio)
    result = bt.run(clk, src)
    # We should see one fill per symbol from on_bar; any post-fact rule
    # breach (e.g. consistency, with only one trading day's data) is
    # orthogonal to what we're testing here -- routing.
    fills = [f for f in result.audit.of_kind("fill") if "forced" not in f.payload]
    syms = sorted(f.payload["symbol"] for f in fills)
    assert syms == ["ES", "NQ"], f"got fills={syms}"


def test_portfolio_unknown_symbol_skipped():
    es_bars = [_bar(i, 4500) for i in range(5)]
    clk = Clock(_ts(0))
    src = InMemoryBarSource({"ES": es_bars}, clk)
    rules = combine_50k()
    # Coordinator only registers NQ; ES bars must NOT trigger any orders.
    portfolio = PortfolioStrategy(components={"NQ": _BuyOnceThenHold("NQ")})
    bt = Backtester(rules=rules, instruments={"ES": _es_inst()}, strategy=portfolio)
    result = bt.run(clk, src)
    assert result.audit.of_kind("fill") == []
    assert result.audit.of_kind("order") == []


def test_portfolio_risk_gate_can_zero_targets():
    """If the risk gate returns empty, the engine sees no targets."""
    es_bars = [_bar(i, 4500 + i * 0.25) for i in range(10)]
    clk = Clock(_ts(0))
    src = InMemoryBarSource({"ES": es_bars}, clk)
    rules = combine_50k()
    portfolio = PortfolioStrategy(
        components={"ES": _BuyOnceThenHold("ES")},
        risk_gate=lambda targets, ctx: [],
    )
    bt = Backtester(rules=rules, instruments={"ES": _es_inst()}, strategy=portfolio)
    result = bt.run(clk, src)
    assert result.audit.of_kind("fill") == []


def test_portfolio_symbol_mismatch_raises():
    class _BadComp:
        symbol = "ZN"  # Mismatch
        def on_bar(self, bar, ctx): return None
    with pytest.raises(ValueError):
        PortfolioStrategy(components={"ES": _BadComp()})
