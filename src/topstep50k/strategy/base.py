"""Strategy interface.

A Strategy emits TARGET POSITIONS — not orders. The engine converts a
target into a market order at the NEXT bar's open, which structurally
prevents the strategy from filling at the current bar's close.

This is the cleanest no-look-ahead convention I know:
  bar[t] closes -> strategy.on_bar(t) -> target -> fill at open[t+1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from topstep50k.engine.clock import Clock
from topstep50k.engine.types import Bar


@dataclass(frozen=True)
class TargetPosition:
    symbol: str
    qty: int  # signed; positive long, negative short, 0 flat
    tag: str = ""


class StrategyContext(Protocol):
    """Read-only view a strategy gets per on_bar call.

    Everything time-sensitive is gated through this context so strategies
    cannot accidentally read raw bar arrays past `now()`.
    """

    @property
    def clock(self) -> Clock: ...
    def history(self, symbol: str, n: int) -> list[Bar]: ...
    def position(self, symbol: str) -> int: ...


class Strategy(Protocol):
    """Implementations override on_bar(). Return [] to mean "no change"."""

    def on_bar(self, symbol: str, bar: Bar, ctx: StrategyContext) -> list[TargetPosition]: ...
