"""Compose many per-symbol Strategy instances into a single Strategy
that the Backtester can run.

Why a wrapper rather than running N Backtesters in parallel? Because all
TopStep rules score the SAME account equity. Running one engine pass
with all symbols multiplexed by ts is the only way the trailing MLL and
daily-loss check see the true combined PnL.

The wrapper only routes on_bar calls; it does NOT change the engine's
no-look-ahead model (orders still fill at the NEXT bar's open) or
override per-symbol position caps.

A second responsibility -- optional -- is correlation-aware position
gating. Construct with a `risk_gate` callable to scale or zero out
target positions based on whatever portfolio-level signal you choose
(open exposure, average pairwise correlation, etc.). The gate runs on
every emitted target BEFORE the engine sees it; the engine still
enforces position caps and rule breaches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from topstep50k.engine.types import Bar
from topstep50k.strategy.base import Strategy, StrategyContext, TargetPosition


class PerSymbolStrategy(Protocol):
    """A strategy that only ever wants its own symbol's bars."""

    symbol: str

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None: ...


RiskGate = Callable[[list[TargetPosition], StrategyContext], list[TargetPosition]]


@dataclass
class PortfolioStrategy:
    """Routes bars to per-symbol strategies; optionally re-shapes the
    resulting targets through a risk gate."""

    components: dict[str, PerSymbolStrategy]
    risk_gate: RiskGate | None = None

    def __post_init__(self) -> None:
        # Validate that every component claims a symbol that matches its key
        for sym, comp in self.components.items():
            claimed = getattr(comp, "symbol", None)
            if claimed is not None and claimed != sym:
                raise ValueError(
                    f"PortfolioStrategy: component for {sym!r} declares "
                    f"symbol={claimed!r}; mismatch."
                )

    def on_bar(self, symbol: str, bar: Bar, ctx: StrategyContext) -> list[TargetPosition]:
        comp = self.components.get(symbol)
        if comp is None:
            return []
        target = comp.on_bar(bar, ctx)
        if target is None:
            return []
        out = [target]
        if self.risk_gate is not None:
            out = self.risk_gate(out, ctx)
        return out


def _ensure_strategy_iface(_: Strategy) -> None: ...  # noqa: D401
_ensure_strategy_iface(PortfolioStrategy(components={}))  # static-shape assertion
