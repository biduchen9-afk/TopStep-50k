"""Core typed primitives for the backtest engine.

Money lives in Decimal, prices in float (tick math is done via the
instrument's tick_size). Timestamps are timezone-aware datetimes, always.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is OrderSide.BUY else -1


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass(frozen=True)
class Instrument:
    """CME futures contract spec.

    point_value: $ per 1.00 of price movement (e.g. ES = $50)
    tick_size:   minimum price increment (e.g. ES = 0.25)
    tick_value:  point_value * tick_size (cached convenience)
    """

    symbol: str
    point_value: Decimal
    tick_size: Decimal
    commission_per_side: Decimal = Decimal("0")  # round-trip = 2 * this

    @property
    def tick_value(self) -> Decimal:
        return self.point_value * self.tick_size

    def pnl(self, side: OrderSide, qty: int, entry: float, exit: float) -> Decimal:
        """Realised PnL for a closed trade in account currency.

        Uses Decimal for the dollar conversion to keep the rules engine
        comparing apples-to-apples with the limit values.
        """
        delta_ticks = round((exit - entry) / float(self.tick_size))
        signed_ticks = side.sign * delta_ticks * qty
        return Decimal(signed_ticks) * self.tick_value


@dataclass(frozen=True)
class Bar:
    """OHLCV bar. The `ts` is the bar's CLOSE time — never the open.

    Engines must treat a Bar as observable only AFTER `ts`. Strategies
    that look at bar[t] and place orders for bar[t] without an explicit
    shift are committing look-ahead.
    """

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise ValueError(f"Bar.ts must be timezone-aware, got naive {self.ts!r}")


@dataclass(frozen=True)
class Order:
    ts: datetime  # creation time = the bar close that decided to send it
    symbol: str
    side: OrderSide
    qty: int
    type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    tag: str = ""  # free-form label for the audit trail


@dataclass(frozen=True)
class Fill:
    """An execution event. fill_ts is the bar at which the fill happens,
    which is structurally constrained to be > order.ts (see executor)."""

    order_ts: datetime
    fill_ts: datetime
    symbol: str
    side: OrderSide
    qty: int
    price: float
    commission: Decimal


def utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc)
