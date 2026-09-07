"""Account ledger: positions, equity curve, day boundaries.

Every state change goes through ONE method so the audit log can hook
a single place. The ledger never decides what to trade — strategies do
that — but it is the authoritative source for "what is the account worth
right now" and is what the rules engine evaluates.

Pricing uses average-cost accounting for the position book. On a fill:
* same direction: weighted-average the entry price, increase qty.
* opposite direction: realise PnL on the closed quantity, reduce qty;
  if oversold, flip side at the new fill price.

Mark-to-market: equity = starting_balance + realised_pnl
  + sum_over_open(unrealised_pnl(mark_price))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from topstep50k.engine.types import Fill, Instrument, OrderSide

_CENTRAL = ZoneInfo("America/Chicago")
_SESSION_ROLL_CT = time(16, 0)


@dataclass
class Position:
    qty: int = 0  # signed: positive = long, negative = short
    avg_price: float = 0.0


@dataclass
class Ledger:
    """One Combine account.

    Parameters
    ----------
    starting_balance : starting balance in account currency
    instruments      : mapping symbol -> Instrument (for PnL math)
    """

    starting_balance: Decimal
    instruments: dict[str, Instrument]

    realised_pnl: Decimal = field(init=False, default_factory=lambda: Decimal("0"))
    total_commission: Decimal = field(init=False, default_factory=lambda: Decimal("0"))
    positions: dict[str, Position] = field(init=False, default_factory=dict)

    # Day tracking — anchored on UTC date by default; the engine can swap
    # this for a session-calendar date if/when we add CME session windows.
    _current_day: date | None = field(init=False, default=None)
    _sod_equity: Decimal = field(init=False, default=Decimal("0"))
    daily_pnl: dict[date, Decimal] = field(init=False, default_factory=dict)

    # ------- core accounting --------------------------------------------

    def apply_fill(self, fill: Fill) -> Decimal:
        """Mutates positions and realised PnL. Returns the realised
        delta produced by this fill (may be zero on a pure open)."""
        inst = self.instruments[fill.symbol]
        pos = self.positions.setdefault(fill.symbol, Position())
        signed_qty = fill.side.sign * fill.qty
        realised_delta = Decimal("0")

        if pos.qty == 0:
            pos.qty = signed_qty
            pos.avg_price = fill.price
        elif (pos.qty > 0) == (signed_qty > 0):
            # Adding to the same direction: weighted average.
            new_qty = pos.qty + signed_qty
            pos.avg_price = (
                pos.avg_price * abs(pos.qty) + fill.price * abs(signed_qty)
            ) / abs(new_qty)
            pos.qty = new_qty
        else:
            # Reducing or flipping.
            closing = min(abs(pos.qty), abs(signed_qty))
            close_side = OrderSide.BUY if pos.qty > 0 else OrderSide.SELL
            realised_delta = inst.pnl(
                side=close_side,
                qty=closing,
                entry=pos.avg_price,
                exit=fill.price,
            )
            self.realised_pnl += realised_delta
            remaining = abs(pos.qty) - closing
            if remaining == 0 and abs(signed_qty) > closing:
                # Flipped: leftover opens a new position at fill price
                leftover = abs(signed_qty) - closing
                pos.qty = (1 if signed_qty > 0 else -1) * leftover
                pos.avg_price = fill.price
            else:
                pos.qty = (1 if pos.qty > 0 else -1) * remaining
                if pos.qty == 0:
                    pos.avg_price = 0.0

        self.total_commission += fill.commission
        return realised_delta

    def equity(self, marks: dict[str, float]) -> Decimal:
        """Total account equity at given mark prices.

        marks must include every symbol with a non-zero position.
        """
        unrealised = Decimal("0")
        for symbol, pos in self.positions.items():
            if pos.qty == 0:
                continue
            mark = marks[symbol]
            inst = self.instruments[symbol]
            side = OrderSide.BUY if pos.qty > 0 else OrderSide.SELL
            unrealised += inst.pnl(
                side=side, qty=abs(pos.qty), entry=pos.avg_price, exit=mark
            )
        return self.starting_balance + self.realised_pnl + unrealised - self.total_commission

    def abs_position(self, symbol: str) -> int:
        pos = self.positions.get(symbol)
        return abs(pos.qty) if pos else 0

    # ------- day boundary handling --------------------------------------

    def begin_day(self, day: date, current_equity: Decimal) -> None:
        """Called by the engine at the FIRST bar of a new trading day,
        AFTER any rollover housekeeping. Records start-of-day equity for
        DLL comparisons."""
        self._current_day = day
        self._sod_equity = current_equity

    def end_day(self, day: date, eod_equity: Decimal) -> Decimal:
        """Books the day's realised + unrealised PnL into daily_pnl and
        returns it. Engine uses this to feed the trailing-MLL anchor
        update and the end-of-cycle consistency check."""
        delta = eod_equity - self._sod_equity
        self.daily_pnl[day] = self.daily_pnl.get(day, Decimal("0")) + delta
        return self.daily_pnl[day]

    @property
    def sod_equity(self) -> Decimal:
        return self._sod_equity

    @property
    def current_day(self) -> date | None:
        return self._current_day


def trading_day(ts: datetime) -> date:
    """CME/Combine session-day bucketing: a trading day runs 16:00 CT ->
    16:00 CT (the daily maintenance-halt boundary; CME reopens at 17:00 CT
    but nothing trades 16:00-17:00 CT so the exact cutoff within that gap
    doesn't matter). A bar at or after 16:00 CT belongs to the NEXT trade
    date -- matching TopStep's actual Combine day reset and the standard
    NinjaTrader/Sierra "trade date" convention.

    Previously this used raw UTC-calendar-date bucketing, which does not
    line up with either TopStep's real reset or the exchange session
    boundary -- see the look-ahead/leakage audit that replaced this.
    """
    ct = ts.astimezone(_CENTRAL)
    d = ct.date()
    if ct.time() >= _SESSION_ROLL_CT:
        d = d + timedelta(days=1)
    return d
