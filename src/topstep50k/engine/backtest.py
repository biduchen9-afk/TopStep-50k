"""Event-driven backtest loop.

Flow per bar t for symbol s:
  1. clock.advance_to(bar.ts)               # bar is now "observable"
  2. Mark-to-market with bar.close, update equity
  3. Check intraday rules (DLL soft, MLL hard, position cap)
     - On HARD breach: flatten everything, halt
     - On SOFT breach: flatten everything, mark "no new entries today"
  4. If day rolled: book yesterday's PnL, update MLL trail, reset DLL gate
  5. Strategy.on_bar(t) -> target positions
  6. Compute order deltas (target - current). Defer fills to bar t+1 open.
  7. At the START of bar t+1: any deferred orders fill at open[t+1].

The deferral is what enforces no-look-ahead at the execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from topstep50k.audit import AuditEvent, AuditLog, InMemoryAuditLog
from topstep50k.engine.clock import Clock

if TYPE_CHECKING:
    from topstep50k.data.source import BarStream
from topstep50k.engine.ledger import Ledger, trading_day
from topstep50k.engine.types import Bar, Fill, Instrument, OrderSide
from topstep50k.rules.topstep import (
    BreachReason,
    BreachType,
    RuleBreach,
    TopstepRules,
    TrailingMLL,
)
from topstep50k.strategy.base import Strategy, StrategyContext, TargetPosition


@dataclass
class _PendingOrder:
    """A market order queued at bar t, to fill at bar t+1's open."""

    symbol: str
    side: OrderSide
    qty: int
    decided_at: object  # clock ts when strategy emitted it
    tag: str = ""


class _Context:
    """Concrete StrategyContext bound to one bar event."""

    def __init__(self, clock: Clock, data, ledger: Ledger) -> None:
        self._clock = clock
        self._data = data
        self._ledger = ledger

    @property
    def clock(self) -> Clock:
        return self._clock

    def history(self, symbol: str, n: int) -> list[Bar]:
        return self._data.history(symbol, n)

    def position(self, symbol: str) -> int:
        pos = self._ledger.positions.get(symbol)
        return pos.qty if pos else 0


@dataclass
class BacktestResult:
    final_equity: Decimal
    realised_pnl: Decimal
    commissions: Decimal
    breach: RuleBreach | None
    passed: bool
    daily_pnl: dict[date, Decimal]
    audit: AuditLog
    equity_curve: list[tuple[object, Decimal]]


@dataclass
class Backtester:
    rules: TopstepRules
    instruments: dict[str, Instrument]
    strategy: Strategy
    audit: AuditLog = field(default_factory=InMemoryAuditLog)

    def run(self, clock: Clock, data: "BarStream") -> BacktestResult:
        ledger = Ledger(starting_balance=self.rules.starting_balance, instruments=self.instruments)
        mll = TrailingMLL(self.rules.starting_balance, self.rules.max_loss_limit_distance)
        ctx = _Context(clock, data, ledger)

        pending: list[_PendingOrder] = []
        last_marks: dict[str, float] = {}
        day_state: dict[str, object] = {"current": None, "halt_entries": False}
        equity_curve: list[tuple[object, Decimal]] = []
        breach_result: RuleBreach | None = None

        def emit(kind: str, **payload):
            self.audit.record(AuditEvent(ts=clock.now(), kind=kind, payload=payload))

        def flatten_all(reason: str):
            """Place synthetic offsetting fills at the current mark for every
            open position. Used on rule breaches and end-of-stream."""
            for sym, pos in list(ledger.positions.items()):
                if pos.qty == 0:
                    continue
                mark = last_marks.get(sym)
                if mark is None:
                    continue
                inst = self.instruments[sym]
                side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
                fill = Fill(
                    order_ts=clock.now(),
                    fill_ts=clock.now(),
                    symbol=sym,
                    side=side,
                    qty=abs(pos.qty),
                    price=mark,
                    commission=inst.commission_per_side * abs(pos.qty),
                )
                realised = ledger.apply_fill(fill)
                emit("fill", symbol=sym, side=side.value, qty=fill.qty,
                     price=fill.price, realised=realised, forced=reason)

        emit("run_start",
             starting_balance=self.rules.starting_balance,
             mll_distance=self.rules.max_loss_limit_distance,
             daily_loss_limit=self.rules.daily_loss_limit)

        for symbol, bar in data.stream(clock):
            # ---- 0. fill any pending orders at THIS bar's open --------
            if pending:
                kept: list[_PendingOrder] = []
                for po in pending:
                    if po.symbol != symbol:
                        kept.append(po)
                        continue
                    if day_state["halt_entries"]:
                        # Soft breach earlier today: only allow flattening
                        cur_qty = ledger.positions.get(po.symbol)
                        current = cur_qty.qty if cur_qty else 0
                        signed = po.side.sign * po.qty
                        # Allow only orders that REDUCE absolute exposure
                        if abs(current + signed) > abs(current):
                            emit("order_rejected", symbol=po.symbol,
                                 reason="halt_entries", tag=po.tag)
                            continue
                    inst = self.instruments[po.symbol]
                    fill = Fill(
                        order_ts=po.decided_at,
                        fill_ts=clock.now(),  # will be advanced just below
                        symbol=po.symbol,
                        side=po.side,
                        qty=po.qty,
                        price=bar.open,
                        commission=inst.commission_per_side * po.qty,
                    )
                    realised = ledger.apply_fill(fill)
                    emit("fill", symbol=po.symbol, side=po.side.value, qty=po.qty,
                         price=bar.open, realised=realised, tag=po.tag)
                pending = kept

            # ---- 1. observe bar ---------------------------------------
            clock.advance_to(bar.ts)
            last_marks[symbol] = bar.close

            # ---- 2. day rollover --------------------------------------
            today = trading_day(bar.ts)
            if day_state["current"] is None:
                day_state["current"] = today
                ledger.begin_day(today, ledger.equity(last_marks))
                emit("day_open", day=today.isoformat(), sod_equity=ledger.sod_equity)
            elif today != day_state["current"]:
                prev_day = day_state["current"]
                eod_eq = ledger.equity(last_marks)
                day_delta = ledger.end_day(prev_day, eod_eq)
                mll.update_end_of_day(eod_eq)
                emit("day_close", day=prev_day.isoformat(),
                     eod_equity=eod_eq, day_pnl=day_delta,
                     mll_anchor=mll.anchor, mll_locked=mll.locked)
                day_state["current"] = today
                day_state["halt_entries"] = False
                ledger.begin_day(today, eod_eq)
                emit("day_open", day=today.isoformat(), sod_equity=eod_eq)

            # ---- 3. mark and check rules ------------------------------
            equity = ledger.equity(last_marks)
            equity_curve.append((bar.ts, equity))

            mll_breach = self.rules.check_max_loss(equity, mll.anchor)
            if mll_breach is not None:
                breach_result = mll_breach
                emit("breach", **{
                    "reason": mll_breach.reason.value,
                    "type": mll_breach.breach_type.value,
                    "observed": mll_breach.observed,
                    "limit": mll_breach.limit,
                })
                flatten_all("max_loss_limit")
                break

            dll_breach = self.rules.check_daily_loss(equity, ledger.sod_equity)
            if dll_breach is not None and not day_state["halt_entries"]:
                day_state["halt_entries"] = True
                emit("breach", reason=dll_breach.reason.value,
                     type=dll_breach.breach_type.value,
                     observed=dll_breach.observed, limit=dll_breach.limit)
                flatten_all("daily_loss_limit")
                continue  # no strategy decisions for the rest of the day

            # ---- 4. strategy decision --------------------------------
            targets = self.strategy.on_bar(symbol, bar, ctx)
            for target in targets:
                cur = ctx.position(target.symbol)
                delta = target.qty - cur
                if delta == 0:
                    continue
                # Position-cap check on the resulting position size
                cap_breach = self.rules.check_position_size(target.symbol, abs(target.qty))
                if cap_breach is not None:
                    emit("order_rejected", symbol=target.symbol,
                         reason="position_cap", target_qty=target.qty,
                         cap=cap_breach.limit, tag=target.tag)
                    continue
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                pending.append(_PendingOrder(
                    symbol=target.symbol,
                    side=side,
                    qty=abs(delta),
                    decided_at=clock.now(),
                    tag=target.tag,
                ))
                emit("order", symbol=target.symbol, side=side.value,
                     qty=abs(delta), tag=target.tag, target=target.qty)

        # ---- end of stream ----------------------------------------------
        # Close last day if we didn't break out on a breach.
        if breach_result is None and day_state["current"] is not None:
            eod = ledger.equity(last_marks)
            day_delta = ledger.end_day(day_state["current"], eod)
            mll.update_end_of_day(eod)
            emit("day_close", day=day_state["current"].isoformat(),
                 eod_equity=eod, day_pnl=day_delta,
                 mll_anchor=mll.anchor, mll_locked=mll.locked)

        # Consistency check (end-of-cycle)
        consistency = self.rules.check_consistency(ledger.daily_pnl)
        if consistency is not None and breach_result is None:
            breach_result = consistency
            emit("breach", reason=consistency.reason.value,
                 type=consistency.breach_type.value,
                 observed=consistency.observed, limit=consistency.limit)

        final_equity = ledger.equity(last_marks)
        passed = (
            breach_result is None
            and self.rules.reached_profit_target(final_equity)
        )
        emit("run_end", final_equity=final_equity,
             realised=ledger.realised_pnl, passed=passed,
             breach=(breach_result.reason.value if breach_result else None))

        return BacktestResult(
            final_equity=final_equity,
            realised_pnl=ledger.realised_pnl,
            commissions=ledger.total_commission,
            breach=breach_result,
            passed=passed,
            daily_pnl=dict(ledger.daily_pnl),
            audit=self.audit,
            equity_curve=equity_curve,
        )
