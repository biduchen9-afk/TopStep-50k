"""Intraday Momentum with Take-Profit (1:1 RR) — symmetric target/stop.

This is a disciplined variant of IntradayMomentum (#5) that adds an explicit
take-profit at 1× the signal range. The 1:1 RR construction ensures:
  - max_loss ≈ avg_win (by construction → max_loss/avg_win < 2.0)
  - Lower variance per trade (both wins and losses are capped)
  - t_stat depends on whether win rate > 50% consistently

Pre-committed parameters
────────────────────────
  signal_minutes    : 30    — first half-hour signal (same as #5, Gao et al.)
  min_signal_pct    : 0.0010 — 0.10% minimum signal size
  stop_multiple     : 1.0    — stop at 1× signal_range
  target_multiple   : 1.0    — take-profit at 1× signal_range (NEW vs #5)
  time_stop_minutes : 120    — flatten after 2 hours if neither TP nor stop hit
  exit_time_local   : 15:45  — flatten before close

Direction: same as IntradayMomentum (first 30-min return predicts continuation,
Gao, Han, Li & Zhou 2018, J. Financial Economics).

The 1:1 RR means breakeven win rate is 50% + commission margin. Academic
work on intraday momentum suggests win rates of 52-55% for this signal in
equity index futures, which would give positive EV with symmetric payoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from topstep50k.engine.types import Bar
from topstep50k.strategy.base import StrategyContext, TargetPosition


EASTERN = ZoneInfo("America/New_York")


@dataclass
class _IMTPState:
    last_seen_date: date | None = None
    signal_start_price: float | None = None
    signal_computed: bool = False
    entry_sent: bool = False
    done_today: bool = False
    direction: int = 0
    stop_price: float | None = None
    target_price: float | None = None
    time_stop_utc: datetime | None = None
    target_qty: int = 0


@dataclass
class IntradayMomentumTP:
    """Intraday Momentum with symmetric 1:1 take-profit and stop-loss.

    Parameters
    ----------
    symbol              : bar-feed symbol.
    qty                 : contracts per trade.
    signal_minutes      : minutes after open to measure signal return.
    min_signal_pct      : minimum |first-N-min return| to trade.
    stop_multiple       : stop distance = signal_range × multiple.
    target_multiple     : TP distance = signal_range × multiple (symmetric = 1.0).
    time_stop_minutes   : minutes after entry to flatten if no exit.
    exit_time_local     : hard flatten time.
    tick_size           : minimum price increment.
    daily_filter        : optional gate.
    """

    symbol: str
    qty: int = 1
    signal_minutes: int = 30
    min_signal_pct: float = 0.0010
    stop_multiple: float = 1.0
    target_multiple: float = 1.0
    time_stop_minutes: int = 120
    exit_time_local: time = time(15, 45)
    session_open_local: time = time(9, 30)
    tick_size: float = 0.25
    daily_filter: Callable[[date], bool] | None = None

    _state: _IMTPState = field(init=False, default_factory=_IMTPState, repr=False)

    def _round_to_tick(self, price: float) -> float:
        ts = self.tick_size
        return round(round(price / ts) * ts, 10)

    def _maybe_roll_day(self, bar: Bar) -> None:
        et = bar.ts.astimezone(EASTERN)
        d = et.date()
        s = self._state
        if s.last_seen_date == d:
            return
        s.last_seen_date = d
        s.signal_start_price = None
        s.signal_computed = False
        s.entry_sent = False
        s.done_today = False
        s.direction = 0
        s.stop_price = None
        s.target_price = None
        s.time_stop_utc = None
        s.target_qty = 0

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        self._maybe_roll_day(bar)
        s = self._state
        et = bar.ts.astimezone(EASTERN)

        # ── Force flat before close ────────────────────────────────────────
        if et.time() >= self.exit_time_local:
            if not s.done_today:
                s.done_today = True
                if ctx.position(self.symbol) != 0 or s.target_qty != 0:
                    s.target_qty = 0
                    return TargetPosition(symbol=self.symbol, qty=0,
                                          tag="imtp_flat_close")
            return None

        if s.done_today:
            return None

        cur_qty = ctx.position(self.symbol)

        # ── Manage open position ───────────────────────────────────────────
        if cur_qty != 0 and s.direction != 0:
            if s.time_stop_utc and bar.ts >= s.time_stop_utc:
                s.target_qty = 0
                s.done_today = True
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="imtp_time_stop")
            if s.direction > 0:
                if bar.high >= s.target_price:
                    s.target_qty = 0
                    s.done_today = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="imtp_target")
                if bar.low <= s.stop_price:
                    s.target_qty = 0
                    s.done_today = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="imtp_stop")
            else:
                if bar.low <= s.target_price:
                    s.target_qty = 0
                    s.done_today = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="imtp_target")
                if bar.high >= s.stop_price:
                    s.target_qty = 0
                    s.done_today = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="imtp_stop")
            return None

        # ── Capture signal start (first 09:30 bar open) ───────────────────
        if s.signal_start_price is None and et.time() >= self.session_open_local:
            s.signal_start_price = bar.open
            return None

        # ── Compute signal after signal_minutes ───────────────────────────
        if (not s.signal_computed and s.signal_start_price is not None
                and not s.entry_sent):
            signal_end_et = datetime.combine(
                et.date(), self.session_open_local, tzinfo=EASTERN
            ) + timedelta(minutes=self.signal_minutes)
            if bar.ts >= signal_end_et.astimezone(bar.ts.tzinfo):
                s.signal_computed = True

                if self.daily_filter is not None:
                    if not self.daily_filter(et.date()):
                        s.done_today = True
                        return None

                ret = (bar.close - s.signal_start_price) / s.signal_start_price
                signal_range = abs(bar.close - s.signal_start_price)

                if abs(ret) < self.min_signal_pct:
                    s.done_today = True
                    return None

                stop_dist   = self._round_to_tick(signal_range * self.stop_multiple)
                target_dist = self._round_to_tick(signal_range * self.target_multiple)

                if ret > 0:
                    s.direction = +1
                    s.stop_price   = self._round_to_tick(bar.close - stop_dist)
                    s.target_price = self._round_to_tick(bar.close + target_dist)
                else:
                    s.direction = -1
                    s.stop_price   = self._round_to_tick(bar.close + stop_dist)
                    s.target_price = self._round_to_tick(bar.close - target_dist)

                entry_et = datetime.combine(et.date(), self.session_open_local,
                                             tzinfo=EASTERN) + timedelta(
                                                 minutes=self.signal_minutes)
                s.time_stop_utc = (
                    entry_et + timedelta(minutes=self.time_stop_minutes)
                ).astimezone(bar.ts.tzinfo)

                s.entry_sent = True
                s.target_qty = s.direction * self.qty
                return TargetPosition(symbol=self.symbol,
                                       qty=s.direction * self.qty,
                                       tag="imtp_enter")

        return None
