"""Intraday Momentum — first-30-minute return predicts same-session direction.

Academic rationale
──────────────────
Gao, Han, Li & Zhou (NBER Working Paper 2018, later J. Financial Economics):
  "Intraday Momentum: The First Half-Hour Return Predicts the Last Half-Hour
   Return." The first 30-minute return in S&P 500 futures significantly
   predicts the last 30-minute return (t-stat ~4.8 in their sample). Mechanism:
   opening auction price discovery and informed order flow persist intraday.
   Extended by Elaut et al. (2020): the signal is strongest in the first 30 min
   and generalises to equity-index futures (ES, NQ).

Pre-committed parameters (no data tuning)
─────────────────────────────────────────
  signal_minutes   : 30     — measure return over first 30 min of RTH
  min_signal_pct   : 0.0010 — 0.10% minimum move to take signal (noise filter)
  exit_time_local  : 15:45  — flatten 15 min before close (avoid close auction)
  stop_multiple    : 2.0    — stop at 2× signal_range (literature: wide stop)
  time_stop_minutes: 180    — abandon trade after 3 hours if neither TP nor stop

Direction
─────────
  First-30-min return > +min_signal_pct → LONG (momentum continuation)
  First-30-min return < -min_signal_pct → SHORT (momentum continuation)

Edge mechanism: documented in academic literature as a structural intraday
pattern driven by the persistence of informed order flow from the opening
auction. The signal is the raw first-half-hour return — no fitting to data.

Regime placement: TRENDING / directional sessions. Weaker on mean-reverting
days. Can coexist with ORB (ORB uses breakout of opening range, momentum uses
direction of first half-hour; they are correlated on trend days).
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
class _IMState:
    last_seen_date: date | None = None
    signal_start_price: float | None = None  # price at 09:30 open bar
    signal_computed: bool = False             # True once 10:00 signal is done
    entry_sent: bool = False
    done_today: bool = False
    direction: int = 0                        # +1 long, -1 short
    stop_price: float | None = None
    time_stop_utc: datetime | None = None
    target_qty: int = 0


@dataclass
class IntradayMomentum:
    """Fade the first-30-min return signal into a directional day trade.

    Parameters
    ----------
    symbol : bar-feed symbol.
    qty    : contracts per trade.
    signal_minutes     : minutes after open to measure signal return (30).
    min_signal_pct     : minimum |first-30-min return| to trade.
    stop_multiple      : stop distance = signal_range × multiple.
    time_stop_minutes  : minutes after entry to abandon if no result.
    exit_time_local    : flatten at or after this local time.
    tick_size          : minimum price increment.
    daily_filter       : optional gate; skip day if returns False.
    """

    symbol: str
    qty: int = 1
    signal_minutes: int = 30
    min_signal_pct: float = 0.0010
    stop_multiple: float = 2.0
    time_stop_minutes: int = 180
    exit_time_local: time = time(15, 45)
    session_open_local: time = time(9, 30)
    tick_size: float = 0.25
    daily_filter: Callable[[date], bool] | None = None

    _state: _IMState = field(init=False, default_factory=_IMState, repr=False)

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
                                          tag="im_flat_close")
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
                                       tag="im_time_stop")
            if s.direction > 0 and bar.low <= s.stop_price:
                s.target_qty = 0
                s.done_today = True
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="im_stop")
            if s.direction < 0 and bar.high >= s.stop_price:
                s.target_qty = 0
                s.done_today = True
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="im_stop")
            return None

        # ── Capture signal start price (first 09:30 bar open) ─────────────
        if s.signal_start_price is None and et.time() >= self.session_open_local:
            s.signal_start_price = bar.open
            return None

        # ── Compute signal after signal_minutes have elapsed ───────────────
        if (not s.signal_computed and s.signal_start_price is not None
                and not s.entry_sent):
            signal_end_et = datetime.combine(
                et.date(), self.session_open_local, tzinfo=EASTERN
            ) + timedelta(minutes=self.signal_minutes)
            if bar.ts >= signal_end_et.astimezone(bar.ts.tzinfo):
                s.signal_computed = True

                # Optional daily gate
                if self.daily_filter is not None:
                    if not self.daily_filter(et.date()):
                        s.done_today = True
                        return None

                ret = (bar.close - s.signal_start_price) / s.signal_start_price
                signal_range = abs(bar.close - s.signal_start_price)

                if abs(ret) < self.min_signal_pct:
                    s.done_today = True
                    return None

                stop_dist = self._round_to_tick(signal_range * self.stop_multiple)
                if ret > 0:
                    s.direction = +1
                    s.stop_price = self._round_to_tick(bar.close - stop_dist)
                else:
                    s.direction = -1
                    s.stop_price = self._round_to_tick(bar.close + stop_dist)

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
                                       tag="im_enter")

        return None
