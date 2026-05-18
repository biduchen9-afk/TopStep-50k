"""Bollinger-band mean reversion on intraday bars.

Definition
----------
For each US RTH session, maintain a rolling window of the last
`lookback` bar closes. Compute mean (`mu`) and stdev (`sigma`) every
bar. The 2-sigma bands are `mu +/- sigma_mult * sigma`.

Entries (one-shot per day):

  * Primed-short: latch when a bar's close > mu + sigma_mult * sigma.
    Fire SHORT on the first subsequent bar whose close comes BACK
    inside the band (close <= mu + sigma_mult * sigma).
  * Primed-long: symmetric on the lower band.

Why the latch-then-confirm pattern? Trading on the bar that pierces
the band is just shorting strength, which has bad expectancy on trend
days. Waiting for a return-into-band is a cheap "spike exhausted"
confirmation that materially improves mean-reversion expectancy in
the literature.

Exits:

  * Take profit: when price touches `mu` of the entry bar (a fixed
    target, NOT the moving mu -- otherwise the target chases the
    position and TP rarely fires).
  * Stop loss: `stop_ticks` against the entry.
  * Time stop: flatten after `time_stop_minutes` from entry.
  * EOD: flatten `flat_before_close_minutes` before session close.

Trading window: only operate in [trade_start_local, trade_end_local)
in US/Eastern. Excludes the open and close auctions where mean
reversion is famously poor.

No look-ahead: the entry decision uses only data from bars closing
at or before `bar.ts`. The engine still defers fills to the next
bar's open. The rolling stats use the closed `bar` as the most
recent observation (we already know its close).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Deque
from zoneinfo import ZoneInfo

from topstep50k.engine.types import Bar
from topstep50k.strategy.base import StrategyContext, TargetPosition


EASTERN = ZoneInfo("America/New_York")


@dataclass
class _MRDayState:
    session_open_utc: datetime
    session_close_utc: datetime
    trade_start_utc: datetime
    trade_end_utc: datetime
    flat_by_utc: datetime
    triggered: bool = False
    side: int = 0
    entry_price: float | None = None
    stop_price: float | None = None
    tp_price: float | None = None
    entry_ts_utc: datetime | None = None
    primed_short: bool = False
    primed_long: bool = False


@dataclass
class MeanReversionBollinger:
    """Per-symbol Bollinger-band fade.

    Parameters
    ----------
    symbol : bar feed symbol.
    qty : contracts per trade (signed by direction). Default 1.
    lookback : number of recent bar closes for mu/sigma. Default 60
        (= 60 minutes on 1-minute bars; a 1-hour rolling window).
    sigma_mult : band width in stdev units. Default 2.0.
    stop_ticks : protective stop distance from entry. Default 15.
    time_stop_minutes : max time in trade. Default 45.
    trade_start_local : earliest entry time, US/Eastern. Default
        10:00 (skips opening 30 minutes which is full of fakeouts).
    trade_end_local : latest entry time. Default 15:00 (skip last
        hour for end-of-day flow).
    flat_before_close_minutes : force-flat lead time before RTH close.
    tick_size : symbol tick size; default 0.25 (ES/NQ/MES/MNQ).
    """

    symbol: str
    qty: int = 1
    lookback: int = 60
    sigma_mult: float = 2.0
    stop_ticks: int = 15
    time_stop_minutes: int = 45
    trade_start_local: time = time(10, 0)
    trade_end_local: time = time(15, 0)
    session_open_local: time = time(9, 30)
    session_close_local: time = time(16, 0)
    flat_before_close_minutes: int = 15
    tick_size: float = 0.25
    _closes: Deque[float] = field(init=False, repr=False,
                                   default_factory=lambda: deque(maxlen=60))
    _day_state: _MRDayState | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        # deque maxlen needs to match self.lookback (not the default 60)
        self._closes = deque(maxlen=self.lookback)

    def _need_new_day(self, bar_ts: datetime) -> bool:
        if self._day_state is None:
            return True
        cur = bar_ts.astimezone(EASTERN).date()
        prev = self._day_state.session_open_utc.astimezone(EASTERN).date()
        return cur != prev

    def _build_day_state(self, bar_ts: datetime) -> _MRDayState:
        eastern = bar_ts.astimezone(EASTERN)
        d = eastern.date()
        so = datetime.combine(d, self.session_open_local, tzinfo=EASTERN)
        sc = datetime.combine(d, self.session_close_local, tzinfo=EASTERN)
        ts_local = datetime.combine(d, self.trade_start_local, tzinfo=EASTERN)
        te_local = datetime.combine(d, self.trade_end_local, tzinfo=EASTERN)
        fb_local = sc - timedelta(minutes=self.flat_before_close_minutes)
        tz = bar_ts.tzinfo
        return _MRDayState(
            session_open_utc=so.astimezone(tz),
            session_close_utc=sc.astimezone(tz),
            trade_start_utc=ts_local.astimezone(tz),
            trade_end_utc=te_local.astimezone(tz),
            flat_by_utc=fb_local.astimezone(tz),
        )

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        if self._need_new_day(bar.ts):
            # Carry rolling-close history across sessions; it's an
            # intraday signal but the band stays sensible at the
            # session boundary using the trailing window.
            self._day_state = self._build_day_state(bar.ts)
        st = self._day_state

        # Outside RTH: stay flat, ignore data (overnight bars would
        # pollute the intraday band).
        if bar.ts < st.session_open_utc or bar.ts >= st.session_close_utc:
            if ctx.position(self.symbol) != 0:
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="mr_session_end")
            return None

        # Update rolling-close history with this bar's close.
        self._closes.append(bar.close)

        cur_qty = ctx.position(self.symbol)

        # ----- exit management for an open position -----------------
        if cur_qty != 0 and st.side != 0 and st.entry_price is not None:
            # Time stop
            if (st.entry_ts_utc is not None
                    and bar.ts - st.entry_ts_utc >=
                    timedelta(minutes=self.time_stop_minutes)):
                st.triggered = True
                return TargetPosition(symbol=self.symbol, qty=0, tag="mr_time_exit")
            # EOD flat
            if bar.ts >= st.flat_by_utc:
                st.triggered = True
                return TargetPosition(symbol=self.symbol, qty=0, tag="mr_eod_exit")
            # TP / stop
            if self._hit_stop(bar, st) or self._hit_tp(bar, st):
                return TargetPosition(symbol=self.symbol, qty=0, tag="mr_exit")
            return None

        # ----- already entered today and currently flat: stand down -
        if st.triggered:
            return None

        # Outside the entry window? No new entries.
        if bar.ts < st.trade_start_utc or bar.ts >= st.trade_end_utc:
            return None

        # Need enough history for the band.
        if len(self._closes) < self.lookback:
            return None

        mu = sum(self._closes) / len(self._closes)
        var = sum((c - mu) ** 2 for c in self._closes) / (len(self._closes) - 1)
        sigma = var ** 0.5
        upper = mu + self.sigma_mult * sigma
        lower = mu - self.sigma_mult * sigma

        # Prime / fire short
        if bar.close > upper:
            st.primed_short = True
        elif st.primed_short and bar.close <= upper:
            return self._enter(bar, st, side=-1, mu=mu)

        if bar.close < lower:
            st.primed_long = True
        elif st.primed_long and bar.close >= lower:
            return self._enter(bar, st, side=+1, mu=mu)

        return None

    def _enter(self, bar: Bar, st: _MRDayState, *, side: int,
                mu: float) -> TargetPosition:
        st.triggered = True
        st.side = side
        st.entry_price = bar.close
        st.entry_ts_utc = bar.ts
        if side > 0:
            st.stop_price = st.entry_price - self.stop_ticks * self.tick_size
        else:
            st.stop_price = st.entry_price + self.stop_ticks * self.tick_size
        st.tp_price = mu
        return TargetPosition(
            symbol=self.symbol,
            qty=side * self.qty,
            tag=f"mr_{'long' if side > 0 else 'short'}",
        )

    def _hit_stop(self, bar: Bar, st: _MRDayState) -> bool:
        if st.stop_price is None:
            return False
        if st.side > 0:
            return bar.low <= st.stop_price
        return bar.high >= st.stop_price

    def _hit_tp(self, bar: Bar, st: _MRDayState) -> bool:
        if st.tp_price is None:
            return False
        if st.side > 0:
            return bar.high >= st.tp_price
        return bar.low <= st.tp_price
