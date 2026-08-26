"""Opening Range Breakout for ES (and any RTH-traded symbol).

Definition
----------
For each US RTH session:

  1. Track the high (`or_high`) and low (`or_low`) of all bars whose
     timestamp falls in [session_open, session_open + or_minutes).
  2. After `session_open + or_minutes`, monitor bars for a breakout:
       * Close > or_high  -> go long the breakout
       * Close < or_low   -> go short the breakout
       (one-shot per day; the strategy never enters again same day.)
  3. Once a position is open, manage it with:
       * Stop loss at the opposite side of the OR (default), OR a
         fixed-tick stop.
       * Take profit at `tp_multiple * OR_width` from entry.
       * Time stop: flatten `flat_before_close_minutes` before
         session_close, regardless of P&L.

Bars consumed must already be `clock.now()`-visible -- the engine
takes care of that via deferred fills (orders fill at the NEXT bar's
open, never the same bar).

Audit
-----
Every meaningful state change (OR formed, breakout signal, stop/TP
attached, time exit) is logged via the audit hooks the engine emits
naturally for fills and orders; the strategy itself does not write
to the audit log directly (it has no clock/audit handles).

Look-ahead
----------
The OR is built only from bars whose ts is < session_open + or_minutes.
The first breakout DECISION can fire as soon as a bar with
ts >= session_open + or_minutes closes ABOVE or_high (or below or_low).
Because the engine defers all fills to the next bar's open, the actual
entry price is the open of the bar AFTER the breakout signal -- which
is what a live trader would get.

Sessions
--------
ES RTH session: 09:30 - 16:00 US/Eastern.
The bar data is in UTC. EDT = UTC-4, EST = UTC-5. To handle DST,
session windows are computed in US/Eastern then converted to UTC
per-day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Literal
from zoneinfo import ZoneInfo

from topstep50k.engine.types import Bar
from topstep50k.strategy.base import StrategyContext, TargetPosition


EASTERN = ZoneInfo("America/New_York")


StopMode = Literal["opposite_range", "fixed_ticks"]
Direction = Literal["long", "short", "both"]


@dataclass
class _DayState:
    session_open_utc: datetime
    session_close_utc: datetime
    or_end_utc: datetime
    flat_by_utc: datetime
    or_high: float | None = None
    or_low: float | None = None
    or_width: float | None = None
    armed: bool = False  # OR closed; ready to look for breakout
    triggered: bool = False  # already entered today
    side: int = 0  # +1 long, -1 short, 0 flat
    entry_price: float | None = None
    stop_price: float | None = None
    tp_price: float | None = None


@dataclass
class OpeningRangeBreakout:
    """Per-symbol ORB strategy. Construct one per symbol you trade.

    Parameters
    ----------
    symbol : the bar feed this instance reacts to. PortfolioStrategy
        will only route its symbol's bars here.
    qty : contracts per breakout (signed by direction). Default 1.
    or_minutes : opening-range window length. 15 / 30 / 60 are common.
    direction : 'long' | 'short' | 'both'.
    stop_mode : 'opposite_range' uses the opposite OR boundary as stop;
        'fixed_ticks' uses `stop_ticks` ticks from entry.
    stop_ticks : only used when stop_mode == 'fixed_ticks'.
    tp_multiple : profit target = entry +/- tp_multiple * OR_width.
        Use None for no take-profit (time stop only). Ignored when
        `tp_ticks` is set.
    tp_ticks : profit target = entry +/- tp_ticks * tick_size, independent
        of OR width. Takes priority over tp_multiple when set -- use this
        for a fixed risk:reward test (pair with stop_mode='fixed_ticks').
    flat_before_close_minutes : seconds before session close to force
        flat regardless of PnL.
    session_open_local : RTH open in US/Eastern, default 09:30.
    session_close_local : RTH close, default 16:00.
    tick_size : the symbol's tick size; needed for fixed-tick stops.
        Default 0.25 (ES, NQ, MES, MNQ; GC is 0.10).
    """

    symbol: str
    qty: int = 1
    or_minutes: int = 30
    direction: Direction = "both"
    stop_mode: StopMode = "opposite_range"
    stop_ticks: int = 40   # used only if stop_mode=='fixed_ticks'
    tp_multiple: float | None = 1.0
    tp_ticks: int | None = None  # fixed-tick TP; overrides tp_multiple when set
    flat_before_close_minutes: int = 15
    session_open_local: time = time(9, 30)
    session_close_local: time = time(16, 0)
    tick_size: float = 0.25
    # Optional chop filter: skip a day when its OR width is < this
    # fraction of the trailing-N median OR width. 0.0 disables.
    min_or_width_vs_median: float = 0.0
    or_width_history_days: int = 20
    # Optional per-day gate. Called once when a new trading day starts;
    # if it returns False the strategy stands flat for the whole day.
    # Argument is the trading date in US/Eastern.
    daily_filter: Callable[[date], bool] | None = None
    _or_widths: list[float] = field(init=False, default_factory=list, repr=False)
    _day_state: _DayState | None = field(init=False, default=None, repr=False)
    _gated_today: bool = field(init=False, default=False, repr=False)

    def _build_day_state(self, bar_ts: datetime) -> _DayState:
        """For the trading day bar_ts belongs to (in Eastern), compute
        session_open / session_close / or_end / flat_by, all in UTC.

        Supports sessions that cross midnight (session_close_local <=
        session_open_local, e.g. an Asia/Tokyo window 19:00-04:00 ET):
        the close is then anchored to the NEXT Eastern calendar day.
        """
        eastern = bar_ts.astimezone(EASTERN)
        session_date = eastern.date()
        overnight = self.session_close_local <= self.session_open_local
        session_open_local = datetime.combine(
            session_date, self.session_open_local, tzinfo=EASTERN,
        )
        close_date = session_date + timedelta(days=1) if overnight else session_date
        session_close_local = datetime.combine(
            close_date, self.session_close_local, tzinfo=EASTERN,
        )
        or_end_local = session_open_local + timedelta(minutes=self.or_minutes)
        flat_by_local = session_close_local - timedelta(
            minutes=self.flat_before_close_minutes
        )
        return _DayState(
            session_open_utc=session_open_local.astimezone(bar_ts.tzinfo),
            session_close_utc=session_close_local.astimezone(bar_ts.tzinfo),
            or_end_utc=or_end_local.astimezone(bar_ts.tzinfo),
            flat_by_utc=flat_by_local.astimezone(bar_ts.tzinfo),
        )

    def _need_new_day(self, bar_ts: datetime) -> bool:
        if self._day_state is None:
            return True
        # Still inside (or not yet past) the currently-tracked session?
        # No rebuild needed -- this is what lets an overnight-spanning
        # session (e.g. Asia 19:00-04:00 ET) survive the Eastern calendar
        # date rolling over in the middle of it, instead of being cut off
        # at midnight.
        if bar_ts < self._day_state.session_close_utc:
            return False
        # Past the tracked session's close: only rebuild once the Eastern
        # date has actually advanced past the session's own open date, so
        # we rebuild once per session cycle rather than on every bar
        # during the dead zone between sessions.
        cur_local_date = bar_ts.astimezone(EASTERN).date()
        prev_open_date = self._day_state.session_open_utc.astimezone(EASTERN).date()
        return cur_local_date != prev_open_date

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        """Return the desired target position after observing `bar`.

        Returns None when no change is needed. Returns a TargetPosition
        with qty=0 to flatten.
        """
        if self._need_new_day(bar.ts):
            self._day_state = self._build_day_state(bar.ts)
            # Daily gate evaluated once at the start of a new trading day.
            if self.daily_filter is not None:
                day_local = bar.ts.astimezone(EASTERN).date()
                self._gated_today = not self.daily_filter(day_local)
            else:
                self._gated_today = False
        if self._gated_today:
            return None

        st = self._day_state
        # Outside the regular session entirely? Stay flat.
        if bar.ts < st.session_open_utc or bar.ts >= st.session_close_utc:
            return self._maybe_close_for_eod(bar, st, ctx)

        # ----- OR formation window: track high/low ---------------------------
        if bar.ts < st.or_end_utc:
            st.or_high = bar.high if st.or_high is None else max(st.or_high, bar.high)
            st.or_low = bar.low if st.or_low is None else min(st.or_low, bar.low)
            return None

        # OR window closed: lock the boundaries the first time we see a
        # post-OR bar.
        if not st.armed:
            if st.or_high is None or st.or_low is None:
                # No bars landed in the OR window (data gap). Skip this day.
                st.triggered = True
                return None
            st.or_width = st.or_high - st.or_low
            st.armed = True
            # Chop filter: if today's OR is much narrower than recent
            # median, skip the day entirely (no entries). Causal --
            # uses only the past or_width_history_days closed days.
            if self.min_or_width_vs_median > 0 and len(self._or_widths) >= 5:
                hist = sorted(self._or_widths[-self.or_width_history_days:])
                median = hist[len(hist) // 2]
                if median > 0 and st.or_width < self.min_or_width_vs_median * median:
                    st.triggered = True  # mark as already entered -> no signal
            # Record this day's OR width for the rolling history
            self._or_widths.append(st.or_width)
            if len(self._or_widths) > self.or_width_history_days * 4:
                # bound memory growth
                self._or_widths = self._or_widths[-self.or_width_history_days * 2 :]

        # ----- post-OR: manage open position or look for breakout -----------
        # Time stop overrides everything.
        if bar.ts >= st.flat_by_utc and ctx.position(self.symbol) != 0:
            st.triggered = True
            return TargetPosition(symbol=self.symbol, qty=0, tag="orb_time_exit")

        # If we already have a position, check stop / tp
        cur_qty = ctx.position(self.symbol)
        if cur_qty != 0 and st.side != 0:
            if self._hit_stop(bar, st) or self._hit_tp(bar, st):
                return TargetPosition(symbol=self.symbol, qty=0, tag="orb_exit")
            return None

        # No position and we have not entered today -> look for breakout
        if st.triggered:
            return None
        signal = self._breakout_signal(bar, st)
        if signal == 0:
            return None
        st.triggered = True
        st.side = signal
        # Entry will actually fill at NEXT bar's open (engine deferral),
        # so we use `bar.close` only for stop/tp planning here -- it's a
        # reasonable approximation; the audit log preserves the real
        # entry fill price.
        st.entry_price = bar.close
        if self.stop_mode == "opposite_range":
            st.stop_price = st.or_low if signal > 0 else st.or_high
        else:
            st.stop_price = (
                st.entry_price - self.stop_ticks * self.tick_size
                if signal > 0
                else st.entry_price + self.stop_ticks * self.tick_size
            )
        if self.tp_ticks is not None:
            st.tp_price = (
                st.entry_price + self.tp_ticks * self.tick_size
                if signal > 0
                else st.entry_price - self.tp_ticks * self.tick_size
            )
        elif self.tp_multiple is not None and st.or_width:
            st.tp_price = (
                st.entry_price + self.tp_multiple * st.or_width
                if signal > 0
                else st.entry_price - self.tp_multiple * st.or_width
            )
        return TargetPosition(
            symbol=self.symbol,
            qty=signal * self.qty,
            tag=f"orb_{'long' if signal > 0 else 'short'}",
        )

    def _maybe_close_for_eod(self, bar, st, ctx) -> TargetPosition | None:
        # Outside the session window: if we still have a position from
        # the previous day's session for some reason, flatten.
        if ctx.position(self.symbol) != 0:
            return TargetPosition(symbol=self.symbol, qty=0, tag="orb_session_end")
        return None

    def _hit_stop(self, bar, st) -> bool:
        if st.stop_price is None:
            return False
        if st.side > 0:
            return bar.low <= st.stop_price
        return bar.high >= st.stop_price

    def _hit_tp(self, bar, st) -> bool:
        if st.tp_price is None:
            return False
        if st.side > 0:
            return bar.high >= st.tp_price
        return bar.low <= st.tp_price

    def _breakout_signal(self, bar, st) -> int:
        if self.direction != "short" and st.or_high is not None and bar.close > st.or_high:
            return +1
        if self.direction != "long" and st.or_low is not None and bar.close < st.or_low:
            return -1
        return 0
