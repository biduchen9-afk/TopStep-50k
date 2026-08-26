"""Power Hour Continuation — the day's already-realized move predicts
the final hour, entering ONLY for that final hour.

Rationale
─────────
Distinct from IntradayMomentum (Gao/Han/Li/Zhou 2018: first-30-min
return predicts the LAST 30 min, entered right after the open and held
most of the day). This strategy is deliberately time-disjoint from
every other stream in the ensemble: it takes NO position until
`power_hour_minutes` before the session close, using the day's full
open-to-that-point cumulative return as the signal, and holds only for
that final window. Two documented, related mechanisms motivate it:

  * End-of-day order-flow persistence / late-session imbalance:
    trading activity concentrates in the first and last portions of
    the RTH session, and late-session order imbalances have
    predictive power for the remaining minutes of that session (see
    market-microstructure literature on closing-period imbalances,
    e.g. work building on Cushing & Madhavan-style intraday volume/
    imbalance studies).
  * CAVEAT, stated plainly: the specific "closing auction imbalance"
    mechanism (NASDAQ-style published auction imbalance data driving
    price into a formal closing auction) does NOT literally apply to
    CME futures (ES/NQ/GC) -- futures have no closing auction, trading
    is continuous and the "close" is just the last traded price. This
    strategy borrows the more general, less mechanism-specific
    intuition (late-session order flow / momentum persistence into
    the close) rather than the auction-imbalance mechanism itself.
    Flagged here explicitly so this isn't overstated as directly-
    transplanted equity microstructure literature -- it's an
    empirically-testable analogy, not a proven-transferable effect.

Pre-committed parameters (no data tuning)
─────────────────────────────────────────
  power_hour_minutes : 60     — signal measured, and position held,
                                 only in the final 60 min of RTH
  min_signal_pct      : 0.0010 — same noise floor as IntradayMomentum
  stop_multiple        : 2.0   — stop at 2x the day's move-so-far
  flat_before_close_minutes : 5 — flatten 5 min before session close

Direction
─────────
  Open-to-(close - power_hour_minutes) return > +min_signal_pct -> LONG
  Open-to-(close - power_hour_minutes) return < -min_signal_pct -> SHORT

Regime placement: designed to be structurally decorrelated from
ORB (fires at open) and MeanRev (mid-session fades) by TIME WINDOW,
not by a data-fit condition -- the same kind of decorrelation OD used
to provide via the overnight window, but entirely within the legal
RTH+pre-flatten trading day.
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
class _PHState:
    last_seen_date: date | None = None
    session_open_price: float | None = None   # price at the RTH open bar
    signal_computed: bool = False
    entry_sent: bool = False
    done_today: bool = False
    direction: int = 0
    stop_price: float | None = None
    target_qty: int = 0


@dataclass
class PowerHourContinuation:
    """Continue the day's realized direction for the final trading hour.

    Parameters
    ----------
    symbol : bar-feed symbol.
    qty    : contracts per trade.
    power_hour_minutes : minutes before session close where the signal
        is measured AND the position is held (default 60).
    min_signal_pct : minimum |open-to-signal-time return| to trade.
    stop_multiple : stop distance = |signal move| x multiple.
    flat_before_close_minutes : minutes before session close to force flat.
    session_open_local, session_close_local : RTH boundaries.
    tick_size : minimum price increment.
    daily_filter : optional gate; skip day if returns False.
    """

    symbol: str
    qty: int = 1
    power_hour_minutes: int = 60
    min_signal_pct: float = 0.0010
    stop_multiple: float = 2.0
    flat_before_close_minutes: int = 5
    session_open_local: time = time(9, 30)
    session_close_local: time = time(16, 0)
    tick_size: float = 0.25
    daily_filter: Callable[[date], bool] | None = None

    _state: _PHState = field(init=False, default_factory=_PHState, repr=False)

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
        s.session_open_price = None
        s.signal_computed = False
        s.entry_sent = False
        s.done_today = False
        s.direction = 0
        s.stop_price = None

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        self._maybe_roll_day(bar)
        s = self._state
        et = bar.ts.astimezone(EASTERN)

        session_open_dt = datetime.combine(et.date(), self.session_open_local,
                                            tzinfo=EASTERN)
        session_close_dt = datetime.combine(et.date(), self.session_close_local,
                                             tzinfo=EASTERN)
        signal_time_dt = session_close_dt - timedelta(minutes=self.power_hour_minutes)
        flat_by_dt = session_close_dt - timedelta(minutes=self.flat_before_close_minutes)

        if et < session_open_dt.astimezone(et.tzinfo) or et >= session_close_dt.astimezone(et.tzinfo):
            if ctx.position(self.symbol) != 0:
                return TargetPosition(symbol=self.symbol, qty=0, tag="ph_session_end")
            return None

        cur_qty = ctx.position(self.symbol)

        # ── Exits: time stop (flatten before close) / price stop ──────────
        if s.entry_sent and cur_qty != 0 and s.direction != 0:
            if bar.ts >= flat_by_dt.astimezone(bar.ts.tzinfo):
                s.direction = 0
                return TargetPosition(symbol=self.symbol, qty=0, tag="ph_time_exit")
            if self._hit_stop(bar, s):
                s.direction = 0
                return TargetPosition(symbol=self.symbol, qty=0, tag="ph_stop")
            return None

        # ── Capture the RTH open price (first bar of the session) ─────────
        if s.session_open_price is None:
            s.session_open_price = bar.open
            return None

        # ── Compute signal + enter, exactly at the power-hour boundary ────
        if (not s.signal_computed and not s.done_today
                and bar.ts >= signal_time_dt.astimezone(bar.ts.tzinfo)):
            s.signal_computed = True

            if self.daily_filter is not None:
                if not self.daily_filter(et.date()):
                    s.done_today = True
                    return None

            ret = (bar.close - s.session_open_price) / s.session_open_price
            move = abs(bar.close - s.session_open_price)

            if abs(ret) < self.min_signal_pct:
                s.done_today = True
                return None

            stop_dist = self._round_to_tick(move * self.stop_multiple)
            if ret > 0:
                s.direction = +1
                s.stop_price = self._round_to_tick(bar.close - stop_dist)
            else:
                s.direction = -1
                s.stop_price = self._round_to_tick(bar.close + stop_dist)

            s.entry_sent = True
            s.target_qty = s.direction * self.qty
            return TargetPosition(symbol=self.symbol, qty=s.direction * self.qty,
                                   tag="ph_enter")

        return None

    def _hit_stop(self, bar: Bar, s: _PHState) -> bool:
        if s.stop_price is None:
            return False
        if s.direction > 0:
            return bar.low <= s.stop_price
        return bar.high >= s.stop_price
