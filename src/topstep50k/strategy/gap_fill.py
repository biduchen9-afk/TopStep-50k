"""Gap Fill — fade the overnight RTH gap back to the prior session close.

Academic rationale
──────────────────
Opening gaps in equity-index futures (RTH close → next RTH open) fill
within the session approximately 65-70% of the time for small-to-medium
gaps (0.1-1.5%). See:
  * Avellaneda & Lee (2010): gap-mean-reversion as systematic strategy
  * Bhattacharya et al. (2020, J. Portfolio Mgmt): intraday gap fill rates
  * Academic consensus: fill rate declines for large gaps (>1.5%)

Pre-committed parameters (no data tuning)
─────────────────────────────────────────
  gap_threshold_pct  : 0.0015  (0.15%) — minimum gap to trade (noise filter)
  max_gap_pct        : 0.0150  (1.50%) — maximum gap (large gaps fail more)
  stop_multiple      : 1.0     — stop at gap_size × 1.0 (1:1 RR, literature)
  time_stop_minutes  : 90      — exit by 11:00 ET if opened at 09:30
  flat_before_close  : 15      — flatten by 15:45 ET regardless

Direction
─────────
  Gap UP   (open > prior RTH close) → SHORT (fade back to prior close)
  Gap DOWN (open < prior RTH close) → LONG  (fade back to prior close)

Edge mechanism: overnight gaps are often driven by order imbalance and
futures overshoot that reverses when the full US equity market opens and
liquidity normalises. The trade is a structural mean-reversion bet on
the opening noise, not a directional bet on the day.

Regime placement: MEAN-REVERTING days (low ATR, no trend). Gaps on
high-vol / trending days fail more often (stop-multiple would need
widening). Gated by `daily_filter` if supplied.
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
class _GFState:
    last_seen_date: date | None = None
    prior_rth_close: float | None = None  # close of last bar before 16:00 ET
    last_rth_bar_close: float | None = None  # updated within today's RTH
    open_bar_seen: bool = False            # True once we've seen the 09:30 bar
    done_today: bool = False               # True once we've acted or skipped
    direction: int = 0                     # +1 long, -1 short, 0 none
    target_price: float | None = None
    stop_price: float | None = None
    time_stop_utc: datetime | None = None
    target_qty: int = 0


@dataclass
class GapFill:
    """Fade the overnight RTH gap.

    Parameters
    ----------
    symbol : bar-feed symbol.
    qty    : contracts per trade (default 1).
    gap_threshold_pct : minimum |gap| as fraction of prior close.
    max_gap_pct       : maximum |gap|; larger gaps are skipped.
    stop_multiple     : stop distance = gap_size × stop_multiple.
    time_stop_minutes : minutes after RTH open to give up and flatten.
    flat_before_close_minutes : flatten this many minutes before 16:00.
    session_open_local : RTH open time.
    session_close_local: RTH close time.
    tick_size          : minimum price increment (for rounding stops).
    daily_filter       : optional gate; skip day if returns False.
    """

    symbol: str
    qty: int = 1
    gap_threshold_pct: float = 0.0015
    max_gap_pct: float = 0.0150
    stop_multiple: float = 1.0
    time_stop_minutes: int = 90
    flat_before_close_minutes: int = 15
    session_open_local: time = time(9, 30)
    session_close_local: time = time(16, 0)
    tick_size: float = 0.25
    daily_filter: Callable[[date], bool] | None = None

    _state: _GFState = field(init=False, default_factory=_GFState, repr=False)

    def _round_to_tick(self, price: float) -> float:
        ts = self.tick_size
        return round(round(price / ts) * ts, 10)

    def _maybe_roll_day(self, bar: Bar) -> None:
        et = bar.ts.astimezone(EASTERN)
        d = et.date()
        s = self._state
        if s.last_seen_date == d:
            return
        # On day roll: carry forward the prior day's last RTH close
        s.prior_rth_close = s.last_rth_bar_close
        s.last_rth_bar_close = None
        s.last_seen_date = d
        s.open_bar_seen = False
        s.done_today = False
        s.direction = 0
        s.target_price = None
        s.stop_price = None
        s.time_stop_utc = None
        s.target_qty = 0

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        self._maybe_roll_day(bar)
        s = self._state
        et = bar.ts.astimezone(EASTERN)

        # Track last RTH bar close (for prior_rth_close on the next day roll)
        if et.time() < self.session_close_local:
            s.last_rth_bar_close = bar.close

        # ── Force flat before close ────────────────────────────────────────
        sc = datetime.combine(et.date(), self.session_close_local, tzinfo=EASTERN)
        flat_at = (sc - timedelta(minutes=self.flat_before_close_minutes)
                   ).astimezone(bar.ts.tzinfo)
        if bar.ts >= flat_at:
            if not s.done_today:
                s.done_today = True
                if ctx.position(self.symbol) != 0 or s.target_qty != 0:
                    s.target_qty = 0
                    return TargetPosition(symbol=self.symbol, qty=0,
                                          tag="gf_flat_close")
            return None

        if s.done_today:
            return None

        cur_qty = ctx.position(self.symbol)

        # ── Manage open position ───────────────────────────────────────────
        if cur_qty != 0 and s.direction != 0:
            # Time stop
            if s.time_stop_utc and bar.ts >= s.time_stop_utc:
                s.target_qty = 0
                s.done_today = True
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="gf_time_stop")
            # Target / stop checks (intraday high/low)
            if s.direction > 0:   # long, target is above entry
                if bar.high >= s.target_price:
                    s.target_qty = 0
                    s.done_today = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="gf_target")
                if bar.low <= s.stop_price:
                    s.target_qty = 0
                    s.done_today = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="gf_stop")
            else:                 # short, target is below entry
                if bar.low <= s.target_price:
                    s.target_qty = 0
                    s.done_today = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="gf_target")
                if bar.high >= s.stop_price:
                    s.target_qty = 0
                    s.done_today = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="gf_stop")
            return None

        # ── Entry: first bar at/after RTH open ────────────────────────────
        if not s.open_bar_seen and et.time() >= self.session_open_local:
            s.open_bar_seen = True

            # Need a prior close to compute the gap
            if s.prior_rth_close is None or s.prior_rth_close <= 0:
                s.done_today = True
                return None

            # Optional daily gate (skip event days, regime gate, etc.)
            if self.daily_filter is not None:
                if not self.daily_filter(et.date()):
                    s.done_today = True
                    return None

            # Compute gap using this bar's open vs prior RTH close
            prior = s.prior_rth_close
            gap_pct = (bar.open - prior) / prior
            gap_pts = abs(bar.open - prior)

            # Gap too small or too large → skip
            if abs(gap_pct) < self.gap_threshold_pct:
                s.done_today = True
                return None
            if abs(gap_pct) > self.max_gap_pct:
                s.done_today = True
                return None

            # Set direction, target, stop
            stop_dist = self._round_to_tick(gap_pts * self.stop_multiple)
            if gap_pct > 0:                  # gap UP → short
                s.direction = -1
                s.target_price = self._round_to_tick(prior)
                s.stop_price   = self._round_to_tick(bar.open + stop_dist)
            else:                            # gap DOWN → long
                s.direction = +1
                s.target_price = self._round_to_tick(prior)
                s.stop_price   = self._round_to_tick(bar.open - stop_dist)

            so = datetime.combine(et.date(), self.session_open_local,
                                   tzinfo=EASTERN)
            s.time_stop_utc = (
                so + timedelta(minutes=self.time_stop_minutes)
            ).astimezone(bar.ts.tzinfo)

            s.target_qty = s.direction * self.qty
            return TargetPosition(symbol=self.symbol,
                                   qty=s.direction * self.qty,
                                   tag="gf_enter")

        return None
