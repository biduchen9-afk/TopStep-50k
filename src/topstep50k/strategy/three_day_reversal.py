"""3-Day Reversal — mean reversion after 3 consecutive directional closes.

Academic rationale
──────────────────
Lehmann (1990, J. Finance): "Fads, Martingales, and Market Efficiency"
  Weekly return reversals in individual stocks; losers outperform winners.
Lo & MacKinlay (1990, Rev. Fin. Studies): similar portfolio reversal effects.
Jegadeesh (1990, J. Finance): contrarian profits in individual stocks.
Extended to equity-index futures in academic literature: after N consecutive
  same-direction daily closes, the next session tends to reverse (mean reversion
  in daily returns). The effect is a micro-structural liquidity effect: forced
  sellers/buyers after sustained directional moves.

Pre-committed parameters (no data tuning)
─────────────────────────────────────────
  lookback        : 3   — 3 consecutive closes in same direction to trigger
  entry_time      : entry at next RTH open bar (09:30 ET)
  exit_time       : exit at 15:45 ET same day (intraday only)
  stop_multiple   : 1.5 × ATR(5-day) of prior closes
  atr_lookback    : 5   — simple range proxy from last 5 bars of prior closes

Direction
─────────
  3 consecutive DOWN closes → LONG (expect reversal upward)
  3 consecutive UP closes   → SHORT (expect reversal downward)

Regime placement: MEAN-REVERTING sessions. The pattern is weaker on trend days
with strong directional flows. Best on days with no major macro catalyst.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from topstep50k.engine.types import Bar
from topstep50k.strategy.base import StrategyContext, TargetPosition


EASTERN = ZoneInfo("America/New_York")

_RTH_OPEN  = time(9, 30)
_RTH_CLOSE = time(16, 0)
_EXIT_TIME = time(15, 45)


@dataclass
class _TDRState:
    last_seen_date: date | None = None
    # Rolling close history: (date, close)
    daily_closes: deque = field(default_factory=lambda: deque(maxlen=10))
    open_bar_seen: bool = False
    done_today: bool = False
    direction: int = 0
    stop_price: float | None = None
    target_qty: int = 0
    last_rth_bar_close: float | None = None


@dataclass
class ThreeDayReversal:
    """Short after 3 consecutive up closes; long after 3 consecutive down closes.

    Parameters
    ----------
    symbol          : bar-feed symbol.
    qty             : contracts per trade.
    lookback        : consecutive same-direction closes required (default 3).
    stop_atr_mult   : stop = lookback_atr × multiple.
    atr_lookback    : bars to estimate ATR (using prior close range proxy).
    exit_time_local : flatten at or after this local time.
    tick_size       : minimum price increment.
    daily_filter    : optional gate; skip day if returns False.
    """

    symbol: str
    qty: int = 1
    lookback: int = 3
    stop_atr_mult: float = 1.5
    atr_lookback: int = 5
    exit_time_local: time = _EXIT_TIME
    session_open_local: time = _RTH_OPEN
    session_close_local: time = _RTH_CLOSE
    tick_size: float = 0.25
    daily_filter: Callable[[date], bool] | None = None

    _state: _TDRState = field(init=False, default_factory=_TDRState, repr=False)

    def _round_to_tick(self, price: float) -> float:
        ts = self.tick_size
        return round(round(price / ts) * ts, 10)

    def _atr_proxy(self) -> float:
        closes = [c for _, c in self._state.daily_closes]
        if len(closes) < 2:
            return 0.0
        ranges = [abs(closes[i] - closes[i-1]) for i in range(1, len(closes))]
        return sum(ranges[-self.atr_lookback:]) / len(ranges[-self.atr_lookback:])

    def _maybe_roll_day(self, bar: Bar) -> None:
        et = bar.ts.astimezone(EASTERN)
        d = et.date()
        s = self._state
        if s.last_seen_date == d:
            return
        # On day roll: record yesterday's last RTH close
        if s.last_rth_bar_close is not None and s.last_seen_date is not None:
            s.daily_closes.append((s.last_seen_date, s.last_rth_bar_close))
        s.last_seen_date = d
        s.last_rth_bar_close = None
        s.open_bar_seen = False
        s.done_today = False
        s.direction = 0
        s.stop_price = None
        s.target_qty = 0

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        self._maybe_roll_day(bar)
        s = self._state
        et = bar.ts.astimezone(EASTERN)

        # Track last RTH bar close (for rolling close history)
        if et.time() < self.session_close_local:
            s.last_rth_bar_close = bar.close

        # ── Force flat before close ────────────────────────────────────────
        if et.time() >= self.exit_time_local:
            if not s.done_today:
                s.done_today = True
                if ctx.position(self.symbol) != 0 or s.target_qty != 0:
                    s.target_qty = 0
                    return TargetPosition(symbol=self.symbol, qty=0,
                                          tag="tdr_flat_close")
            return None

        if s.done_today:
            return None

        cur_qty = ctx.position(self.symbol)

        # ── Manage open position ───────────────────────────────────────────
        if cur_qty != 0 and s.direction != 0:
            if s.direction > 0 and bar.low <= s.stop_price:
                s.target_qty = 0
                s.done_today = True
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="tdr_stop")
            if s.direction < 0 and bar.high >= s.stop_price:
                s.target_qty = 0
                s.done_today = True
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="tdr_stop")
            return None

        # ── Entry: first bar at/after RTH open ────────────────────────────
        if not s.open_bar_seen and et.time() >= self.session_open_local:
            s.open_bar_seen = True

            closes = list(s.daily_closes)
            if len(closes) < self.lookback:
                s.done_today = True
                return None

            # Check last `lookback` closes for consecutive direction
            recent = [c for _, c in closes[-self.lookback:]]
            up_count   = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
            down_count = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])

            if up_count == self.lookback - 1 + 1:
                # Need all lookback moves to be up (lookback closes = lookback-1 pairs)
                pass
            # Re-check properly: need lookback consecutive same-direction closes
            # "3 consecutive closes" means 3 closes each > prior (2 up-moves needed for 3 closes)
            # We use lookback=3 → need closes[0] < closes[1] < closes[2]
            all_up   = all(recent[i] > recent[i-1] for i in range(1, len(recent)))
            all_down = all(recent[i] < recent[i-1] for i in range(1, len(recent)))

            if not all_up and not all_down:
                s.done_today = True
                return None

            # Optional daily gate
            if self.daily_filter is not None:
                if not self.daily_filter(et.date()):
                    s.done_today = True
                    return None

            atr = self._atr_proxy()
            stop_dist = self._round_to_tick(atr * self.stop_atr_mult) if atr > 0 else 0.0
            if stop_dist <= 0:
                s.done_today = True
                return None

            if all_up:
                s.direction = -1
                s.stop_price = self._round_to_tick(bar.open + stop_dist)
            else:
                s.direction = +1
                s.stop_price = self._round_to_tick(bar.open - stop_dist)

            s.target_qty = s.direction * self.qty
            return TargetPosition(symbol=self.symbol,
                                   qty=s.direction * self.qty,
                                   tag="tdr_enter")

        return None
