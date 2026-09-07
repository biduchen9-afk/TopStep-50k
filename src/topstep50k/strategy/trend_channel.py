"""Trend Channel Breakout — Donchian N-day high/low breakout.

Academic rationale
──────────────────
Donchian (1960): Original N-day channel breakout system. Buy when price
  exceeds the highest high of the past N days, short when it breaks the
  lowest low. Documented to capture medium-term trends in commodities.

Covel (2004) "Trend Following": Comprehensive review of CTA trend systems.
  Donchian channel (20-day) is the benchmark for all trend-following CTAs.

Hurst, Ooi & Pedersen (2013, J. Portfolio Mgmt): "A Century of Evidence on
  Trend-Following Investing" — trend following is robust across asset classes
  and historical periods. Time-series momentum (1-12 month) is statistically
  significant (t-stat ~4.9 in their broad cross-asset sample).

Moskowitz, Ooi & Pedersen (2012, J. Finance): "Time Series Momentum"
  Same-direction autocorrelation in returns at 1-12 month horizon.
  Sharpe of diversified TSMOM portfolio: ~0.8 annualized.

Application to futures: a short-lookback (10-day) version captures faster
  trend signals in liquid futures markets (ES, NQ, GC). The shorter lookback
  increases trade frequency vs the classic 20-day channel.

Pre-committed parameters (no data tuning)
─────────────────────────────────────────
  lookback_days    : 10    (breakout of prior 10-day high/low)
  atr_lookback     : 14    (ATR estimate: standard in trend-following)
  stop_atr_mult    : 2.0   (stop 2× ATR from entry, Hurst et al.)
  exit_lookback    : 5     (exit if close reverses below 5-day opposite extreme)
  max_hold_days    : 15    (cap hold period, prevent over-exposure)
  entry_time       : 09:35 ET (entry at open after daily close signal)
  exit_time        : 15:45 ET intraday (if stopped intraday)

Direction
─────────
  Close above 10-day highest high → LONG next session
  Close below 10-day lowest low   → SHORT next session
  Exit: (a) intraday stop at 2×ATR, (b) close beyond 5-day opposite extreme,
        (c) max_hold_days, (d) 15:45 flatten

Signal uses prior-day CLOSE (no intraday data needed for signal computation).
No look-ahead: signal is computed after yesterday's close is final.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from topstep50k.engine.types import Bar
from topstep50k.strategy.base import StrategyContext, TargetPosition


EASTERN = ZoneInfo("America/New_York")

_RTH_OPEN  = time(9, 30)
_RTH_CLOSE = time(16, 0)
_ENTRY_TIME = time(9, 35)
_EXIT_TIME  = time(15, 45)


@dataclass
class _TCState:
    last_seen_date: date | None = None
    last_rth_bar_close: float | None = None
    # rolling daily close history (date, close)
    daily_closes: deque = field(default_factory=lambda: deque(maxlen=25))
    # current position state
    in_trade: bool = False
    direction: int = 0           # +1 long, -1 short
    entry_date: date | None = None
    hold_days: int = 0
    stop_price: float | None = None
    done_today: bool = False
    entry_sent: bool = False
    target_qty: int = 0
    # pending signal (computed from yesterday's close)
    pending_direction: int = 0


@dataclass
class TrendChannel:
    """Donchian N-day channel breakout system for futures.

    Parameters
    ----------
    symbol         : bar-feed symbol.
    qty            : contracts per trade.
    lookback_days  : breakout lookback (default 10).
    atr_lookback   : bars for ATR estimate (default 14).
    stop_atr_mult  : stop = entry ± (ATR × multiple).
    exit_lookback  : exit when close reverses below this many days' opposite extreme.
    max_hold_days  : flatten after this many calendar days.
    tick_size      : minimum price increment.
    """

    symbol: str
    qty: int = 1
    lookback_days: int = 10
    atr_lookback: int = 14
    stop_atr_mult: float = 2.0
    exit_lookback: int = 5
    max_hold_days: int = 15
    tick_size: float = 0.25

    _state: _TCState = field(init=False, default_factory=_TCState, repr=False)

    def _round_to_tick(self, price: float) -> float:
        ts = self.tick_size
        return round(round(price / ts) * ts, 10)

    def _atr(self) -> float:
        closes = [c for _, c in self._state.daily_closes]
        if len(closes) < 2:
            return 0.0
        ranges = [abs(closes[i] - closes[i-1]) for i in range(1, len(closes))]
        recent = ranges[-self.atr_lookback:]
        return sum(recent) / len(recent) if recent else 0.0

    def _check_signal(self) -> int:
        """Return +1 (long), -1 (short), or 0 (no signal) based on daily closes."""
        s = self._state
        closes_raw = list(s.daily_closes)
        if len(closes_raw) < self.lookback_days + 1:
            return 0
        closes = [c for _, c in closes_raw]
        last_close = closes[-1]
        prior_n = closes[-self.lookback_days - 1 : -1]  # N days before last
        if not prior_n:
            return 0
        highest_n = max(prior_n)
        lowest_n  = min(prior_n)
        if last_close > highest_n:
            return +1
        if last_close < lowest_n:
            return -1
        return 0

    def _should_exit(self) -> bool:
        """Check if close-based exit trigger hit."""
        s = self._state
        if not s.in_trade:
            return False
        closes_raw = list(s.daily_closes)
        if len(closes_raw) < self.exit_lookback + 1:
            return False
        closes = [c for _, c in closes_raw]
        last_close = closes[-1]
        prior_exit = closes[-self.exit_lookback - 1 : -1]
        if not prior_exit:
            return False
        if s.direction > 0 and last_close < min(prior_exit):
            return True
        if s.direction < 0 and last_close > max(prior_exit):
            return True
        return False

    def _maybe_roll_day(self, bar: Bar) -> None:
        et = bar.ts.astimezone(EASTERN)
        d = et.date()
        s = self._state
        if s.last_seen_date == d:
            return
        # Day roll: record yesterday's final RTH close and compute signal
        if s.last_rth_bar_close is not None and s.last_seen_date is not None:
            s.daily_closes.append((s.last_seen_date, s.last_rth_bar_close))
            # Check close-based exit for existing position
            if s.in_trade and self._should_exit():
                s.pending_direction = -999  # flag: close-based exit pending
            elif not s.in_trade:
                s.pending_direction = self._check_signal()
        s.last_seen_date = d
        s.last_rth_bar_close = None
        s.done_today = False
        s.entry_sent = False
        if s.in_trade:
            s.hold_days += 1

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        self._maybe_roll_day(bar)
        s = self._state
        et = bar.ts.astimezone(EASTERN)

        # Track last RTH bar close
        if et.time() < _RTH_CLOSE:
            s.last_rth_bar_close = bar.close

        # ── Force flat before close ────────────────────────────────────────
        if et.time() >= _EXIT_TIME:
            if not s.done_today:
                s.done_today = True
                cur = ctx.position(self.symbol)
                if cur != 0 or s.target_qty != 0:
                    s.in_trade = False
                    s.direction = 0
                    s.hold_days = 0
                    s.target_qty = 0
                    return TargetPosition(symbol=self.symbol, qty=0,
                                          tag="tc_flat_close")
            return None

        if s.done_today:
            return None

        cur_qty = ctx.position(self.symbol)

        # ── Manage open position: intraday stop + max hold ─────────────────
        if cur_qty != 0 and s.direction != 0:
            # Intraday stop check
            if s.stop_price is not None:
                if s.direction > 0 and bar.low <= s.stop_price:
                    s.in_trade = False
                    s.direction = 0
                    s.hold_days = 0
                    s.stop_price = None
                    s.target_qty = 0
                    s.done_today = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="tc_stop")
                if s.direction < 0 and bar.high >= s.stop_price:
                    s.in_trade = False
                    s.direction = 0
                    s.hold_days = 0
                    s.stop_price = None
                    s.target_qty = 0
                    s.done_today = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="tc_stop")
            # Max hold
            if s.hold_days >= self.max_hold_days:
                s.in_trade = False
                s.direction = 0
                s.hold_days = 0
                s.target_qty = 0
                s.done_today = True
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="tc_max_hold")
            return None

        # ── Process pending signals at 09:35 ET ───────────────────────────
        if not s.entry_sent and et.time() >= _ENTRY_TIME:
            s.entry_sent = True

            # Close-based exit: flatten (already flat but clear pending)
            if s.pending_direction == -999:
                s.pending_direction = 0
                if cur_qty != 0:
                    s.target_qty = 0
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="tc_close_exit")
                return None

            # New signal entry
            if s.pending_direction != 0 and not s.in_trade:
                direction = s.pending_direction
                s.pending_direction = 0
                atr = self._atr()
                if atr <= 0:
                    return None
                stop_dist = self._round_to_tick(atr * self.stop_atr_mult)
                s.direction = direction
                s.in_trade = True
                s.hold_days = 0
                s.entry_date = et.date()
                if direction > 0:
                    s.stop_price = self._round_to_tick(bar.open - stop_dist)
                else:
                    s.stop_price = self._round_to_tick(bar.open + stop_dist)
                s.target_qty = direction * self.qty
                return TargetPosition(symbol=self.symbol,
                                       qty=direction * self.qty,
                                       tag="tc_enter")

        return None
