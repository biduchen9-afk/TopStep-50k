"""Sweep → MSS / Impulse strategy (Pine Script port).

Ported from the user's Pine Script `USTEC Version B2 - Sweep -> MSS/Impulse`
strategy. Logic:

  1. Track session highs/lows during two liquidity windows (Asia/Bangkok
     time): London 12:00-14:00 and NY 18:00-19:30.
  2. After each window closes, watch for a SWEEP — price extends BEYOND
     the window's high or low.
  3. After sweep, wait up to `max_wait_bars_after_sweep` (5-minute bars)
     for an impulse 5-min bar in the OPPOSITE direction that also makes
     a market-structure shift (MSS — close beyond an N-bar prior extreme).
  4. Enter on the signal bar's close (engine defers fill to next 1-min open).
  5. Stop = sweep extreme (extended further by MSS reference); target =
     entry ± (risk × RR).
  6. Force close at 23:59 Bangkok (16:59 UTC).
  7. One trade per window per day.

This strategy consumes 1-minute bars and aggregates internally to 5-minute
bars for the impulse + MSS evaluation, matching the Pine Script's use of
`request.security(syminfo.tickerid, "5", ...)`.

Note on no-look-ahead: the strategy ONLY reads bar[t] data inside on_bar(t),
and the engine fills any emitted TargetPosition at the open of bar[t+1].
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Deque
from zoneinfo import ZoneInfo

from topstep50k.engine.types import Bar
from topstep50k.strategy.base import StrategyContext, TargetPosition


BANGKOK = ZoneInfo("Asia/Bangkok")


@dataclass
class _Bar5m:
    """Completed 5-minute bar."""
    ts_end: datetime  # close time of the 5-min interval
    open: float
    high: float
    low: float
    close: float


@dataclass
class _Accum5m:
    """In-progress 5-minute bar accumulator."""
    slot_end_minute_key: tuple  # (year, month, day, hour, slot_end_minute_in_hour)
    open: float
    high: float
    low: float
    close: float
    ts_end_template: datetime  # last 1-min bar seen, used for ts_end on close


@dataclass
class _DayState:
    london_high: float | None = None
    london_low: float | None = None
    ny_high: float | None = None
    ny_low: float | None = None

    london_swept_high: bool = False
    london_swept_low: bool = False
    ny_swept_high: bool = False
    ny_swept_low: bool = False

    london_sweep_5m_index: int | None = None  # index in 5m history when sweep occurred
    ny_sweep_5m_index: int | None = None
    london_sweep_extreme_high: float | None = None
    london_sweep_extreme_low: float | None = None
    ny_sweep_extreme_high: float | None = None
    ny_sweep_extreme_low: float | None = None

    london_traded: bool = False
    ny_traded: bool = False

    # Active position state
    active_side: int = 0          # +1 long, -1 short, 0 flat
    active_stop: float | None = None
    active_target: float | None = None
    force_closed: bool = False    # already issued the daily force-close

    # Previous-bar signal state (for edge-detection: newBear = bearSig & ~prev)
    prev_bear_signal: bool = False
    prev_bull_signal: bool = False


@dataclass
class SweepMSS:
    """Sweep → MSS/Impulse strategy.

    Parameters mirror the Pine Script user inputs. Defaults match the
    user's tuned settings (5m ATR mult=0.65, close-in-extreme=0.65,
    MSS lookback=5) NOT the Pine Script defaults.
    """

    symbol: str
    qty: int = 1
    tick_size: float = 0.10  # GC default

    # Session windows in Asia/Bangkok local time
    london_start: time = time(12, 0)
    london_end: time = time(14, 0)
    ny_start: time = time(18, 0)
    ny_end: time = time(19, 30)
    force_close_time: time = time(23, 59)

    use_london: bool = True
    use_ny: bool = True

    # Strategy logic params (user-tuned defaults)
    rr: float = 2.0
    atr_len_5m: int = 14
    atr_mult: float = 0.65
    close_near_extreme: float = 0.65
    mss_lookback_5m: int = 5
    max_wait_bars_after_sweep: int = 24

    # ── Internal state (init=False) ────────────────────────────────────
    _bars5m: Deque[_Bar5m] = field(init=False, repr=False,
                                    default_factory=lambda: deque(maxlen=64))
    _accum: _Accum5m | None = field(init=False, default=None, repr=False)
    _last_day_bkk: date | None = field(init=False, default=None, repr=False)
    _state: _DayState = field(init=False, default_factory=_DayState, repr=False)
    _prev_atr: float | None = field(init=False, default=None, repr=False)
    _tr_buffer: Deque[float] = field(init=False, repr=False,
                                      default_factory=lambda: deque(maxlen=14))
    _prev_close_5m: float | None = field(init=False, default=None, repr=False)

    # ── 5-minute aggregation ──────────────────────────────────────────
    def _slot_key(self, ts: datetime) -> tuple:
        """Stable key identifying the 5-min interval a 1-min bar belongs
        to. The interval (a, a+5min] is identified by its end-minute."""
        # ts is the CLOSE time of the 1-min bar covering (ts-1m, ts].
        # If ts.minute is 0, 5, 10, ... that bar is the LAST one of the
        # 5-min interval ending at ts.
        # If ts.minute is 1-4, 6-9, ..., it's an intermediate bar of the
        # interval ending at the next multiple of 5.
        m = ts.minute
        # Slot end minute (in the same hour, possibly rolling over)
        if m == 0:
            slot_end_min = 0  # this bar belongs to the slot ending at this minute
            return (ts.year, ts.month, ts.day, ts.hour, slot_end_min)
        slot_end_min = ((m - 1) // 5 + 1) * 5
        if slot_end_min == 60:
            # rolls over to next hour
            from datetime import timedelta
            nxt = ts.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return (nxt.year, nxt.month, nxt.day, nxt.hour, 0)
        return (ts.year, ts.month, ts.day, ts.hour, slot_end_min)

    def _ingest_1m(self, bar: Bar) -> _Bar5m | None:
        """Fold a 1-min bar into the rolling 5-min accumulator.
        Returns a freshly-closed 5-min bar (or None if not yet complete)."""
        key = self._slot_key(bar.ts)
        closed: _Bar5m | None = None
        if self._accum is None or self._accum.slot_end_minute_key != key:
            if self._accum is not None:
                closed = _Bar5m(
                    ts_end=self._accum.ts_end_template,
                    open=self._accum.open, high=self._accum.high,
                    low=self._accum.low, close=self._accum.close,
                )
            self._accum = _Accum5m(
                slot_end_minute_key=key,
                open=bar.open, high=bar.high, low=bar.low, close=bar.close,
                ts_end_template=bar.ts,
            )
        else:
            self._accum.high = max(self._accum.high, bar.high)
            self._accum.low  = min(self._accum.low, bar.low)
            self._accum.close = bar.close
            self._accum.ts_end_template = bar.ts
        return closed

    def _push_5m(self, b5: _Bar5m) -> None:
        """Add a closed 5-min bar to history and update ATR(14)."""
        # True Range
        if self._prev_close_5m is None:
            tr = b5.high - b5.low
        else:
            tr = max(b5.high - b5.low,
                     abs(b5.high - self._prev_close_5m),
                     abs(b5.low  - self._prev_close_5m))
        self._tr_buffer.append(tr)
        if len(self._tr_buffer) >= self.atr_len_5m:
            self._prev_atr = sum(self._tr_buffer) / len(self._tr_buffer)
        else:
            self._prev_atr = sum(self._tr_buffer) / max(1, len(self._tr_buffer))
        self._prev_close_5m = b5.close
        self._bars5m.append(b5)

    # ── Per-day reset ─────────────────────────────────────────────────
    def _bkk_date(self, ts: datetime) -> date:
        return ts.astimezone(BANGKOK).date()

    def _maybe_roll_day(self, bar: Bar) -> None:
        d = self._bkk_date(bar.ts)
        if self._last_day_bkk != d:
            self._last_day_bkk = d
            self._state = _DayState()

    # ── Window membership ─────────────────────────────────────────────
    def _in_window(self, ts: datetime, start: time, end: time) -> bool:
        t = ts.astimezone(BANGKOK).time()
        return start <= t < end

    def _past_window(self, ts: datetime, end: time) -> bool:
        return ts.astimezone(BANGKOK).time() >= end

    # ── Main entry point ──────────────────────────────────────────────
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        self._maybe_roll_day(bar)
        s = self._state

        # ── Manage open position FIRST (intra-bar TP/SL check) ─────────
        cur_qty = ctx.position(self.symbol)
        if cur_qty != 0 and s.active_side != 0 and not s.force_closed:
            if s.active_side > 0:
                if s.active_stop is not None and bar.low <= s.active_stop:
                    s.active_side = 0
                    s.force_closed = True  # one-shot per window already locked in
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="sweepmss_sl")
                if s.active_target is not None and bar.high >= s.active_target:
                    s.active_side = 0
                    s.force_closed = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="sweepmss_tp")
            else:  # short
                if s.active_stop is not None and bar.high >= s.active_stop:
                    s.active_side = 0
                    s.force_closed = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="sweepmss_sl")
                if s.active_target is not None and bar.low <= s.active_target:
                    s.active_side = 0
                    s.force_closed = True
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="sweepmss_tp")

        # ── Force close at end of trading day (Bangkok 23:59) ──────────
        bkk_time = bar.ts.astimezone(BANGKOK).time()
        if not s.force_closed and bkk_time >= self.force_close_time:
            if cur_qty != 0:
                s.force_closed = True
                s.active_side = 0
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="sweepmss_force_close")
            s.force_closed = True
            return None

        # ── Track session high/low during liquidity windows ────────────
        if self.use_london and self._in_window(bar.ts,
                                                self.london_start,
                                                self.london_end):
            s.london_high = bar.high if s.london_high is None else max(s.london_high, bar.high)
            s.london_low  = bar.low  if s.london_low  is None else min(s.london_low, bar.low)
        if self.use_ny and self._in_window(bar.ts, self.ny_start, self.ny_end):
            s.ny_high = bar.high if s.ny_high is None else max(s.ny_high, bar.high)
            s.ny_low  = bar.low  if s.ny_low  is None else min(s.ny_low, bar.low)

        # ── Detect sweep (after window has ended) ─────────────────────
        if (self.use_london and not s.london_traded
                and s.london_high is not None and s.london_low is not None
                and self._past_window(bar.ts, self.london_end)):
            if not s.london_swept_high and bar.high > s.london_high:
                s.london_swept_high = True
                s.london_sweep_extreme_high = bar.high
                s.london_sweep_5m_index = len(self._bars5m)
            if not s.london_swept_low and bar.low < s.london_low:
                s.london_swept_low = True
                s.london_sweep_extreme_low = bar.low
                s.london_sweep_5m_index = len(self._bars5m)
        if (self.use_ny and not s.ny_traded
                and s.ny_high is not None and s.ny_low is not None
                and self._past_window(bar.ts, self.ny_end)):
            if not s.ny_swept_high and bar.high > s.ny_high:
                s.ny_swept_high = True
                s.ny_sweep_extreme_high = bar.high
                s.ny_sweep_5m_index = len(self._bars5m)
            if not s.ny_swept_low and bar.low < s.ny_low:
                s.ny_swept_low = True
                s.ny_sweep_extreme_low = bar.low
                s.ny_sweep_5m_index = len(self._bars5m)

        # ── Aggregate to 5-min ────────────────────────────────────────
        closed_5m = self._ingest_1m(bar)
        if closed_5m is None:
            return None
        self._push_5m(closed_5m)

        # We have a freshly-closed 5-min bar. Evaluate impulse + MSS.
        if cur_qty != 0:
            return None  # already in position — no new entries

        if len(self._bars5m) < self.mss_lookback_5m + 1 or self._prev_atr is None:
            # update prev signal state but no entry yet
            s.prev_bear_signal = False
            s.prev_bull_signal = False
            return None

        b = self._bars5m[-1]
        body = abs(b.close - b.open)
        rng = max(b.high - b.low, self.tick_size)
        close_pct = (b.close - b.low) / rng

        bear_impulse = (b.close < b.open
                         and body >= self._prev_atr * self.atr_mult
                         and close_pct <= (1.0 - self.close_near_extreme))
        bull_impulse = (b.close > b.open
                         and body >= self._prev_atr * self.atr_mult
                         and close_pct >= self.close_near_extreme)

        # priorHigh / priorLow over the previous `mss_lookback_5m` bars
        # (Pine: ta.highest(high[1], mssLookback5) means bars indexed 1..lookback)
        prior_bars = list(self._bars5m)[-(self.mss_lookback_5m + 1):-1]
        prior_high = max(pb.high for pb in prior_bars)
        prior_low  = min(pb.low  for pb in prior_bars)

        bear_mss = b.close < prior_low
        bull_mss = b.close > prior_high

        bear_sig = bear_impulse and bear_mss
        bull_sig = bull_impulse and bull_mss

        new_bear = bear_sig and not s.prev_bear_signal
        new_bull = bull_sig and not s.prev_bull_signal

        s.prev_bear_signal = bear_sig
        s.prev_bull_signal = bull_sig

        if not (new_bear or new_bull):
            return None

        # ── Decide which sweep this signal pairs with ─────────────────
        # NY-first priority (matches Pine Script ordering)
        def _bars_since(ix: int | None) -> int | None:
            if ix is None:
                return None
            return len(self._bars5m) - 1 - ix  # how many 5m closed since sweep

        # SHORT signal — pairs with sweep of a HIGH
        valid_ny_short = (self.use_ny and not s.ny_traded
                           and s.ny_swept_high
                           and _bars_since(s.ny_sweep_5m_index) is not None
                           and _bars_since(s.ny_sweep_5m_index) <= self.max_wait_bars_after_sweep
                           and new_bear)
        valid_london_short = (self.use_london and not s.london_traded
                              and s.london_swept_high
                              and _bars_since(s.london_sweep_5m_index) is not None
                              and _bars_since(s.london_sweep_5m_index) <= self.max_wait_bars_after_sweep
                              and new_bear)
        valid_ny_long = (self.use_ny and not s.ny_traded
                          and s.ny_swept_low
                          and _bars_since(s.ny_sweep_5m_index) is not None
                          and _bars_since(s.ny_sweep_5m_index) <= self.max_wait_bars_after_sweep
                          and new_bull)
        valid_london_long = (self.use_london and not s.london_traded
                              and s.london_swept_low
                              and _bars_since(s.london_sweep_5m_index) is not None
                              and _bars_since(s.london_sweep_5m_index) <= self.max_wait_bars_after_sweep
                              and new_bull)

        short_entry = valid_ny_short or (valid_london_short and not valid_ny_short)
        long_entry  = valid_ny_long  or (valid_london_long  and not valid_ny_long)

        if not (short_entry or long_entry):
            return None

        entry_px = b.close
        if long_entry:
            sweep_low = (s.ny_sweep_extreme_low if valid_ny_long
                          else s.london_sweep_extreme_low)
            stop = min(sweep_low, prior_low)
            risk = entry_px - stop
            if risk <= self.tick_size:
                return None
            target = entry_px + risk * self.rr
            stop = round(stop / self.tick_size) * self.tick_size
            target = round(target / self.tick_size) * self.tick_size
            s.active_side = +1
            s.active_stop = stop
            s.active_target = target
            if valid_ny_long:
                s.ny_traded = True
            else:
                s.london_traded = True
            return TargetPosition(symbol=self.symbol, qty=self.qty,
                                   tag="sweepmss_long")
        else:  # short
            sweep_high = (s.ny_sweep_extreme_high if valid_ny_short
                           else s.london_sweep_extreme_high)
            stop = max(sweep_high, prior_high)
            risk = stop - entry_px
            if risk <= self.tick_size:
                return None
            target = entry_px - risk * self.rr
            stop = round(stop / self.tick_size) * self.tick_size
            target = round(target / self.tick_size) * self.tick_size
            s.active_side = -1
            s.active_stop = stop
            s.active_target = target
            if valid_ny_short:
                s.ny_traded = True
            else:
                s.london_traded = True
            return TargetPosition(symbol=self.symbol, qty=-self.qty,
                                   tag="sweepmss_short")
