"""VWAP + StdDev Bands + Williams Fractal Pullback (Paper 2).

Hypothesis (fine-tuned version, RR=3:1):
  Inside an RTH session, trend bias is set by the running VWAP. When
  price pulls back to a standard-deviation band (1x to 4x) AND a
  Williams fractal confirms a swing reversal, take an entry in the
  direction of the prevailing VWAP trend with a wide-target (3x stop)
  exit.

Mechanical rules (faithful to the paper):

  * Bars: 15-minute, internally aggregated from 1-min input.
  * Session: 09:30 -> 16:00 ET. Dead zone 12:00-15:00 (no NEW entries).
    Time exit at 15:45 ET.
  * Running session VWAP from 09:30 (causal, close * volume weighted).
  * EXPANDING standard deviation of hlc3 within the session.
  * Bands: +/-{1, 2, 2.25, 3, 3.5, 4} * STD around VWAP.
  * Williams 5-bar fractal: bar T is a bullish fractal (swing low) if
    low[T-2] > low[T] AND low[T-1] > low[T] AND low[T+1] > low[T] AND
    low[T+2] > low[T]. Symmetric for bearish. The fractal at T is
    KNOWN only at T+2 (when the +2 confirmation bar closes). We use
    the +2 bar's close as the decision point -- paper: "confirm two
    bars later."
  * Entry rules at decision bar T+2:
      Long  : close > VWAP AND prev bar's close is within 2 pts of any
              LOWER band AND a bullish fractal at T was confirmed.
      Short : close < VWAP AND prev bar's close is within 2 pts of any
              UPPER band AND a bearish fractal at T was confirmed.
  * Stop distance: max(0.5 * band_mult * session_std, 8 pts).
    The band_mult used is the band whose proximity triggered the entry.
  * Profit target: 3x stop distance (RR=3:1, paper's fine-tuning).
  * VWAP flip exit: if long and bar close < VWAP, exit; symmetric short.
  * Time exit at 15:45 ET.

Look-ahead audit:

  * Running VWAP uses only bars in [09:30, current_15min_close]. CAUSAL.
  * STD is an EXPANDING stdev of hlc3 -- causal, includes only past bars.
  * Fractal at T is identified at T+2; we enter at T+2 close + slip.
    That is causal as long as we never look beyond T+2.
  * NO use of the next bar's open price for any decision; the engine
    defers fills.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Deque
from zoneinfo import ZoneInfo

from topstep50k.engine.types import Bar
from topstep50k.strategy.base import StrategyContext, TargetPosition


EASTERN = ZoneInfo("America/New_York")

# Bands (paper-explicit ordering matters for the proximity match)
BAND_MULTS = [1.0, 2.0, 2.25, 3.0, 3.5, 4.0]


@dataclass
class _Bar15:
    slot: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class _SessState:
    session_date: date
    session_open_utc: datetime
    session_close_utc: datetime
    flat_by_utc: datetime
    dead_start_utc: datetime
    dead_end_utc: datetime
    # 15-min aggregator
    cur_slot: datetime | None = None
    cur_15: _Bar15 | None = None
    # Closed 15-min bars in session order (for fractal/expanding-std use)
    closed_bars: list[_Bar15] = field(default_factory=list)
    # Running session VWAP sums (over CLOSED 15-min bars)
    sum_vp: float = 0.0
    sum_v: float = 0.0
    # Expanding-stdev sums for hlc3
    sum_hlc3: float = 0.0
    sum_hlc3_sq: float = 0.0
    n_closed: int = 0
    # Trade state
    in_trade: bool = False
    side: int = 0
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None


@dataclass
class VwapStdFractal:
    """VWAP + StdDev + Williams-Fractal pullback strategy (RR=3)."""

    symbol: str
    qty: int = 1
    bar_minutes: int = 15
    band_proximity_points: float = 2.0
    min_stop_points: float = 8.0
    rr: float = 3.0
    flat_before_close_minutes: int = 15
    dead_zone_start_local: time = time(12, 0)
    dead_zone_end_local: time = time(15, 0)
    session_open_local: time = time(9, 30)
    session_close_local: time = time(16, 0)
    tick_size: float = 0.25
    daily_filter: Callable[[date], bool] | None = None
    _state: _SessState | None = field(init=False, default=None, repr=False)
    _gated_today: bool = field(init=False, default=False, repr=False)

    # --------- session bookkeeping ---------------------------------------

    def _need_new_session(self, bar_ts: datetime) -> bool:
        if self._state is None:
            return True
        cur_d = bar_ts.astimezone(EASTERN).date()
        return cur_d != self._state.session_date

    def _build_session(self, bar_ts: datetime) -> _SessState:
        eastern = bar_ts.astimezone(EASTERN)
        d = eastern.date()
        so = datetime.combine(d, self.session_open_local, tzinfo=EASTERN)
        sc = datetime.combine(d, self.session_close_local, tzinfo=EASTERN)
        ds = datetime.combine(d, self.dead_zone_start_local, tzinfo=EASTERN)
        de = datetime.combine(d, self.dead_zone_end_local, tzinfo=EASTERN)
        fb = sc - timedelta(minutes=self.flat_before_close_minutes)
        tz = bar_ts.tzinfo
        return _SessState(
            session_date=d,
            session_open_utc=so.astimezone(tz),
            session_close_utc=sc.astimezone(tz),
            flat_by_utc=fb.astimezone(tz),
            dead_start_utc=ds.astimezone(tz),
            dead_end_utc=de.astimezone(tz),
        )

    def _slot_for(self, bar_ts: datetime) -> datetime:
        e = bar_ts.astimezone(EASTERN)
        slot_minute = (e.minute // self.bar_minutes) * self.bar_minutes
        e_slot = e.replace(minute=slot_minute, second=0, microsecond=0)
        return e_slot.astimezone(bar_ts.tzinfo)

    # ------------- running stats over CLOSED 15-min bars -----------------

    def _vwap(self) -> float | None:
        s = self._state
        if s.sum_v <= 0:
            return None
        return s.sum_vp / s.sum_v

    def _std_hlc3(self) -> float | None:
        s = self._state
        if s.n_closed < 2:
            return None
        mean = s.sum_hlc3 / s.n_closed
        var = (s.sum_hlc3_sq / s.n_closed) - mean * mean
        if var <= 0:
            return 0.0
        return var ** 0.5

    def _absorb_closed(self, b: _Bar15) -> None:
        s = self._state
        s.sum_vp += b.close * b.volume
        s.sum_v += b.volume
        hlc3 = (b.high + b.low + b.close) / 3.0
        s.sum_hlc3 += hlc3
        s.sum_hlc3_sq += hlc3 * hlc3
        s.n_closed += 1
        s.closed_bars.append(b)

    # -------- Williams 5-bar fractal at T (need bars T-2..T+2) -----------

    @staticmethod
    def _bullish_fractal_at(bars: list[_Bar15], t_idx: int) -> bool:
        if t_idx < 2 or t_idx > len(bars) - 3:
            return False
        L = bars[t_idx].low
        return (bars[t_idx - 2].low > L and bars[t_idx - 1].low > L
                and bars[t_idx + 1].low > L and bars[t_idx + 2].low > L)

    @staticmethod
    def _bearish_fractal_at(bars: list[_Bar15], t_idx: int) -> bool:
        if t_idx < 2 or t_idx > len(bars) - 3:
            return False
        H = bars[t_idx].high
        return (bars[t_idx - 2].high < H and bars[t_idx - 1].high < H
                and bars[t_idx + 1].high < H and bars[t_idx + 2].high < H)

    # ----------- band proximity ------------------------------------------

    def _nearest_lower_band(self, vwap: float, std: float, price: float
                             ) -> tuple[float | None, float | None]:
        """Return (band_mult, band_value) for the nearest lower band the
        price is within `band_proximity_points` of. None if not near any."""
        for m in BAND_MULTS:
            band_val = vwap - m * std
            if abs(price - band_val) <= self.band_proximity_points:
                return m, band_val
        return None, None

    def _nearest_upper_band(self, vwap: float, std: float, price: float
                             ) -> tuple[float | None, float | None]:
        for m in BAND_MULTS:
            band_val = vwap + m * std
            if abs(price - band_val) <= self.band_proximity_points:
                return m, band_val
        return None, None

    # ------------------- on_bar (driven by 1-min bars) -------------------

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        if self._need_new_session(bar.ts):
            self._state = self._build_session(bar.ts)
            if self.daily_filter is not None:
                d = bar.ts.astimezone(EASTERN).date()
                self._gated_today = not self.daily_filter(d)
            else:
                self._gated_today = False
        s = self._state

        if bar.ts < s.session_open_utc or bar.ts >= s.session_close_utc:
            if ctx.position(self.symbol) != 0:
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="vsf_session_end")
            return None
        if self._gated_today:
            if ctx.position(self.symbol) != 0:
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="vsf_gated_flat")
            return None

        # 1) Update 15-min aggregator, possibly closing a slot
        slot = self._slot_for(bar.ts)
        closed: _Bar15 | None = None
        if s.cur_slot is None:
            s.cur_slot = slot
            s.cur_15 = _Bar15(slot=slot, open=bar.open, high=bar.high,
                               low=bar.low, close=bar.close,
                               volume=bar.volume)
        elif slot != s.cur_slot:
            closed = s.cur_15
            s.cur_slot = slot
            s.cur_15 = _Bar15(slot=slot, open=bar.open, high=bar.high,
                               low=bar.low, close=bar.close,
                               volume=bar.volume)
        else:
            cur = s.cur_15
            cur.high = max(cur.high, bar.high)
            cur.low = min(cur.low, bar.low)
            cur.close = bar.close
            cur.volume += bar.volume

        if closed is not None:
            self._absorb_closed(closed)

        # 2) Exits on every 1-min bar
        cur_qty = ctx.position(self.symbol)
        if s.in_trade and cur_qty != 0 and s.side != 0:
            if bar.ts >= s.flat_by_utc:
                s.in_trade = False
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="vsf_time_exit")
            # Stop assumed first if both stop and target in same bar
            if self._hit_stop(bar, s):
                s.in_trade = False
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="vsf_stop")
            if self._hit_tp(bar, s):
                s.in_trade = False
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="vsf_target")
            # VWAP flip exit (checked on bar close)
            vwap_now = self._vwap()
            if vwap_now is not None:
                if s.side > 0 and bar.close < vwap_now:
                    s.in_trade = False
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="vsf_vwap_flip")
                if s.side < 0 and bar.close > vwap_now:
                    s.in_trade = False
                    return TargetPosition(symbol=self.symbol, qty=0,
                                           tag="vsf_vwap_flip")
            return None

        # 3) Entry decision -- only on the 15-min boundary AND only after
        # we have at least 5 closed bars (need T-2..T+2 fractal window).
        if closed is None:
            return None
        # Dead zone: 12:00 - 15:00 ET, no NEW entries (but exits already
        # processed above).
        if s.dead_start_utc <= bar.ts < s.dead_end_utc:
            return None
        if s.n_closed < 5:
            return None

        vwap = self._vwap()
        std = self._std_hlc3()
        if vwap is None or std is None or std <= 0:
            return None

        # The CURRENT decision bar (close at bar.ts) is the LAST closed
        # 15-min bar, i.e., index len(closed_bars) - 1. The fractal we
        # are looking for sits 2 bars earlier (so its T+2 confirmation
        # is the current bar).
        bars = s.closed_bars
        cur_idx = len(bars) - 1
        T_idx = cur_idx - 2     # fractal sits here
        if T_idx < 2:
            return None
        decision_bar = bars[cur_idx]
        prev_bar = bars[cur_idx - 1]  # "previous bar's close" per paper

        # Entry conditions
        signal = 0
        band_mult = None
        if (decision_bar.close > vwap
                and self._bullish_fractal_at(bars, T_idx)):
            band_mult, _ = self._nearest_lower_band(vwap, std, prev_bar.close)
            if band_mult is not None:
                signal = +1
        if (signal == 0 and decision_bar.close < vwap
                and self._bearish_fractal_at(bars, T_idx)):
            band_mult, _ = self._nearest_upper_band(vwap, std, prev_bar.close)
            if band_mult is not None:
                signal = -1

        if signal == 0:
            return None

        # Compute stop/target distances per paper formula
        stop_dist = max(0.5 * band_mult * std, self.min_stop_points)
        target_dist = self.rr * stop_dist
        entry_ref = decision_bar.close
        if signal > 0:
            s.stop_price = entry_ref - stop_dist
            s.target_price = entry_ref + target_dist
        else:
            s.stop_price = entry_ref + stop_dist
            s.target_price = entry_ref - target_dist
        s.in_trade = True
        s.side = signal
        s.entry_price = entry_ref
        return TargetPosition(
            symbol=self.symbol, qty=signal * self.qty,
            tag=f"vsf_{'long' if signal > 0 else 'short'}",
        )

    def _hit_stop(self, bar: Bar, s: _SessState) -> bool:
        if s.stop_price is None:
            return False
        if s.side > 0:
            return bar.low <= s.stop_price
        return bar.high >= s.stop_price

    def _hit_tp(self, bar: Bar, s: _SessState) -> bool:
        if s.target_price is None:
            return False
        if s.side > 0:
            return bar.high >= s.target_price
        return bar.low <= s.target_price
