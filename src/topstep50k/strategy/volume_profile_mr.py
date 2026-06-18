"""Intraday Volume-Profile Mean Reversion (Paper 1).

Hypothesis (from the report):
  Within an RTH session, ~70% of volume transacts inside VWAP +/- 1
  volume-weighted standard deviation (the "value area"). When price
  closes meaningfully outside this band, it tends to revert toward
  VWAP. The strategy fades excursions and targets VWAP.

Mechanical rules (faithful to the paper):

  * Aggregation: 1-minute bars are aggregated INTERNALLY into 5-minute
    OHLCV buckets keyed on the bar's wall-clock minute mod 5.
  * Session: 09:30 -> 15:45 ET. RTH only. Positions flat at 15:45.
  * Running session stats (CAUSAL, recomputed at every 5-min close):
      VWAP_t  = sum_{i<=t} (close_i * volume_i) / sum_{i<=t} volume_i
      VAR_t   = sum_{i<=t} (volume_i * (close_i - VWAP_t)^2) / sum_{i<=t} volume_i
      STD_t   = sqrt(VAR_t)
      VAL_t   = VWAP_t - STD_t
      VAH_t   = VWAP_t + STD_t
    All sums are over 5-minute bars in [09:30, t] -- strict past only.
  * Signal at the 5-minute bar close:
      Long  : close <= VAL_t - threshold_points (default 2)
      Short : close >= VAH_t + threshold_points
  * Fill at the NEXT 1-minute bar's open (engine deferral). Because the
    5-min bar boundary lines up with the next 1-min open, this matches
    the paper's "next bar's open" rule.
  * Position management (per the paper):
      Profit target  = VWAP_at_signal              (FROZEN)
      Stop loss      = VWAP_at_signal -/+ 2 * STD_at_signal (FROZEN)
      Time exit      = 15:45 ET (forced flat)
    "Fixed session statistics: The VWAP and standard deviation computed
    when the signal is generated remain fixed for the trade's stop and
    target." -- Backtest Assumptions, Paper 1 p.2.
  * One trade at a time. After exit, the strategy can re-enter on a new
    signal the same day (the paper does ~5 trades/day on average).

Look-ahead audit:

  * VWAP and STD are RUNNING (only bars whose 5-min close is at or
    before t). Never the whole-session VWAP -- that bug would let the
    strategy "see" the day's eventual fair value. This is the most
    likely source of the paper's outsized Sharpe.
  * Stop/target are frozen at signal time (paper-explicit). They do
    not chase the moving VWAP.
  * Fills are deferred to next-bar-open by the engine, not the same
    bar's close.

Differences from the paper that I am DELIBERATELY making:

  * We check stops/targets on every 1-minute bar (not just at 5-min
    boundaries). This is more realistic intrabar but may close trades
    earlier than the paper. The paper says "the conservative logic
    always assumes the stop is hit first" within a bar; we apply the
    same convention bar-by-bar.

Costs: handled by the engine via `commission_per_side` on the
instrument; the paper's "1.5 pts round-trip" on NQ = $30/contract,
matching our $2.50/side standard ES rate after pt-value adjustment.
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
class _FiveMinBar:
    slot: datetime  # UTC start of the 5-min slot
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class _SessionState:
    session_date: date
    session_open_utc: datetime
    session_close_utc: datetime    # 16:00 ET (hard end of RTH)
    flat_by_utc: datetime          # 15:45 ET (force-flat by paper)
    cur_slot: datetime | None = None        # the in-progress 5-min slot
    cur_5min: _FiveMinBar | None = None     # the in-progress aggregator
    # Running session sums for the VW stats
    sum_vp: float = 0.0            # sum(close * vol) over CLOSED 5-min bars
    sum_v: float = 0.0             # sum(vol) over CLOSED 5-min bars
    sum_v_x_close_sq: float = 0.0  # sum(vol * close^2) for variance shortcut
    # Trade state
    in_trade: bool = False
    side: int = 0
    entry_price: float | None = None
    frozen_vwap: float | None = None
    frozen_std: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    trades_today: int = 0


@dataclass
class VolumeProfileMeanReversion:
    """Per-symbol volume-profile mean-reversion strategy.

    Parameters
    ----------
    symbol : bar feed.
    qty : contracts per trade.
    bar_minutes : aggregation window (default 5 to match the paper).
    entry_threshold_points : extra distance beyond VAL/VAH required for
        the signal. Default 2.0 (paper's setting).
    stop_sigma_mult : stop distance from FROZEN VWAP in sigmas. Default
        2.0 (paper's "VWAP +/- 2*STD").
    flat_before_close_minutes : minutes before RTH close to force flat.
        Default 15 (matches paper's 15:45 ET on a 16:00 close).
    session_open_local, session_close_local : RTH boundaries.
    min_warm_5min_bars : minimum number of completed 5-min bars before
        any signal can fire (warm-up). Default 6 (=30 min) so the
        session VWAP/STD has some stability.
    tick_size : symbol tick size; default 0.25.
    daily_filter : optional gate.
    """

    symbol: str
    qty: int = 1
    bar_minutes: int = 5
    entry_threshold_points: float = 2.0
    stop_sigma_mult: float = 2.0
    flat_before_close_minutes: int = 15
    session_open_local: time = time(9, 30)
    session_close_local: time = time(16, 0)
    min_warm_5min_bars: int = 6
    tick_size: float = 0.25
    daily_filter: Callable[[date], bool] | None = None
    _state: _SessionState | None = field(init=False, default=None, repr=False)
    _gated_today: bool = field(init=False, default=False, repr=False)
    _completed_5min_count: int = field(init=False, default=0, repr=False)

    # ------------------------ session bookkeeping ------------------------

    def _need_new_session(self, bar_ts: datetime) -> bool:
        if self._state is None:
            return True
        cur_d = bar_ts.astimezone(EASTERN).date()
        return cur_d != self._state.session_date

    def _build_session(self, bar_ts: datetime) -> _SessionState:
        eastern = bar_ts.astimezone(EASTERN)
        d = eastern.date()
        so = datetime.combine(d, self.session_open_local, tzinfo=EASTERN)
        sc = datetime.combine(d, self.session_close_local, tzinfo=EASTERN)
        fb = sc - timedelta(minutes=self.flat_before_close_minutes)
        tz = bar_ts.tzinfo
        return _SessionState(
            session_date=d,
            session_open_utc=so.astimezone(tz),
            session_close_utc=sc.astimezone(tz),
            flat_by_utc=fb.astimezone(tz),
        )

    def _slot_for(self, bar_ts: datetime) -> datetime:
        """Bucket the 1-min bar into its 5-min slot (slot start, UTC)."""
        e = bar_ts.astimezone(EASTERN)
        slot_minute = (e.minute // self.bar_minutes) * self.bar_minutes
        e_slot = e.replace(minute=slot_minute, second=0, microsecond=0)
        return e_slot.astimezone(bar_ts.tzinfo)

    # ----- running VW stats from CLOSED 5-min bars (strict past) --------

    def _vwap(self) -> float | None:
        s = self._state
        if s is None or s.sum_v <= 0:
            return None
        return s.sum_vp / s.sum_v

    def _std(self) -> float | None:
        s = self._state
        if s is None or s.sum_v <= 0:
            return None
        mean = s.sum_vp / s.sum_v
        var = (s.sum_v_x_close_sq / s.sum_v) - mean * mean
        if var <= 0:
            return 0.0
        return var ** 0.5

    def _absorb_closed_5min(self, b: _FiveMinBar) -> None:
        """Add a JUST-CLOSED 5-min bar's stats to the running sums."""
        s = self._state
        assert s is not None
        s.sum_vp += b.close * b.volume
        s.sum_v += b.volume
        s.sum_v_x_close_sq += b.volume * b.close * b.close
        self._completed_5min_count += 1

    # ------------------------- main on_bar ------------------------------

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        if self._need_new_session(bar.ts):
            # Roll the day. NEVER carry sums across the daily boundary.
            self._state = self._build_session(bar.ts)
            self._completed_5min_count = 0
            if self.daily_filter is not None:
                d = bar.ts.astimezone(EASTERN).date()
                self._gated_today = not self.daily_filter(d)
            else:
                self._gated_today = False
        s = self._state

        # Outside session: stay flat, do not feed bars into stats.
        if bar.ts < s.session_open_utc or bar.ts >= s.session_close_utc:
            if ctx.position(self.symbol) != 0:
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="vp_session_end")
            return None
        if self._gated_today:
            if ctx.position(self.symbol) != 0:
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="vp_gated_flat")
            return None

        # 1) Update 5-min aggregator -- possibly closing a prior slot.
        slot = self._slot_for(bar.ts)
        closed_bar: _FiveMinBar | None = None
        if s.cur_slot is None:
            s.cur_slot = slot
            s.cur_5min = _FiveMinBar(slot=slot, open=bar.open, high=bar.high,
                                      low=bar.low, close=bar.close,
                                      volume=bar.volume)
        elif slot != s.cur_slot:
            # We have crossed a slot boundary. The PREVIOUS slot is done.
            closed_bar = s.cur_5min
            s.cur_slot = slot
            s.cur_5min = _FiveMinBar(slot=slot, open=bar.open, high=bar.high,
                                      low=bar.low, close=bar.close,
                                      volume=bar.volume)
        else:
            # Same slot: extend the in-progress 5-min bar.
            cur = s.cur_5min
            cur.high = max(cur.high, bar.high)
            cur.low = min(cur.low, bar.low)
            cur.close = bar.close
            cur.volume += bar.volume

        if closed_bar is not None:
            self._absorb_closed_5min(closed_bar)

        # 2) Exits: check stop / target / time / EOD on EVERY 1-min bar.
        cur_qty = ctx.position(self.symbol)
        if s.in_trade and cur_qty != 0 and s.side != 0:
            # Time exit.
            if bar.ts >= s.flat_by_utc:
                s.in_trade = False
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="vp_time_exit")
            # Stop (assume stop hits before target if both in same bar -- paper)
            if self._hit_stop(bar, s):
                s.in_trade = False
                return TargetPosition(symbol=self.symbol, qty=0, tag="vp_stop")
            if self._hit_tp(bar, s):
                s.in_trade = False
                return TargetPosition(symbol=self.symbol, qty=0, tag="vp_target")
            return None

        # 3) Entry: only on the close of a 5-min bar (i.e., when we JUST
        # closed a 5-min slot). The just-closed bar's close is the decision
        # price; the next 1-min bar's open is the fill.
        if closed_bar is None:
            return None
        if self._completed_5min_count < self.min_warm_5min_bars:
            return None
        vwap = self._vwap()
        std = self._std()
        if vwap is None or std is None or std <= 0:
            return None
        val = vwap - std
        vah = vwap + std
        signal = 0
        if closed_bar.close <= val - self.entry_threshold_points:
            signal = +1   # long fade of a dip below VAL
        elif closed_bar.close >= vah + self.entry_threshold_points:
            signal = -1   # short fade of a spike above VAH

        if signal == 0:
            return None
        # Freeze VWAP & STD at signal time for the duration of the trade.
        s.in_trade = True
        s.side = signal
        # Entry price approximation -- engine fills at next bar's open.
        s.entry_price = closed_bar.close
        s.frozen_vwap = vwap
        s.frozen_std = std
        if signal > 0:
            s.stop_price = vwap - self.stop_sigma_mult * std
            s.target_price = vwap
        else:
            s.stop_price = vwap + self.stop_sigma_mult * std
            s.target_price = vwap
        s.trades_today += 1
        return TargetPosition(
            symbol=self.symbol, qty=signal * self.qty,
            tag=f"vp_{'long' if signal > 0 else 'short'}",
        )

    def _hit_stop(self, bar: Bar, s: _SessionState) -> bool:
        if s.stop_price is None:
            return False
        if s.side > 0:
            return bar.low <= s.stop_price
        return bar.high >= s.stop_price

    def _hit_tp(self, bar: Bar, s: _SessionState) -> bool:
        if s.target_price is None:
            return False
        if s.side > 0:
            return bar.high >= s.target_price
        return bar.low <= s.target_price
