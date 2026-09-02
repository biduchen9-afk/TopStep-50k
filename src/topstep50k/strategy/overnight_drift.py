"""Overnight drift -- long the index from just-before RTH close to
just-after next RTH open.

The well-documented "overnight equity premium": measured open-to-prior-
close returns on SPY / ES are positive on average over long horizons
while RTH-only returns are roughly flat or negative. The simplest
strategy that captures this is a long-only flip:

  * Enter long at `entry_offset_minutes` before RTH close.
  * Exit at `exit_offset_minutes` after the next RTH open.
  * No intraday signal, no stop -- pure passive overnight exposure.
    (Optional `overnight_stop_pct` available; default OFF.)

This is intentionally as different from intraday strategies as
possible: zero overlap with RTH hours, zero correlation with ORB or
mean-reversion in expectation.

Long-only? Yes. The documented edge is unidirectional. A signed
version (e.g. trend-follow the prior RTH session's direction) is
straightforward to add later but starts as long-only here.

EDGE WARNING: this edge is small and can disappear in bear regimes
(was negative in 2022 for SPY). Sign of the strategy is empirical
on this dataset.

RETIRED 2026-08-26: TopStep does not allow overnight or weekend
holding at ANY account stage (Combine, XFA, or Live Funded) -- all
positions must be flat by 3:10 PM CT every day, confirmed current as
of a July 1, 2026 rules update (see docs/rules_sources.md). This
strategy's entire premise -- holding from before RTH close to after
the next RTH open -- is therefore not tradeable on this platform at
all, independent of how well it backtests. It is kept in the codebase
(and its bugfix history preserved) for reference and because the
underlying edge/engine code is still useful to have tested, but it
must not be included in any live-deployed ensemble weight going
forward. See evaluate_ensemble_databento_recent_evgate.py's retirement
note for the ensemble-level consequence.
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
class _ODState:
    last_seen_date: object | None = None  # python date
    target_qty: int = 0  # what we want our position to be
    # entry/exit thresholds for the current US/Eastern trading day
    cur_session_open_utc: datetime | None = None
    cur_entry_utc: datetime | None = None       # 15:55 ET prev day
    cur_exit_utc: datetime | None = None        # 09:35 ET this day
    entry_filter_rejected: bool = False         # True once filter says no for this cycle
    exit_done_today: bool = False               # True once today's exit has fired


@dataclass
class OvernightDrift:
    """Hold one contract long from `entry_offset_minutes` before RTH
    close to `exit_offset_minutes` after next RTH open.

    Parameters
    ----------
    symbol : bar feed symbol.
    qty : signed contract count; default +1 (long-only overnight).
        Set to -1 to test reversed direction.
    entry_offset_minutes : minutes before RTH close to enter. Default 5.
    exit_offset_minutes : minutes after RTH open to exit. Default 5.
    session_open_local / session_close_local : RTH boundaries.
    """

    symbol: str
    qty: int = 1
    entry_offset_minutes: int = 5
    exit_offset_minutes: int = 5
    session_open_local: time = time(9, 30)
    session_close_local: time = time(16, 0)
    # Optional per-entry gate. Called once when the entry trigger fires
    # (in the late afternoon of day N). The argument is the US/Eastern
    # date of the trade's entry day. If it returns False the strategy
    # skips that overnight cycle. This is the place to plug in the
    # NY-Fed-style "prior RTH session down" conditioner.
    entry_filter: Callable[[date], bool] | None = None
    _state: _ODState = field(init=False, default_factory=_ODState, repr=False)

    def _maybe_roll_day(self, bar_ts: datetime) -> None:
        eastern = bar_ts.astimezone(EASTERN)
        d = eastern.date()
        if self._state.last_seen_date == d:
            return
        self._state.last_seen_date = d
        self._state.entry_filter_rejected = False
        self._state.exit_done_today = False
        tz = bar_ts.tzinfo
        so_local = datetime.combine(d, self.session_open_local, tzinfo=EASTERN)
        sc_local = datetime.combine(d, self.session_close_local, tzinfo=EASTERN)
        # Today's RTH open and today's RTH close, in bar-tz
        self._state.cur_session_open_utc = so_local.astimezone(tz)
        self._state.cur_exit_utc = (
            so_local + timedelta(minutes=self.exit_offset_minutes)
        ).astimezone(tz)
        self._state.cur_entry_utc = (
            sc_local - timedelta(minutes=self.entry_offset_minutes)
        ).astimezone(tz)

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        self._maybe_roll_day(bar.ts)
        s = self._state

        cur_qty = ctx.position(self.symbol)

        # Exit window: from today's exit trigger up to (but not including)
        # today's OWN entry trigger a few hours later. Using today's entry
        # time as the upper bound -- instead of a fixed few-minute window
        # after cur_exit_utc -- fixes a real bug: the old window was
        # `[open, open + exit_offset)`, which is an EMPTY interval
        # whenever exit_offset_minutes=0 (open == exit), so the exit
        # never fired at all and the position rode forever. Bounding by
        # cur_entry_utc instead is never empty (entry is always well
        # after open) at ANY exit_offset, including zero, and it still
        # excludes the afternoon (a position opened for TONIGHT's cycle
        # must not be immediately flattened by THIS MORNING's
        # already-passed exit trigger -- that was the failure mode of a
        # naive unbounded `bar.ts >= cur_exit_utc` check). exit_done_today
        # is a defensive belt-and-suspenders against firing twice.
        if (s.cur_exit_utc is not None and s.cur_entry_utc is not None
                and s.cur_exit_utc <= bar.ts < s.cur_entry_utc
                and not s.exit_done_today and cur_qty != 0 and s.target_qty != 0):
            s.target_qty = 0
            s.exit_done_today = True
            return TargetPosition(symbol=self.symbol, qty=0, tag="od_exit")

        # Entry window: at-or-after today's entry trigger, if flat, go long.
        if s.cur_entry_utc is not None and bar.ts >= s.cur_entry_utc:
            if cur_qty == 0 and s.target_qty == 0 and not s.entry_filter_rejected:
                # Consult the entry filter, if any. Uses the date in
                # US/Eastern of the entry bar.
                if self.entry_filter is not None:
                    entry_date = bar.ts.astimezone(EASTERN).date()
                    if not self.entry_filter(entry_date):
                        # Mark so we skip every subsequent bar in this cycle.
                        s.entry_filter_rejected = True
                        return None
                s.target_qty = self.qty
                return TargetPosition(symbol=self.symbol, qty=self.qty,
                                       tag="od_enter")

        return None
