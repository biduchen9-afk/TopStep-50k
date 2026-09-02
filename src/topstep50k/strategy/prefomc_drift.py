"""Pre-FOMC Drift — hold ES long from RTH close on FOMC eve to RTH open on FOMC day.

Academic rationale
──────────────────
Lucca & Moench (2015, J. Finance): "The Pre-FOMC Announcement Drift"
  Average S&P 500 excess return of +49bp in the 24h window BEFORE each FOMC
  announcement since 1994. The drift is concentrated in the overnight window
  (RTH close on FOMC eve → RTH open on FOMC announcement day).
  The effect is:
    * Statistically robust (t-stat ~5.5 over their sample)
    * Concentrated in the equity premium, not bond or currency markets
    * Robust to alternative windows (6h, 12h, 48h); peak at 24h
    * Explained by pre-announcement risk premium / uncertainty resolution

  Savor & Wilson (2013, J. Finance): Macro announcement risk premium.
  Bernile, Bhagwat & Rau (2016, J. Finance): informed trading before FOMC.
  All consistent with a genuine risk premium for holding equity risk overnight
  before an FOMC announcement.

  We use the EXACT FOMC dates hardcoded in calendar.py (from Fed schedule —
  zero look-ahead risk since dates are announced months in advance).

Pre-committed parameters
────────────────────────
  entry_time_local   : 15:55 ET on FOMC eve (5 min before RTH close)
  exit_time_local    : 09:35 ET on FOMC announcement day (5 min after open)
  stop_pct           : None (no stop — hold overnight as documented in paper)
  qty                : 1

Direction: always LONG (the paper documents a long-only effect; short is noise).

Implementation note: since we use 1-min bar data covering RTH+extended hours,
we enter at the 15:55 bar on FOMC eve and exit at the 09:35 bar on FOMC day.
The overnight gap is captured implicitly by the price difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from topstep50k.data.calendar import is_fomc_day, is_fomc_eve
from topstep50k.engine.types import Bar
from topstep50k.strategy.base import StrategyContext, TargetPosition


EASTERN = ZoneInfo("America/New_York")

_ENTRY_TIME  = time(15, 55)
_EXIT_TIME   = time(9, 35)


@dataclass
class _PFState:
    last_seen_date: date | None = None
    in_trade: bool = False        # currently holding overnight
    entry_date: date | None = None
    done_today: bool = False
    target_qty: int = 0


@dataclass
class PreFOMCDrift:
    """Long overnight hold from FOMC eve close to FOMC day open.

    Parameters
    ----------
    symbol : bar-feed symbol.
    qty    : contracts per trade (default 1).
    """

    symbol: str
    qty: int = 1

    _state: _PFState = field(init=False, default_factory=_PFState, repr=False)

    def _maybe_roll_day(self, bar: Bar) -> None:
        et = bar.ts.astimezone(EASTERN)
        d = et.date()
        s = self._state
        if s.last_seen_date == d:
            return
        # New calendar day
        s.last_seen_date = d
        s.done_today = False
        # Keep in_trade across day boundary (overnight position)

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        self._maybe_roll_day(bar)
        s = self._state
        et = bar.ts.astimezone(EASTERN)
        d = et.date()
        cur_qty = ctx.position(self.symbol)

        # ── Exit: on FOMC announcement day at 09:35 ET ────────────────────
        if s.in_trade and is_fomc_day(d):
            if et.time() >= _EXIT_TIME and not s.done_today:
                s.in_trade = False
                s.done_today = True
                s.target_qty = 0
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="pf_exit_fomc_open")

        # ── Safety: if we're in a trade and it's NOT FOMC day and NOT FOMC eve,
        #    something went wrong (e.g. holiday). Force flat.
        if s.in_trade and not is_fomc_day(d) and not is_fomc_eve(d):
            s.in_trade = False
            s.done_today = True
            s.target_qty = 0
            return TargetPosition(symbol=self.symbol, qty=0,
                                   tag="pf_safety_flat")

        # ── Entry: on FOMC eve at 15:55 ET ────────────────────────────────
        if is_fomc_eve(d) and not s.in_trade and not s.done_today:
            if et.time() >= _ENTRY_TIME:
                s.in_trade = True
                s.entry_date = d
                s.done_today = True
                s.target_qty = self.qty
                return TargetPosition(symbol=self.symbol, qty=self.qty,
                                       tag="pf_enter_eve")

        return None
