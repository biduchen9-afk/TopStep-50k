"""Macro Event Overnight Drift — long overnight before FOMC and NFP releases.

Academic rationale
──────────────────
Lucca & Moench (2015, J. Finance): +49bp average S&P 500 return in the 24h
  BEFORE each FOMC announcement. Mechanism: pre-announcement uncertainty
  premium; informed order flow; options dealers hedging.

Savor & Wilson (2013, J. Finance): "Money Illusion and the Stock Market"
  More broadly, the equity premium is concentrated on macro announcement
  days (FOMC, NFP, CPI). Announcement days earn ~49bp vs ~0 on non-days.
  Pre-announcement overnight drift is a risk premium for bearing macro
  resolution risk.

  Pre-NFP: Employment Situation release (first Friday of month, 08:30 ET).
  The evening before is analogous to FOMC eve — holders bear announcement
  risk → expect a risk premium → overnight long before NFP.

  Combined FOMC eve + NFP eve: ~8 + 12 = ~20 events/year — clearing the
  minimum frequency gate. Both effects are documented in the same academic
  literature and driven by the same uncertainty-resolution mechanism.

Pre-committed parameters
────────────────────────
  entry_time_local : 15:55 ET on event eve (near RTH close)
  exit_time_local  : 09:35 ET on announcement day
  direction        : LONG ONLY (papers document long-only drift)
  stop             : None (overnight hold, consistent with documented effect)
  qty              : 1

FOMC announcement days: known months in advance from Fed calendar.
NFP release days: known in advance from BLS calendar.
Zero look-ahead risk — all dates from hardcoded calendar.py schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo

from topstep50k.data.calendar import (
    fomc_announcement_dates,
    is_fomc_day,
    is_fomc_eve,
    nfp_dates,
)
from topstep50k.engine.types import Bar
from topstep50k.strategy.base import StrategyContext, TargetPosition


EASTERN = ZoneInfo("America/New_York")

_ENTRY_TIME = dtime(15, 55)
_EXIT_TIME  = dtime(9, 35)

# NFP eves: calendar day before each NFP release date
_NFP_EVE_DATES: frozenset[date] = frozenset(
    d - timedelta(days=1) for d in nfp_dates()
)


def is_nfp_eve(d: date) -> bool:
    return d in _NFP_EVE_DATES


def is_nfp_day(d: date) -> bool:
    return d in nfp_dates()


def is_macro_eve(d: date) -> bool:
    """True if d is FOMC eve or NFP eve (enter overnight trade)."""
    return is_fomc_eve(d) or is_nfp_eve(d)


def is_macro_day(d: date) -> bool:
    """True if d is FOMC announcement day or NFP release day (exit trade)."""
    return is_fomc_day(d) or is_nfp_day(d)


@dataclass
class _MOState:
    last_seen_date: date | None = None
    in_trade: bool = False
    done_today: bool = False
    target_qty: int = 0


@dataclass
class MacroOvernight:
    """Long overnight before FOMC and NFP announcements.

    Parameters
    ----------
    symbol : bar-feed symbol.
    qty    : contracts per trade (default 1).
    """

    symbol: str
    qty: int = 1

    _state: _MOState = field(init=False, default_factory=_MOState, repr=False)

    def _maybe_roll_day(self, bar: Bar) -> None:
        et = bar.ts.astimezone(EASTERN)
        d = et.date()
        s = self._state
        if s.last_seen_date == d:
            return
        s.last_seen_date = d
        s.done_today = False
        # Keep in_trade across day boundary

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> TargetPosition | None:
        self._maybe_roll_day(bar)
        s = self._state
        et = bar.ts.astimezone(EASTERN)
        d = et.date()

        # ── Exit: on macro announcement day at 09:35 ET ───────────────────
        if s.in_trade and is_macro_day(d):
            if et.time() >= _EXIT_TIME and not s.done_today:
                s.in_trade = False
                s.done_today = True
                s.target_qty = 0
                return TargetPosition(symbol=self.symbol, qty=0,
                                       tag="mo_exit_announcement")

        # ── Safety flat: in trade but not on eve or announcement day ──────
        if s.in_trade and not is_macro_day(d) and not is_macro_eve(d):
            s.in_trade = False
            s.done_today = True
            s.target_qty = 0
            return TargetPosition(symbol=self.symbol, qty=0,
                                   tag="mo_safety_flat")

        # ── Entry: on macro eve at 15:55 ET ───────────────────────────────
        if is_macro_eve(d) and not s.in_trade and not s.done_today:
            if et.time() >= _ENTRY_TIME:
                s.in_trade = True
                s.done_today = True
                s.target_qty = self.qty
                return TargetPosition(symbol=self.symbol, qty=self.qty,
                                       tag="mo_enter_eve")

        return None
