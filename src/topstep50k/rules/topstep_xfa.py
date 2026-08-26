"""Topstep Express Funded Account (XFA) rule encoding.

The Combine rules in `topstep.py` cover the EVALUATION phase only. This
module covers what happens AFTER you pass: the funded account's payout
eligibility, payout caps, profit split, the Scaling Plan contract cap,
and the trailing-drawdown mechanic across a payout event.

Sourced from Topstep's help-center articles and third-party 2026
aggregators (help.topstep.com is blocked from this sandbox's network
policy -- see the verification note in docs/rules_sources.md, which
this module's numbers were cross-checked against on 2026-08-26). Two
numbers here carry real financial risk if wrong:

  1. The payout caps below are the POST-APRIL-28-2026 numbers for a
     NEW Combine purchase. Accounts purchased before that date keep
     their OLD (higher) caps on rebill and on Reset Credit -- this
     module does not know which cohort a given account is in; pass
     `payout_cap_standard` / `payout_cap_consistency` explicitly if
     you're on a pre-April-28 account.
  2. The post-payout drawdown reset (`PostPayoutDrawdown` below) is
     modeled CONSERVATIVELY from a third-party worked example, not
     Topstep's own wording -- see the class docstring. Verify this
     directly with Topstep before making a real payout decision based
     on it; getting it wrong in the optimistic direction risks the
     live account.

Design notes (mirrors topstep.py):
* Rules are pure functions / small state machines over an account-state
  snapshot. They never read clocks, files, or globals.
* The trailing MLL itself (pre-first-payout) is IDENTICAL in mechanics
  to the Combine's `TrailingMLL` -- see `PostPayoutDrawdown` below,
  which mirrors that behavior before the first payout and then
  diverges for the post-payout re-anchoring (a distinct enough state
  machine that it's implemented standalone rather than subclassed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Mapping


@dataclass(frozen=True)
class ScalingStep:
    """One rung of the Scaling Plan: at `profit_threshold` cumulative
    funded-account profit, the contract cap becomes `max_contracts`."""
    profit_threshold: Decimal
    max_contracts: int


@dataclass(frozen=True)
class XFARules:
    """Parameters for a single Express Funded Account size.

    Two independent payout paths exist; a trader qualifies for whichever
    they meet first, and the caps differ:

      Standard    : >= min_winning_days days with net PnL >= min_winning_day_pnl
                    (need not be consecutive). Cap: payout_cap_standard.
      Consistency : >= min_consistency_days trading days AND
                    (largest single winning day / total net profit) <= max_consistency_pct.
                    Cap: payout_cap_consistency.

    Either way, the actual payout amount is also capped at
    `payout_cap_pct_of_balance` of current account balance -- whichever
    of the two caps is lower governs.
    """

    starting_balance: Decimal
    mll_distance: Decimal  # same $ distance as the Combine this XFA came from
    profit_split_trader: Decimal  # e.g. 0.90 = trader keeps 90%

    min_winning_days: int
    min_winning_day_pnl: Decimal
    payout_cap_standard: Decimal

    min_consistency_days: int
    max_consistency_pct: Decimal
    payout_cap_consistency: Decimal

    payout_cap_pct_of_balance: Decimal

    scaling_plan: tuple[ScalingStep, ...] = ()

    # ----- payout eligibility ------------------------------------------

    def winning_days(self, daily_pnl: Mapping[date, Decimal]) -> int:
        return sum(1 for v in daily_pnl.values() if v >= self.min_winning_day_pnl)

    def is_standard_eligible(self, daily_pnl: Mapping[date, Decimal]) -> bool:
        return self.winning_days(daily_pnl) >= self.min_winning_days

    def consistency_pct(self, daily_pnl: Mapping[date, Decimal]) -> Decimal | None:
        """Largest single-day net profit / total net profit. None if
        total profit <= 0 (undefined / not meaningful)."""
        total = sum(daily_pnl.values(), start=Decimal(0))
        if total <= 0:
            return None
        best = max((v for v in daily_pnl.values() if v > 0), default=Decimal(0))
        return best / total

    def is_consistency_eligible(self, daily_pnl: Mapping[date, Decimal]) -> bool:
        n_trading_days = sum(1 for v in daily_pnl.values() if v != 0)
        if n_trading_days < self.min_consistency_days:
            return False
        pct = self.consistency_pct(daily_pnl)
        return pct is not None and pct <= self.max_consistency_pct

    def eligible_paths(self, daily_pnl: Mapping[date, Decimal]) -> list[str]:
        paths = []
        if self.is_standard_eligible(daily_pnl):
            paths.append("standard")
        if self.is_consistency_eligible(daily_pnl):
            paths.append("consistency")
        return paths

    # ----- payout sizing -------------------------------------------------

    def max_payout(self, path: str, current_balance: Decimal) -> Decimal:
        """The largest single payout request allowed on `path` given the
        account's current balance -- min(path cap, 50% of balance)."""
        cap = self.payout_cap_standard if path == "standard" else self.payout_cap_consistency
        pct_cap = current_balance * self.payout_cap_pct_of_balance
        return min(cap, pct_cap)

    def trader_take(self, payout_amount: Decimal) -> Decimal:
        return payout_amount * self.profit_split_trader

    # ----- scaling plan ---------------------------------------------------

    def max_contracts(self, cumulative_funded_profit: Decimal) -> int:
        """Contract cap for the current cumulative profit level. Falls
        back to the lowest rung's cap if no scaling_plan is configured."""
        if not self.scaling_plan:
            return 0
        cap = self.scaling_plan[0].max_contracts
        for step in sorted(self.scaling_plan, key=lambda s: s.profit_threshold):
            if cumulative_funded_profit >= step.profit_threshold:
                cap = step.max_contracts
        return cap


def xfa_50k() -> XFARules:
    """Topstep $50K Express Funded Account, post-April-28-2026 caps.

    Verify against a pre-April-28 account before relying on the caps --
    those are grandfathered at the old (higher) values.
    """
    return XFARules(
        starting_balance=Decimal("50000"),
        mll_distance=Decimal("2000"),
        profit_split_trader=Decimal("0.90"),
        min_winning_days=5,
        min_winning_day_pnl=Decimal("150"),
        payout_cap_standard=Decimal("2000"),
        min_consistency_days=3,
        max_consistency_pct=Decimal("0.40"),
        payout_cap_consistency=Decimal("3000"),
        payout_cap_pct_of_balance=Decimal("0.50"),
        # Scaling Plan thresholds: not independently re-verified this
        # session (carried from docs/rules_sources.md's prior note,
        # "2 -> 3 -> 5 contracts at $1,500 / $2,000 profit thresholds");
        # treat as approximate until sourced directly.
        scaling_plan=(
            ScalingStep(Decimal("0"), 2),
            ScalingStep(Decimal("1500"), 3),
            ScalingStep(Decimal("2000"), 5),
        ),
    )


# ----- post-payout drawdown state machine -----------------------------------


@dataclass
class PostPayoutDrawdown:
    """Trailing-drawdown state across the funded account's life, including
    what happens to the cushion after a payout is withdrawn.

    CONSERVATIVE MODEL -- verify before trusting for real risk decisions.
    Sourced from a third-party worked example (not Topstep's own wording):
    "balance $6,000, $2,000 trailing floor [floor=$4,000]; request a
    $2,000 payout; balance drops to $4,000 and MLL resets to $0" -- i.e.
    the cushion between balance and floor collapses to zero at the
    instant of a payout, rather than the floor simply re-basing $2,000
    below the new (lower) balance.

    Before the first payout this behaves exactly like `TrailingMLL`
    (including the one-time lock at breakeven once equity climbs
    `distance` above `starting_balance` -- the normal Combine/XFA
    de-risking event). A payout re-anchors the floor at the post-payout
    balance with ZERO cushion and, from that point on, trails
    indefinitely with the same `distance` and does NOT lock again --
    the one-time lock-to-breakeven already happened relative to the
    account's true starting capital (in practice, before a trader can
    even qualify for a payout, the floor has usually already locked:
    winning-days/consistency eligibility requires real profit, and the
    floor locks the moment cumulative EOD profit reaches `distance`).
    Re-applying that same lock rule relative to each new post-payout
    base would freeze the floor the first time equity revisits the
    payout level, which has no support in the source material and
    would make the account unrunnable after a second payout -- so this
    implementation deliberately does not do that. Flagged in
    docs/rules_sources.md for anyone who can get a definitive answer
    from Topstep directly.
    """

    starting_balance: Decimal
    distance: Decimal
    anchor: Decimal = field(init=False)
    locked: bool = field(default=False, init=False)
    total_paid_out: Decimal = field(default=Decimal("0"), init=False)
    n_payouts: int = field(default=0, init=False)
    _post_payout: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.anchor = self.starting_balance

    @property
    def line(self) -> Decimal:
        return self.anchor - self.distance

    def update_end_of_day(self, eod_balance: Decimal) -> None:
        if self.locked:
            return
        candidate = max(self.anchor, eod_balance)
        if not self._post_payout and candidate - self.distance >= self.starting_balance:
            # Pre-payout: same one-time lock-at-breakeven as TrailingMLL.
            self.anchor = self.starting_balance + self.distance
            self.locked = True
        else:
            self.anchor = candidate

    def apply_payout(self, payout_amount: Decimal, post_payout_balance: Decimal) -> None:
        """Re-anchor the trail at the post-payout balance with zero
        cushion; from here it trails indefinitely (same distance, no
        further locking). Call this the same "day" the payout clears,
        before that day's update_end_of_day.

        `payout_amount` is the dollar amount withdrawn (for bookkeeping
        only); `post_payout_balance` is the account balance immediately
        after the withdrawal (what the new floor re-anchors to).
        """
        self.anchor = post_payout_balance + self.distance  # line == post_payout_balance
        self.locked = False
        self._post_payout = True
        self.total_paid_out += payout_amount
        self.n_payouts += 1
