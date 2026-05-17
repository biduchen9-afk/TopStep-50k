"""TopStep Trading Combine rule encoding.

The numbers and mechanics here come from Topstep's published help-center
articles. Citations live in docs/rules_sources.md so any update to those
articles forces a code-level diff.

Design notes
------------
* Rules are encoded as PURE FUNCTIONS over an account-state snapshot plus
  the event being evaluated. They never read clocks, files, or globals.
* Every breach returns a structured RuleBreach with the deterministic
  inputs that produced it, so the audit log can reproduce the decision.
* The Max Loss Limit (MLL) trails the highest END-OF-DAY balance and
  locks at the starting balance once it gets there. Intraday equity does
  not move the trail — confirmed by Topstep's MLL help-center article.
* The Daily Loss Limit (DLL) is a SOFT breach: hitting it auto-flattens
  and bars new entries for the session, but does not fail the Combine.
* The consistency rule (best-day <= 50% of total cycle PnL) is a
  payout/eval-completion gate, not an intra-session liquidation trigger.
* The scaling-plan max-position cap on the $50K Combine is 5 standard
  contracts or 50 micros at any one time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Mapping


class BreachType(str, Enum):
    """Hard = ends the Combine. Soft = blocks the rest of the session only."""

    HARD = "hard"
    SOFT = "soft"


class BreachReason(str, Enum):
    MAX_LOSS_LIMIT = "max_loss_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    CONSISTENCY = "consistency"  # evaluated at end-of-combine only


@dataclass(frozen=True)
class RuleBreach:
    reason: BreachReason
    breach_type: BreachType
    observed: Decimal
    limit: Decimal
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.reason.value}[{self.breach_type.value}] observed={self.observed} limit={self.limit} {self.detail}"


@dataclass(frozen=True)
class TopstepRules:
    """Parameters for a single Trading Combine account size.

    All monetary values are in account currency (USD) and stored as
    Decimal so the rule arithmetic is exact -- never float comparisons
    against limit values.

    Contract-cap semantics (verified 2026-05): on the $50K Combine the
    cap is ACCOUNT-LEVEL total exposure, not per-symbol. Topstep counts
    each standard contract as `micro_to_standard_ratio` micros toward
    the cap (default 10:1). So the legal maxima are 5 standard, or 50
    micros, or any mix totalling <= 50 micro-equivalents. See
    docs/rules_sources.md for sources.
    """

    starting_balance: Decimal
    profit_target: Decimal
    max_loss_limit_distance: Decimal  # distance below the trail anchor
    daily_loss_limit: Decimal | None  # None = not enforced (Combine default)
    max_contracts_standard: int  # account-wide cap, standard-equivalent
    max_contracts_micro: int     # account-wide cap, micro-equivalent
    consistency_max_best_day_pct: Decimal  # e.g. Decimal("0.50") = 50%
    micro_to_standard_ratio: int = 10
    # Symbols recognised as "micro" for the contract cap. The standard cap
    # applies to everything else.
    micro_symbols: frozenset[str] = field(
        default_factory=lambda: frozenset({"MES", "MNQ", "MYM", "M2K", "MGC", "MCL"})
    )

    def position_cap(self, symbol: str) -> int:
        """Per-symbol convenience cap. Returns the maximum number of
        contracts of `symbol` you could hold if it were the ONLY open
        position. For the actual account-level check use
        `check_position_size`.
        """
        return self.max_contracts_micro if symbol in self.micro_symbols else self.max_contracts_standard

    def micro_equivalents(self, positions: Mapping[str, int]) -> int:
        """Sum of |qty| across positions, scaling standard contracts up
        by `micro_to_standard_ratio`. This is the quantity TopStep
        compares to the cap."""
        total = 0
        for sym, qty in positions.items():
            size = abs(int(qty))
            if size == 0:
                continue
            total += size if sym in self.micro_symbols else size * self.micro_to_standard_ratio
        return total

    # ----- core evaluators -------------------------------------------------

    def check_position_size(
        self, projected_positions: Mapping[str, int]
    ) -> RuleBreach | None:
        """Account-level total-exposure check.

        `projected_positions` is what the open-position dict would look
        like AFTER the order under consideration fills (engine's job to
        construct). Cap is `max_contracts_micro` in micro-equivalent
        units.
        """
        total = self.micro_equivalents(projected_positions)
        cap = self.max_contracts_micro
        if total > cap:
            return RuleBreach(
                reason=BreachReason.POSITION_SIZE_EXCEEDED,
                breach_type=BreachType.HARD,
                observed=Decimal(total),
                limit=Decimal(cap),
                detail=f"positions={dict(projected_positions)}",
            )
        return None

    def check_max_loss(
        self, current_equity: Decimal, mll_anchor: Decimal
    ) -> RuleBreach | None:
        """MLL line = mll_anchor - max_loss_limit_distance.

        mll_anchor starts at starting_balance and only moves UP, anchored
        to the highest end-of-day balance, and locks once
        anchor - distance >= starting_balance.

        Caller is responsible for maintaining the anchor (see TrailingMLL).
        """
        line = mll_anchor - self.max_loss_limit_distance
        if current_equity <= line:
            return RuleBreach(
                reason=BreachReason.MAX_LOSS_LIMIT,
                breach_type=BreachType.HARD,
                observed=current_equity,
                limit=line,
                detail=f"anchor={mll_anchor}",
            )
        return None

    def check_daily_loss(
        self, current_equity: Decimal, sod_equity: Decimal
    ) -> RuleBreach | None:
        """Soft breach when intraday PnL <= -daily_loss_limit.

        Returns None when `daily_loss_limit` is None: the TopStep Combine
        does not impose a DLL (verified 2026-05; see docs/rules_sources.md).
        The field is retained for platforms / account types that DO
        enforce one (Express Funded with a personal DLL configured, or
        platform-side defaults on NinjaTrader/Tradovate/Quantower).
        """
        if self.daily_loss_limit is None:
            return None
        intraday_pnl = current_equity - sod_equity
        if intraday_pnl <= -self.daily_loss_limit:
            return RuleBreach(
                reason=BreachReason.DAILY_LOSS_LIMIT,
                breach_type=BreachType.SOFT,
                observed=intraday_pnl,
                limit=-self.daily_loss_limit,
                detail=f"sod_equity={sod_equity}",
            )
        return None

    def check_consistency(
        self, daily_pnl: Mapping[date, Decimal]
    ) -> RuleBreach | None:
        """End-of-combine evaluation. Pass requires
            max(positive daily pnl) / sum(daily pnl) <= consistency_max_best_day_pct
        with sum > 0.
        """
        if not daily_pnl:
            return None
        total = sum(daily_pnl.values(), start=Decimal(0))
        if total <= 0:
            return None  # consistency only matters if you actually made money
        best = max((v for v in daily_pnl.values() if v > 0), default=Decimal(0))
        ratio = best / total
        if ratio > self.consistency_max_best_day_pct:
            return RuleBreach(
                reason=BreachReason.CONSISTENCY,
                breach_type=BreachType.HARD,
                observed=ratio,
                limit=self.consistency_max_best_day_pct,
                detail=f"best_day={best} total={total}",
            )
        return None

    def reached_profit_target(self, eod_equity: Decimal) -> bool:
        return eod_equity - self.starting_balance >= self.profit_target


# ----- trailing-anchor state machine ---------------------------------------


@dataclass
class TrailingMLL:
    """End-of-day trailing Maximum Loss Limit anchor.

    Mechanics (per Topstep MLL article):
      * Starts at starting_balance.
      * After each trading day, anchor = max(anchor, end_of_day_balance).
      * Once anchor - distance >= starting_balance, the anchor LOCKS at
        the value that produces a line == starting_balance.
      * Anchor never decreases.
    """

    starting_balance: Decimal
    distance: Decimal
    anchor: Decimal = field(init=False)
    locked: bool = False

    def __post_init__(self) -> None:
        self.anchor = self.starting_balance

    @property
    def line(self) -> Decimal:
        return self.anchor - self.distance

    def update_end_of_day(self, eod_balance: Decimal) -> None:
        if self.locked:
            return
        candidate = max(self.anchor, eod_balance)
        # Will the new anchor push the line >= starting_balance?
        if candidate - self.distance >= self.starting_balance:
            self.anchor = self.starting_balance + self.distance
            self.locked = True
        else:
            self.anchor = candidate


# ----- presets -------------------------------------------------------------


def combine_50k() -> TopstepRules:
    """Topstep $50K Trading Combine parameters.

    The Combine itself does NOT enforce a Daily Loss Limit (verified
    2026-05; see docs/rules_sources.md). The DLL field is left as None
    so the engine skips that check. Pass any positive Decimal here if
    you want to model a personal DLL or a platform that enforces one.
    """
    return TopstepRules(
        starting_balance=Decimal("50000"),
        profit_target=Decimal("3000"),
        max_loss_limit_distance=Decimal("2000"),
        daily_loss_limit=None,
        max_contracts_standard=5,
        max_contracts_micro=50,
        consistency_max_best_day_pct=Decimal("0.50"),
    )
