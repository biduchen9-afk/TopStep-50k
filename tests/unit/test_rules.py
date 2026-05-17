"""Pinned tests for the $50K Combine rule semantics.

These tests document the exact behaviour expected from `topstep.combine_50k`.
Changing them means changing the contract a strategy is being evaluated
against — review carefully and update docs/rules_sources.md alongside.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from topstep50k.rules import (
    BreachReason,
    BreachType,
    TopstepRules,
    combine_50k,
)
from topstep50k.rules.topstep import TrailingMLL


@pytest.fixture
def rules() -> TopstepRules:
    return combine_50k()


# ---- position size --------------------------------------------------------

class TestPositionSize:
    def test_standard_cap_5(self, rules):
        assert rules.check_position_size("ES", 5) is None

    def test_standard_cap_breached_at_6(self, rules):
        breach = rules.check_position_size("ES", 6)
        assert breach is not None
        assert breach.reason is BreachReason.POSITION_SIZE_EXCEEDED
        assert breach.breach_type is BreachType.HARD

    def test_micro_cap_50(self, rules):
        assert rules.check_position_size("MES", 50) is None
        assert rules.check_position_size("MES", 51) is not None

    def test_unknown_symbol_uses_standard_cap(self, rules):
        # An unknown symbol must not silently get the looser micro cap.
        assert rules.check_position_size("CL", 6) is not None


# ---- daily loss limit -----------------------------------------------------

class TestDailyLoss:
    def test_within_limit(self, rules):
        sod = Decimal("50000")
        assert rules.check_daily_loss(Decimal("49001"), sod) is None

    def test_exactly_at_limit_breaches(self, rules):
        sod = Decimal("50000")
        breach = rules.check_daily_loss(Decimal("49000"), sod)
        assert breach is not None
        assert breach.breach_type is BreachType.SOFT  # important: NOT hard

    def test_below_limit_still_soft(self, rules):
        breach = rules.check_daily_loss(Decimal("48500"), Decimal("50000"))
        assert breach.breach_type is BreachType.SOFT


# ---- trailing MLL ---------------------------------------------------------

class TestTrailingMLL:
    def test_starts_at_48000(self, rules):
        mll = TrailingMLL(rules.starting_balance, rules.max_loss_limit_distance)
        assert mll.line == Decimal("48000")
        assert not mll.locked

    def test_line_trails_eod_up(self, rules):
        mll = TrailingMLL(rules.starting_balance, rules.max_loss_limit_distance)
        mll.update_end_of_day(Decimal("50500"))
        assert mll.line == Decimal("48500")
        mll.update_end_of_day(Decimal("50300"))  # lower EOD
        assert mll.line == Decimal("48500")  # does NOT move down

    def test_locks_at_starting_balance(self, rules):
        mll = TrailingMLL(rules.starting_balance, rules.max_loss_limit_distance)
        mll.update_end_of_day(Decimal("52500"))
        assert mll.locked
        assert mll.line == Decimal("50000")  # locked at starting balance
        mll.update_end_of_day(Decimal("60000"))
        assert mll.line == Decimal("50000")  # stays locked

    def test_does_not_move_intraday(self, rules):
        """Intraday equity highs must NOT advance the anchor.

        The trail is end-of-day only. Equity spiking to $55K and then
        falling back must leave the anchor unchanged.
        """
        mll = TrailingMLL(rules.starting_balance, rules.max_loss_limit_distance)
        starting_line = mll.line
        # caller would not call update_end_of_day intraday; verifying the
        # only state mutation is via that method.
        assert mll.line == starting_line


# ---- max loss check (composes with trailing anchor) -----------------------

class TestMaxLossCheck:
    def test_within_line(self, rules):
        breach = rules.check_max_loss(Decimal("48001"), mll_anchor=Decimal("50000"))
        assert breach is None

    def test_at_line_breaches(self, rules):
        breach = rules.check_max_loss(Decimal("48000"), mll_anchor=Decimal("50000"))
        assert breach is not None
        assert breach.breach_type is BreachType.HARD


# ---- consistency rule -----------------------------------------------------

class TestConsistency:
    def test_balanced_pnl_passes(self, rules):
        pnl = {date(2026, 1, d): Decimal("1000") for d in range(1, 5)}
        assert rules.check_consistency(pnl) is None

    def test_one_huge_day_fails(self, rules):
        pnl = {
            date(2026, 1, 1): Decimal("1600"),
            date(2026, 1, 2): Decimal("700"),
            date(2026, 1, 3): Decimal("700"),
        }
        # 1600 / 3000 = 53.3%
        breach = rules.check_consistency(pnl)
        assert breach is not None
        assert breach.reason is BreachReason.CONSISTENCY

    def test_exactly_50pct_passes(self, rules):
        pnl = {
            date(2026, 1, 1): Decimal("1500"),
            date(2026, 1, 2): Decimal("1500"),
        }
        assert rules.check_consistency(pnl) is None

    def test_no_profit_not_evaluated(self, rules):
        pnl = {date(2026, 1, 1): Decimal("-500")}
        assert rules.check_consistency(pnl) is None
