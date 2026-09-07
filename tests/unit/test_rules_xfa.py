"""Pinned tests for the Express Funded Account (XFA) and rebill-lifecycle
rule semantics -- see docs/rules_sources.md for citations and the
explicit "verify before trusting" caveats on the post-payout drawdown
model and the April-28-2026 payout caps.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from topstep50k.rules.rebill import (
    CREDIT_EXPIRY_POLICY_START,
    MAX_REDEMPTIONS_PER_DAY,
    REBILL_INTERVAL_DAYS,
    RebillLifecycle,
)
from topstep50k.rules.topstep_xfa import PostPayoutDrawdown, ScalingStep, XFARules, xfa_50k


@pytest.fixture
def xfa() -> XFARules:
    return xfa_50k()


# ---- payout eligibility ----------------------------------------------------


def test_standard_path_requires_five_150_days(xfa):
    daily = {date(2026, 1, i): Decimal("150") for i in range(1, 5)}  # only 4
    assert xfa.winning_days(daily) == 4
    assert not xfa.is_standard_eligible(daily)
    daily[date(2026, 1, 5)] = Decimal("150")
    assert xfa.is_standard_eligible(daily)


def test_standard_path_day_below_threshold_does_not_count(xfa):
    daily = {date(2026, 1, i): Decimal("149.99") for i in range(1, 6)}
    assert xfa.winning_days(daily) == 0
    assert not xfa.is_standard_eligible(daily)


def test_consistency_path_needs_min_days_and_pct(xfa):
    # 2 trading days only -- fails the >=3 day requirement even though
    # the pct would pass.
    daily = {date(2026, 1, 1): Decimal("500"), date(2026, 1, 2): Decimal("500")}
    assert not xfa.is_consistency_eligible(daily)

    # 3 days, evenly split -> pct = 1/3 = 33.3% <= 40%, eligible.
    daily = {
        date(2026, 1, 1): Decimal("500"),
        date(2026, 1, 2): Decimal("500"),
        date(2026, 1, 3): Decimal("500"),
    }
    assert float(xfa.consistency_pct(daily)) == pytest.approx(1 / 3)
    assert xfa.is_consistency_eligible(daily)

    # 3 days, one day dominates -> pct > 40%, not eligible.
    daily = {
        date(2026, 1, 1): Decimal("1000"),
        date(2026, 1, 2): Decimal("10"),
        date(2026, 1, 3): Decimal("10"),
    }
    assert xfa.consistency_pct(daily) > Decimal("0.40")
    assert not xfa.is_consistency_eligible(daily)


def test_consistency_pct_undefined_when_not_profitable(xfa):
    daily = {date(2026, 1, 1): Decimal("-100"), date(2026, 1, 2): Decimal("50")}
    assert xfa.consistency_pct(daily) is None
    assert not xfa.is_consistency_eligible(daily)


# ---- payout sizing -----------------------------------------------------


def test_payout_capped_by_lower_of_path_cap_and_pct_of_balance(xfa):
    # Balance large enough that the flat cap governs.
    assert xfa.max_payout("standard", Decimal("100000")) == Decimal("2000")
    assert xfa.max_payout("consistency", Decimal("100000")) == Decimal("3000")
    # Balance small enough that 50% governs instead.
    assert xfa.max_payout("standard", Decimal("2000")) == Decimal("1000")


def test_trader_take_applies_profit_split(xfa):
    assert xfa.trader_take(Decimal("2000")) == Decimal("1800.00")


def test_scaling_plan_thresholds(xfa):
    assert xfa.max_contracts(Decimal("0")) == 2
    assert xfa.max_contracts(Decimal("1499")) == 2
    assert xfa.max_contracts(Decimal("1500")) == 3
    assert xfa.max_contracts(Decimal("1999")) == 3
    assert xfa.max_contracts(Decimal("2000")) == 5


# ---- post-payout drawdown --------------------------------------------------


def test_post_payout_drawdown_zero_cushion_at_moment_of_payout():
    pp = PostPayoutDrawdown(starting_balance=Decimal("50000"), distance=Decimal("2000"))
    pp.update_end_of_day(Decimal("56000"))  # locks: anchor=52000, line=50000
    assert pp.locked

    pp.apply_payout(payout_amount=Decimal("2000"), post_payout_balance=Decimal("54000"))
    assert pp.line == Decimal("54000")  # zero cushion: floor == post-payout balance
    assert pp.n_payouts == 1
    assert pp.total_paid_out == Decimal("2000")

    # A trivial loss the same "day" as the payout would breach.
    assert Decimal("53999") <= pp.line


def test_post_payout_drawdown_floor_pinned_until_distance_regained():
    pp = PostPayoutDrawdown(starting_balance=Decimal("50000"), distance=Decimal("2000"))
    pp.update_end_of_day(Decimal("52000"))
    pp.apply_payout(payout_amount=Decimal("1000"), post_payout_balance=Decimal("51000"))
    assert pp.line == Decimal("51000")  # zero cushion at the moment of payout

    # Same as the ORIGINAL trail being pinned at starting_balance - distance
    # until equity climbs a full `distance` above the base: the floor stays
    # frozen at the post-payout balance (every dollar of gain is pure
    # cushion) until equity regains the full $2,000 above it.
    pp.update_end_of_day(Decimal("52500"))
    assert pp.line == Decimal("51000")  # still pinned -- 52500 < 51000+2000

    # Once equity climbs the full distance above the post-payout balance,
    # normal trailing resumes -- and does NOT lock again (unlike the
    # pre-payout trail, which locks the first time this happens).
    pp.update_end_of_day(Decimal("53500"))
    assert pp.line == Decimal("51500")  # 53500 - 2000, trail resumed
    assert not pp.locked

    # Keeps trailing further -- confirms it really doesn't lock.
    pp.update_end_of_day(Decimal("55000"))
    assert pp.line == Decimal("53000")  # 55000 - 2000
    assert not pp.locked


def test_post_payout_drawdown_preserve_cushion_leaves_the_floor_untouched():
    # Same setup as the zero-cushion test above, but with the alternate
    # reading requested: a partial payout should leave real cushion
    # behind, not reset to zero regardless of amount withdrawn.
    pp = PostPayoutDrawdown(starting_balance=Decimal("50000"), distance=Decimal("2000"))
    pp.update_end_of_day(Decimal("51500"))  # not locked yet (peak 51500 < 52000)
    line_before = pp.line
    assert line_before == Decimal("49500")  # 51500 - 2000

    # Withdraw only $500 of the available cushion, preserve_cushion=True.
    pp.apply_payout(payout_amount=Decimal("500"), post_payout_balance=Decimal("51000"),
                     preserve_cushion=True)
    assert pp.line == line_before  # floor didn't move at all
    assert pp.n_payouts == 1
    assert pp.total_paid_out == Decimal("500")
    # Real cushion left behind: balance(51000) - floor(49500) = 1500,
    # NOT zero -- this is the whole point of preserve_cushion.
    assert Decimal("51000") - pp.line == Decimal("1500")


def test_post_payout_drawdown_preserve_cushion_keeps_the_lock():
    pp = PostPayoutDrawdown(starting_balance=Decimal("50000"), distance=Decimal("2000"))
    pp.update_end_of_day(Decimal("56000"))  # locks: anchor=52000, line=50000
    assert pp.locked

    pp.apply_payout(payout_amount=Decimal("1000"), post_payout_balance=Decimal("55000"),
                     preserve_cushion=True)
    assert pp.locked  # unlike preserve_cushion=False, the lock survives
    assert pp.line == Decimal("50000")  # unchanged


# ---- rebill lifecycle -------------------------------------------------


def test_rebill_adds_credit_and_advances_date():
    rb = RebillLifecycle(account_size="50K", rebill_date=date(2026, 1, 1))
    credit = rb.rebill(date(2026, 1, 1))
    assert rb.rebill_date == date(2026, 1, 1) + timedelta(days=REBILL_INTERVAL_DAYS)
    assert credit in rb.credits
    assert rb.n_available_credits(date(2026, 1, 1)) == 1


def test_rebill_before_policy_start_never_expires():
    rb = RebillLifecycle(account_size="50K")
    credit = rb.rebill(CREDIT_EXPIRY_POLICY_START - timedelta(days=1))
    assert credit.expires_on is None
    assert rb.n_available_credits(date(2099, 1, 1)) == 1  # still available far in the future


def test_rebill_on_or_after_policy_start_expires_in_one_year():
    rb = RebillLifecycle(account_size="50K")
    credit = rb.rebill(CREDIT_EXPIRY_POLICY_START)
    assert credit.expires_on == CREDIT_EXPIRY_POLICY_START + timedelta(days=365)
    assert rb.n_available_credits(credit.expires_on - timedelta(days=1)) == 1
    assert rb.n_available_credits(credit.expires_on) == 0  # expired


def test_redeem_reset_credit_consumes_the_soonest_expiring_one():
    rb = RebillLifecycle(account_size="50K")
    early = rb.rebill(date(2026, 1, 1))       # expires 2027-01-01
    later = rb.rebill(date(2026, 1, 5))       # expires 2027-01-05
    used = rb.redeem_reset_credit(date(2026, 1, 10))
    assert used is early
    assert early.used and not later.used


def test_redeem_reset_credit_pushes_rebill_date():
    rb = RebillLifecycle(account_size="50K", rebill_date=date(2026, 1, 1))
    rb.rebill(date(2026, 1, 1))
    rb.redeem_reset_credit(date(2026, 1, 15))
    assert rb.rebill_date == date(2026, 1, 15) + timedelta(days=REBILL_INTERVAL_DAYS)


def test_redeem_reset_credit_raises_when_none_available():
    rb = RebillLifecycle(account_size="50K")
    with pytest.raises(ValueError):
        rb.redeem_reset_credit(date(2026, 1, 1))


def test_redeem_reset_credit_daily_cap_enforced():
    rb = RebillLifecycle(account_size="50K")
    for _ in range(MAX_REDEMPTIONS_PER_DAY):
        rb.rebill(date(2026, 1, 1))
    for _ in range(MAX_REDEMPTIONS_PER_DAY):
        rb.redeem_reset_credit(date(2026, 1, 1))
    rb.rebill(date(2026, 1, 1))
    with pytest.raises(ValueError):
        rb.redeem_reset_credit(date(2026, 1, 1))
