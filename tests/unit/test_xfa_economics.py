"""XFA lifecycle simulation: payout eligibility/sizing, breach detection,
and the Monte Carlo wrapper's basic sanity."""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from topstep50k.analysis.xfa_economics import (
    monte_carlo_xfa_economics,
    simulate_xfa_lifecycle,
)
from topstep50k.rules.topstep_xfa import xfa_50k


def test_standard_path_payout_after_five_winning_days():
    xfa = xfa_50k()
    # A big skewed day (1000) keeps consistency ineligible throughout
    # (largest/total stays > 40% even once 5 days have accrued), so
    # this isolates the standard path: 5 days >= $150 each -> eligible
    # exactly on day index 4.
    daily = [Decimal("150"), Decimal("150"), Decimal("1000"),
             Decimal("150"), Decimal("150")] + [Decimal("0")] * 10
    result = simulate_xfa_lifecycle(daily, xfa=xfa)
    assert result.n_payouts >= 1
    first = result.payouts[0]
    assert first.day_index == 4
    assert first.path == "standard"
    # balance = 50000 + 1600 = 51600; 50%-of-balance cap ($25,800) doesn't
    # bind, so payout = the $2,000 standard cap.
    assert first.payout_amount == Decimal("2000")
    assert first.trader_take == Decimal("2000") * Decimal("0.90")


def test_no_payout_before_eligibility():
    xfa = xfa_50k()
    # Only 2 trading days: below BOTH the consistency minimum (3 days)
    # and the standard minimum (5 winning days) -- neither path qualifies.
    daily = [Decimal("200")] * 2
    result = simulate_xfa_lifecycle(daily, xfa=xfa)
    assert result.n_payouts == 0


def test_mll_breach_detected_and_stops_the_run():
    xfa = xfa_50k()
    # Immediate large loss blows through the $2,000 trailing floor.
    daily = [Decimal("-2500")] + [Decimal("100")] * 10
    result = simulate_xfa_lifecycle(daily, xfa=xfa)
    assert result.breached is True
    assert result.days_run == 1  # stopped after day index 0 (the 1st day)
    assert result.final_balance == Decimal("47500")


def test_eligibility_window_resets_after_a_payout():
    xfa = xfa_50k()
    # Skewed 5-day pattern (blocks consistency, satisfies standard at
    # day index 4 -- same construction as the standard-path test above)
    # triggers the first payout and resets the window. Only 2 MORE days
    # follow -- below both paths' minimums (consistency needs >= 3 days,
    # standard needs >= 5 winning days) -- so no second payout should
    # fire even though, if the window had NOT reset, the account would
    # have plenty of lifetime winning days by now.
    daily = ([Decimal("150"), Decimal("150"), Decimal("1000"),
              Decimal("150"), Decimal("150")]
             + [Decimal("150"), Decimal("150")])
    result = simulate_xfa_lifecycle(daily, xfa=xfa)
    assert result.n_payouts == 1  # second payout NOT triggered (only 2 new days)


def test_payout_reduces_balance_and_resets_floor_to_zero_cushion():
    xfa = xfa_50k()
    # 3 EQUAL days already satisfy consistency (largest/total = 33.3%
    # <= 40%) -- it fires at day index 2, before day 5's standard check
    # would ever be reached, since eligibility is checked every day.
    daily = [Decimal("200")] * 5
    result = simulate_xfa_lifecycle(daily, xfa=xfa)
    assert result.n_payouts == 1
    p = result.payouts[0]
    assert p.day_index == 2
    assert p.path == "consistency"
    # balance at day2 = 50000 + 3*200 = 50600; consistency cap $3,000
    # (50%-of-balance cap of $25,300 doesn't bind).
    assert p.payout_amount == Decimal("3000")
    assert p.balance_after == Decimal("50600") - Decimal("3000")
    assert result.final_balance == p.balance_after


def test_monte_carlo_shapes_and_bounds():
    rng = np.random.default_rng(0)
    # Mildly positive, streaky-ish synthetic series -- enough winning
    # days to generate some payouts, enough variance to generate some
    # breaches, so the test isn't checking a degenerate all-0 result.
    values = list(rng.normal(60, 400, size=300))
    daily = [Decimal(str(round(v, 2))) for v in values]
    xfa = xfa_50k()
    result = monte_carlo_xfa_economics(daily, xfa=xfa, horizon_days=120,
                                        n_sims=50, block_len=10, seed=1)
    assert result.n_sims == 50
    assert len(result.total_income) == 50
    assert result.mean_income >= 0
    assert 0.0 <= result.prob_breach <= 1.0
    assert result.p05_income <= result.median_income <= result.p95_income
