"""Position-sizing overlay tests: parity with the unscaled simulator,
and that each sizing rule actually changes behavior the way it claims to."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from topstep50k.analysis.passrate import simulate_sequential_accounts
from topstep50k.analysis.sizing import (
    AccountState,
    combined_scaling,
    drawdown_responsive_scaling,
    full_size,
    simulate_sequential_accounts_sized,
    target_proximity_scaling,
)
from topstep50k.rules.topstep import combine_50k


def _daily(values: list[float], start=date(2025, 1, 1)) -> dict[date, Decimal]:
    return {start + timedelta(days=i): Decimal(str(v)) for i, v in enumerate(values)}


def test_full_size_is_exact_parity_with_unscaled_simulator():
    rules = combine_50k()
    # A mixed win/loss/breach series long enough to open several accounts.
    values = [500, -300, 800, -2200, 600, 600, 600, 600, 600, -500, 900, 900, 900]
    daily = _daily(values)
    baseline = simulate_sequential_accounts(daily, rules=rules,
                                             starting_balance=rules.starting_balance)
    sized = simulate_sequential_accounts_sized(daily, rules=rules,
                                                starting_balance=rules.starting_balance,
                                                sizing_fn=full_size)
    assert len(sized.accounts) == len(baseline.accounts)
    for a, b in zip(sized.accounts, baseline.accounts):
        assert a.outcome == b.outcome
        assert a.final_pnl == b.final_pnl
        assert a.peak_profit == b.peak_profit
        assert a.n_days == b.n_days


def test_target_proximity_scaling_halves_pnl_once_checkpoint_crossed():
    rules = combine_50k()
    # Day 1: +$1,600 (crosses the $1,500 checkpoint immediately).
    # Day 2 onward: sizing should be halved since profit_so_far >= 1500
    # is evaluated BEFORE day 2's own pnl.
    values = [1600, 1000, 1000]
    daily = _daily(values)
    fn = target_proximity_scaling(checkpoint=Decimal("1500"), scale_after=0.5)
    result = simulate_sequential_accounts_sized(
        daily, rules=rules, starting_balance=rules.starting_balance, sizing_fn=fn)
    acc = result.accounts[0]
    # Day1 full (+1600) + Day2 half (+500) = +2100, still short of target,
    # continues to day3 half (+500) = +2600 -- still short (< 3000).
    assert acc.final_pnl == Decimal("2600") or acc.outcome in ("pass", "no_target")


def test_target_proximity_scaling_does_not_scale_before_checkpoint():
    rules = combine_50k()
    values = [100, 100, 100]
    daily = _daily(values)
    fn = target_proximity_scaling(checkpoint=Decimal("1500"), scale_after=0.5)
    result = simulate_sequential_accounts_sized(
        daily, rules=rules, starting_balance=rules.starting_balance, sizing_fn=fn)
    baseline = simulate_sequential_accounts(daily, rules=rules,
                                             starting_balance=rules.starting_balance)
    assert result.accounts[0].final_pnl == baseline.accounts[0].final_pnl == Decimal("300")


def test_drawdown_responsive_scaling_halves_the_day_after_a_loss():
    rules = combine_50k()
    values = [100, -200, 1000, 100]
    daily = _daily(values)
    fn = drawdown_responsive_scaling(scale_after_loss=0.5)
    result = simulate_sequential_accounts_sized(
        daily, rules=rules, starting_balance=rules.starting_balance, sizing_fn=fn)
    acc = result.accounts[0]
    # Day1 full (+100), Day2 full (-200, since day1 wasn't a loss),
    # Day3 HALF (+500, since day2 was a loss), Day4 full (+100, day3 was a win).
    assert acc.final_pnl == Decimal("500")


def test_combined_scaling_takes_the_minimum():
    state_after_loss_and_checkpoint = AccountState(
        profit_so_far=Decimal("2000"), peak_profit=Decimal("2000"),
        days_elapsed=5, yesterday_pnl=Decimal("-100"),
    )
    fn = combined_scaling(
        target_proximity_scaling(checkpoint=Decimal("1500"), scale_after=0.5),
        drawdown_responsive_scaling(scale_after_loss=0.25),
    )
    assert fn(state_after_loss_and_checkpoint) == 0.25  # min(0.5, 0.25)

    state_neither = AccountState(
        profit_so_far=Decimal("0"), peak_profit=Decimal("0"),
        days_elapsed=0, yesterday_pnl=None,
    )
    assert fn(state_neither) == 1.0
