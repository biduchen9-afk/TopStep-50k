"""Realized Combine pass-rate over a daily-PnL series.

Properties asserted:
  * A clean profit-target window with multi-day dilution passes.
  * A window that hits the MLL line fails as 'mll_breach' on the
    correct day.
  * A window that hits target on a single day fails consistency.
  * Sliding-window count is len(days) - window + 1 at stride=1.
  * Non-overlapping stride=window cuts windows by exactly window_days.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from topstep50k.analysis.passrate import (
    realized_pass_rate,
    simulate_combine_window,
    simulate_sequential_accounts,
)
from topstep50k.rules.topstep import combine_50k


def _daily(start_day: date, pnls: list[float]) -> list[tuple[date, Decimal]]:
    return [(start_day + timedelta(days=i), Decimal(str(round(v, 2))))
            for i, v in enumerate(pnls)]


def test_passes_when_target_hit_with_consistency():
    rules = combine_50k()
    # 6 days, +600 each = +3,600 -> hits 3000 on day 5 (1-indexed); best day
    # is 600, total is 3600, ratio 0.167 -- passes consistency.
    res = simulate_combine_window(
        _daily(date(2024, 1, 2), [600] * 6),
        rules=rules, starting_balance=Decimal("50000"),
    )
    assert res.outcome == "pass"
    assert res.days_to_outcome == 5


def test_consistency_fail_when_one_day_dominates():
    rules = combine_50k()
    # +3000 on day 1, +100 on subsequent days -> best/total > 50%
    pnls = [3000] + [100] * 5
    res = simulate_combine_window(
        _daily(date(2024, 1, 2), pnls),
        rules=rules, starting_balance=Decimal("50000"),
    )
    assert res.outcome == "consistency_fail"


def test_mll_breach_on_first_red_run():
    rules = combine_50k()
    # MLL starts $2000 below balance. -$2100 on day 1 breaches.
    pnls = [-2100, 500, 500]
    res = simulate_combine_window(
        _daily(date(2024, 1, 2), pnls),
        rules=rules, starting_balance=Decimal("50000"),
    )
    assert res.outcome == "mll_breach"
    assert res.breach_day == date(2024, 1, 2)


def test_no_target_when_window_runs_out_flat():
    rules = combine_50k()
    res = simulate_combine_window(
        _daily(date(2024, 1, 2), [50] * 10),
        rules=rules, starting_balance=Decimal("50000"),
    )
    assert res.outcome == "no_target"
    assert res.days_to_outcome == -1


def test_mll_ratchets_then_kills_giveback():
    rules = combine_50k()
    # Build $2,500 over 5 days, then give it all back. After the rise,
    # the MLL anchor has trailed up and locked at starting balance, so
    # equity dipping back to 49999 breaches.
    pnls = [500] * 5 + [-1500, -1500]
    res = simulate_combine_window(
        _daily(date(2024, 1, 2), pnls),
        rules=rules, starting_balance=Decimal("50000"),
    )
    assert res.outcome == "mll_breach"


def test_sliding_window_count_at_stride_1():
    rules = combine_50k()
    pnls = _daily(date(2024, 1, 2), [100] * 50)
    rr = realized_pass_rate(
        pnls, rules=rules, starting_balance=Decimal("50000"),
        window_days=30, stride_days=1,
    )
    assert rr.n_windows == 21  # 50 - 30 + 1


def test_non_overlapping_windows():
    rules = combine_50k()
    pnls = _daily(date(2024, 1, 2), [100] * 90)
    rr = realized_pass_rate(
        pnls, rules=rules, starting_balance=Decimal("50000"),
        window_days=30, stride_days=30,
    )
    assert rr.n_windows == 3


def test_passrate_with_mixed_outcomes():
    rules = combine_50k()
    # 30 days of "good" 200 PnL followed by 30 days of "bad" -100 PnL
    pnls = _daily(date(2024, 1, 2), [200] * 30 + [-100] * 30)
    rr = realized_pass_rate(
        pnls, rules=rules, starting_balance=Decimal("50000"),
        window_days=30, stride_days=10,
    )
    # 4 windows: [0:30], [10:40], [20:50], [30:60]
    assert rr.n_windows == 4
    # 0.0 <= pass_rate <= 1.0
    assert 0.0 <= rr.pass_rate <= 1.0
    # Wilson CI sanity
    assert rr.ci_95_low <= rr.pass_rate <= rr.ci_95_high


def test_rejects_too_short_history():
    rules = combine_50k()
    pnls = _daily(date(2024, 1, 2), [50] * 10)
    with pytest.raises(ValueError):
        realized_pass_rate(pnls, rules=rules,
                           starting_balance=Decimal("50000"),
                           window_days=30)


def test_post_target_mll_breach_does_not_override_pass():
    """Target hit on day 5; massive loss on day 7 must NOT turn outcome to
    mll_breach. In a real Combine the trader stops on day 5."""
    rules = combine_50k()
    # Days 1-5: +600 each = +3,000 -> target hit on day 5.
    # Day 6: big loss that would breach MLL if trading continued.
    pnls = [600] * 5 + [-5000]
    res = simulate_combine_window(
        _daily(date(2024, 1, 2), pnls),
        rules=rules, starting_balance=Decimal("50000"),
    )
    assert res.outcome == "pass", (
        f"Got {res.outcome!r} — post-target day must not override the pass"
    )
    assert res.days_to_outcome == 5


def test_post_target_days_excluded_from_consistency():
    """Target hit on day 5 with clean consistency; a huge day on day 6
    must NOT cause a consistency_fail since the Combine ended on day 5."""
    rules = combine_50k()
    # Days 1-5: even +600/day -> target on day 5, best_day/total = 20%.
    # Day 6: +50,000 (would push best_day ratio > 50% of inflated total).
    pnls = [600] * 5 + [50000]
    res = simulate_combine_window(
        _daily(date(2024, 1, 2), pnls),
        rules=rules, starting_balance=Decimal("50000"),
    )
    assert res.outcome == "pass", (
        f"Got {res.outcome!r} — post-target day must not contaminate consistency"
    )


# ---- simulate_sequential_accounts ------------------------------------------


def _daily_dict(start_day: date, pnls: list[float]) -> dict[date, Decimal]:
    return {start_day + timedelta(days=i): Decimal(str(round(v, 2)))
            for i, v in enumerate(pnls)}


def test_sequential_accounts_never_overlap():
    """A new account starts the day AFTER the previous one resolves --
    never while a prior account is still 'open'."""
    rules = combine_50k()
    # Acct 1: +1000/day x3 -> passes on day 3 (idx 2).
    # Acct 2 should start day 4 (idx 3): -2100 -> breaches immediately.
    # Acct 3 starts day 5 (idx 4): drifts flat, never resolves.
    pnls = [1000, 1000, 1000, -2100] + [10] * 6
    daily = _daily_dict(date(2025, 1, 1), pnls)
    summary = simulate_sequential_accounts(daily, rules=rules,
                                            starting_balance=Decimal("50000"))
    assert summary.n_accounts == 3
    a1, a2, a3 = summary.accounts
    assert a1.outcome == "pass" and a1.end_day == date(2025, 1, 3)
    assert a2.outcome == "mll_breach"
    assert a2.start_day == a1.end_day + timedelta(days=1)  # no overlap
    assert a3.outcome == "no_target"
    assert a3.start_day == a2.end_day + timedelta(days=1)  # no overlap
    assert summary.pass_rate == pytest.approx(1 / 3)


def test_sequential_accounts_checkpoint_pass_rate():
    """checkpoint_pass_rate is conditional: of accounts that ever reached
    the checkpoint profit, what fraction actually went on to pass?"""
    rules = combine_50k()
    # Acct 1: climbs to +1800 (above the $1500 checkpoint) then gives it
    # all back to a breach -- reached the checkpoint but did NOT pass.
    pnls_1 = [900, 900, -3900]  # peak +1800, then -2100 net after breach check
    # Acct 2: climbs straight to +3000 -- reached checkpoint AND passed.
    pnls_2 = [1500, 1500]
    # Acct 3: never gets anywhere near the checkpoint.
    pnls_3 = [50] * 5
    daily = _daily_dict(date(2025, 1, 1), pnls_1 + pnls_2 + pnls_3)
    summary = simulate_sequential_accounts(daily, rules=rules,
                                            starting_balance=Decimal("50000"),
                                            checkpoint=Decimal("1500"))
    outcomes = [(a.outcome, a.reached_checkpoint) for a in summary.accounts]
    assert ("mll_breach", True) in outcomes
    assert ("pass", True) in outcomes
    assert any(not a.reached_checkpoint for a in summary.accounts)
    # Of the 2 accounts that reached the checkpoint, only 1 passed.
    assert summary.checkpoint_pass_rate == pytest.approx(0.5)


def test_sequential_accounts_empty_daily_pnl():
    rules = combine_50k()
    summary = simulate_sequential_accounts({}, rules=rules,
                                            starting_balance=Decimal("50000"))
    assert summary.n_accounts == 0
    assert summary.pass_rate == 0.0
    assert summary.checkpoint_pass_rate == 0.0
