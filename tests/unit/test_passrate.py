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
