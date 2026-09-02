"""Monte Carlo pass-rate resampling sanity tests."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest

from topstep50k.analysis.montecarlo import monte_carlo_pass_rate
from topstep50k.rules.topstep import combine_50k


def _synthetic_daily_pnl(n=200, seed=0):
    rng = np.random.default_rng(seed)
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]
    # Mildly-positive-drift streaky series (autocorrelated via a
    # random-walk-of-regimes trick) so results aren't a degenerate
    # all-pass / all-fail edge case.
    regime = np.repeat(rng.choice([1, -1], size=n // 10 + 1), 10)[:n]
    values = rng.normal(60, 150, size=n) + regime * 80
    return {d: Decimal(str(round(float(v), 2))) for d, v in zip(days, values)}


def test_deterministic_with_fixed_seed():
    daily = _synthetic_daily_pnl()
    rules = combine_50k()
    r1 = monte_carlo_pass_rate(daily, rules=rules, starting_balance=rules.starting_balance,
                                n_sims=50, seed=7)
    r2 = monte_carlo_pass_rate(daily, rules=rules, starting_balance=rules.starting_balance,
                                n_sims=50, seed=7)
    assert r1.pass_rates.tolist() == r2.pass_rates.tolist()


def test_result_shape_and_bounds():
    daily = _synthetic_daily_pnl()
    rules = combine_50k()
    result = monte_carlo_pass_rate(daily, rules=rules, starting_balance=rules.starting_balance,
                                    n_sims=64, block_len=10)
    assert result.n_sims == 64
    assert len(result.pass_rates) == 64
    assert 0.0 <= result.mean_pass_rate <= 1.0
    assert 0.0 <= result.median_pass_rate <= 1.0
    assert result.p05 <= result.median_pass_rate <= result.p95
    assert 0.0 <= result.prob_above_50pct <= 1.0
    assert result.mean_n_accounts > 0


def test_spread_shrinks_with_larger_block_on_a_streaky_series():
    # Not a strict monotonicity guarantee in general (block bootstrap
    # variance behavior depends on the true dependence structure), but on
    # a strongly regime-streaky series, a block_len that matches the
    # regime length should look less erratic than block_len=1 (pure iid
    # shuffle, which destroys the streaks that made the original path's
    # pass rate what it was).
    daily = _synthetic_daily_pnl(n=300, seed=3)
    rules = combine_50k()
    matched = monte_carlo_pass_rate(daily, rules=rules, starting_balance=rules.starting_balance,
                                     n_sims=80, block_len=10, seed=1)
    iid = monte_carlo_pass_rate(daily, rules=rules, starting_balance=rules.starting_balance,
                                 n_sims=80, block_len=1, seed=1)
    assert np.std(matched.pass_rates) <= np.std(iid.pass_rates) + 0.15  # loose sanity bound


def test_too_few_days_raises():
    rules = combine_50k()
    tiny = {date(2025, 1, 1): Decimal("100")}
    with pytest.raises(ValueError):
        monte_carlo_pass_rate(tiny, rules=rules, starting_balance=rules.starting_balance,
                               n_sims=5, block_len=10)
