"""EVGate + fractional Kelly: gating semantics + size math."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from topstep50k.risk import EVGate, fractional_kelly
from topstep50k.rules.topstep import combine_50k


def _oos_series(mu_daily_dollars: float, sigma_daily: float, n: int, seed: int):
    rng = np.random.default_rng(seed)
    return [Decimal(str(round(v, 2))) for v in rng.normal(mu_daily_dollars, sigma_daily, n)]


def test_fractional_kelly_basic_math():
    # mu=0.001, var=0.0004 -> full = 2.5, fraction 0.25 -> 0.625, cap=1.0 -> 0.625
    assert fractional_kelly(0.001, 0.0004, fraction=0.25) == pytest.approx(0.625, abs=1e-9)


def test_fractional_kelly_negative_mean_is_zero():
    assert fractional_kelly(-0.001, 0.0004) == 0.0


def test_fractional_kelly_cap_applies():
    assert fractional_kelly(0.01, 0.0001, fraction=0.25, cap=1.0) == 1.0


def test_evgate_rejects_short_oos_series():
    rules = combine_50k()
    gate = EVGate(rules=rules, bootstrap_draws=200)
    res = gate.evaluate(
        oos_daily_pnl=[Decimal("100"), Decimal("200")],
        all_trial_sharpes_annual=[0.5],
        starting_balance=Decimal("50000"),
    )
    assert not res.passed
    assert "too short" in res.detail.lower()


def test_evgate_blocks_when_dsr_fails():
    """If the OOS Sharpe is positive but modest, and we tried hundreds
    of variants with high cross-trial Sharpe spread, the expected-max
    benchmark exceeds the observed Sharpe and DSR collapses."""
    rules = combine_50k()
    # 60 days, modest positive drift (mu/sigma small per day)
    pnl = _oos_series(5.0, 200.0, 60, seed=2)
    # Hundreds of trial sharpes with big spread -> expected_max is large
    trial_sharpes = list(np.random.default_rng(3).normal(0.0, 1.5, size=500))
    gate = EVGate(rules=rules, bootstrap_draws=500, min_dsr=0.95)
    res = gate.evaluate(
        oos_daily_pnl=pnl,
        all_trial_sharpes_annual=trial_sharpes,
        starting_balance=Decimal("50000"),
    )
    assert not res.passed, f"unexpectedly passed: {res.detail}"
    assert "dsr=" in res.detail


def test_evgate_zero_multiplier_when_blocked():
    rules = combine_50k()
    gate = EVGate(rules=rules, bootstrap_draws=200)
    # Empty -> gate fails -> size returns multiplier 0 regardless of mu, var.
    fail = gate.evaluate(
        oos_daily_pnl=[],
        all_trial_sharpes_annual=[0.0],
        starting_balance=Decimal("50000"),
    )
    decision = gate.size(0.01, 0.0001, gate=fail)
    assert decision.multiplier == 0.0
    assert decision.fractional_kelly_used == 0.0


def test_evgate_sizes_when_passed():
    rules = combine_50k()
    gate = EVGate(rules=rules, bootstrap_draws=200, min_dsr=0.0,
                  min_pass_rate=0.0, min_pass_rate_ci_low=-1.0)
    # Bypass actual gating logic by relaxing thresholds; check the size math.
    res = gate.evaluate(
        oos_daily_pnl=_oos_series(100.0, 200.0, 60, seed=4),
        all_trial_sharpes_annual=[0.5],
        starting_balance=Decimal("50000"),
    )
    assert res.passed
    decision = gate.size(0.001, 0.0004, gate=res)
    # Same math as test_fractional_kelly_basic_math
    assert decision.multiplier == pytest.approx(0.625, abs=1e-9)
