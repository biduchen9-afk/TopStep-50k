"""Probabilistic / Deflated Sharpe and PBO sanity tests.

Properties we assert:

  * PSR is monotone in N (more data -> higher confidence for the same
    sample Sharpe). [ref:bailey_psr]
  * PSR -> 1 as N -> infinity when the empirical Sharpe is positive.
  * DSR deflated benchmark > 0 when there are multiple trials.
  * PBO of pure noise (no skill anywhere) is ~ 0.5 +/- bound.
  * PBO of a matrix where the same strategy always wins is small (no
    overfit signal because the in-sample winner is also OOS winner).
"""

from __future__ import annotations

import numpy as np
import pytest

from topstep50k.analysis.dsr import (
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
    probability_of_backtest_overfit,
)


def _make_returns(n, mu, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return list(rng.normal(mu, sigma, size=n))


def test_psr_monotone_in_n():
    # Fix mu/sigma so sample Sharpe is roughly constant; PSR should grow.
    small = _make_returns(40, 0.001, 0.01, seed=0)
    big = _make_returns(400, 0.001, 0.01, seed=0)
    p_small = probabilistic_sharpe(small).psr
    p_big = probabilistic_sharpe(big).psr
    assert p_big > p_small


def test_psr_high_for_strong_signal():
    r = _make_returns(1000, 0.002, 0.005, seed=1)  # Sharpe annual ~ 6
    res = probabilistic_sharpe(r)
    assert res.psr > 0.99


def test_psr_low_for_negative_sharpe_against_zero_bench():
    r = _make_returns(1000, -0.002, 0.005, seed=2)
    res = probabilistic_sharpe(r)
    assert res.psr < 0.01


def test_expected_max_sharpe_grows_with_n_trials():
    e2 = expected_max_sharpe(2, 0.1)
    e20 = expected_max_sharpe(20, 0.1)
    e200 = expected_max_sharpe(200, 0.1)
    assert e2 < e20 < e200


def test_dsr_deflates_when_many_trials():
    # Winner has positive Sharpe; with 100 trials and a meaningful spread
    # the deflated DSR should be MUCH smaller than the undeflated PSR.
    winner = _make_returns(252, 0.0006, 0.01, seed=3)
    trial_sharpes = [0.6, 0.55, 0.5, 0.45, 0.4] + list(
        np.random.default_rng(4).normal(0.0, 0.3, 95)
    )
    res = deflated_sharpe(winner, all_trial_sharpes_annual=trial_sharpes)
    # benchmark should be positive when we have spread
    assert res.expected_max_sharpe > 0
    # ...and DSR should be lower than the raw PSR (which has bench=0)
    raw = probabilistic_sharpe(winner).psr
    assert res.dsr <= raw + 1e-9


def test_pbo_noise_around_half():
    rng = np.random.default_rng(7)
    T, N = 320, 20
    mat = rng.normal(0.0, 0.01, size=(T, N))
    pbo = probability_of_backtest_overfit(mat, s=16, rng=rng)
    # With 100% noise, PBO has high variance but mean ~0.5; allow [0.3, 0.7]
    assert 0.3 <= pbo <= 0.7


def test_pbo_persistent_winner_is_low():
    rng = np.random.default_rng(8)
    T, N = 320, 20
    mat = rng.normal(0.0, 0.01, size=(T, N))
    # Inject a strategy with a real edge into column 0
    mat[:, 0] += 0.003
    pbo = probability_of_backtest_overfit(mat, s=16, rng=rng)
    assert pbo < 0.3, f"PBO={pbo} should be low when col0 dominates IS+OOS"


def test_psr_rejects_too_few_periods():
    with pytest.raises(ValueError):
        probabilistic_sharpe([0.1, 0.1, 0.1])
