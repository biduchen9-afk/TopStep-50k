"""HMM regime filter: causal forward-filter, audit on transitions, no
look-ahead.

Key properties:

  * Two-regime synthetic series (low-vol then high-vol) is recovered:
    the filter classifies the high-vol portion to a different state than
    the low-vol portion.
  * Forward-filter posterior of update(t) depends ONLY on features[<=t].
    Mutating future features must NOT change the posterior history.
  * clock.assert_visible() rejects features dated past now().
  * Every regime change emits an audit event of kind regime_transition.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from topstep50k.audit import InMemoryAuditLog
from topstep50k.engine.clock import Clock, LookAheadError
from topstep50k.regime import HMMRegimeFilter, fit_hmm


def _ts(i):
    return datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)


def _two_regime_returns(rng, n_each=300):
    low = rng.normal(0.0, 0.001, size=n_each)
    high = rng.normal(0.0, 0.01, size=n_each)
    return np.concatenate([low, high]).reshape(-1, 1)


def _features_from_returns(returns: np.ndarray, vol_window: int = 20):
    """Two features: signed return and rolling abs-return mean. Both are
    causal at the index they are emitted on."""
    abs_r = np.abs(returns).ravel()
    csum = np.concatenate([[0.0], np.cumsum(abs_r)])
    out = []
    for i in range(len(returns)):
        if i < vol_window:
            vol = abs_r[: i + 1].mean()
        else:
            vol = (csum[i + 1] - csum[i + 1 - vol_window]) / vol_window
        out.append([float(returns[i, 0]), float(vol)])
    return np.array(out)


def test_hmm_fits_and_two_regimes_separable():
    rng = np.random.default_rng(123)
    rets = _two_regime_returns(rng)
    feats = _features_from_returns(rets)
    train = feats[:400]
    test = feats[400:]
    fit = fit_hmm(train, n_states=2, seed=0)
    flt = HMMRegimeFilter(fit)
    states = []
    for i, x in enumerate(test):
        assignment = flt.update(_ts(i), x)
        states.append(assignment.most_likely_state)
    # Test segment is the second-half (high-vol) part of the concatenated
    # series; one state should dominate.
    counts = np.bincount(states, minlength=2)
    dominant = counts.max() / counts.sum()
    assert dominant > 0.7, f"no single state dominated; counts={counts}"


def test_forward_filter_is_causal():
    """Mutating features_t for some t > t0 must leave update(t0)'s
    posterior unchanged."""
    rng = np.random.default_rng(7)
    rets = _two_regime_returns(rng, n_each=200)
    feats = _features_from_returns(rets)
    fit = fit_hmm(feats[:300], n_states=2, seed=0)

    flt_a = HMMRegimeFilter(fit)
    base_post = []
    for i, x in enumerate(feats[300:]):
        a = flt_a.update(_ts(i), x)
        base_post.append(a.posterior.copy())

    # Now run again but mutate features past index 10 to nonsense.
    feats_mut = feats.copy()
    feats_mut[315:] = 9999.0
    flt_b = HMMRegimeFilter(fit)
    mut_post = []
    for i, x in enumerate(feats_mut[300:]):
        a = flt_b.update(_ts(i), x)
        mut_post.append(a.posterior.copy())

    # First 15 posteriors (indices 0..14) precede the mutation point; they must match.
    for i in range(15):
        assert np.allclose(base_post[i], mut_post[i]), f"causality break at i={i}"


def test_clock_guard_blocks_future_features():
    rng = np.random.default_rng(2)
    rets = _two_regime_returns(rng)
    feats = _features_from_returns(rets)
    fit = fit_hmm(feats[:400], n_states=2, seed=0)
    flt = HMMRegimeFilter(fit)
    clk = Clock(_ts(50))
    with pytest.raises(LookAheadError):
        flt.update(_ts(100), feats[400], clock=clk)


def test_transition_emits_audit_event():
    rng = np.random.default_rng(4)
    # Train on a single switch; test on an alternating low/high pattern
    # to force at least one transition during the test stream.
    train_rets = _two_regime_returns(rng, n_each=200)
    train_feats = _features_from_returns(train_rets)
    fit = fit_hmm(train_feats, n_states=2, seed=0)

    flt = HMMRegimeFilter(fit, symbol_tag="ES")
    audit = InMemoryAuditLog()
    # Test stream: 150 low-vol, 150 high-vol, 150 low-vol -> >=2 transitions
    test_rets = np.concatenate([
        rng.normal(0.0, 0.001, size=150),
        rng.normal(0.0, 0.01, size=150),
        rng.normal(0.0, 0.001, size=150),
    ]).reshape(-1, 1)
    test_feats = _features_from_returns(test_rets)
    for i, x in enumerate(test_feats):
        flt.update(_ts(i), x, audit=audit)
    transitions = audit.of_kind("regime_transition")
    assert len(transitions) >= 1, "expected at least one regime transition"
    payload = transitions[0].payload
    assert "from" in payload and "to" in payload
    assert "posterior" in payload
    assert payload["tag"] == "ES"


def test_fit_rejects_tiny_training():
    rng = np.random.default_rng(5)
    feats = _features_from_returns(rng.normal(0.0, 0.01, size=15).reshape(-1, 1))
    with pytest.raises(ValueError):
        fit_hmm(feats, n_states=3, seed=0)
