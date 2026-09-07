"""Rolling correlation: causality, alignment, look-ahead enforcement.
Plus stream_correlation_report(): static batch correlation for
ensemble-construction-time diversification checks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

from topstep50k.analysis.correlation import (
    RollingCorrelation,
    aligned_returns_from_closes,
    stream_correlation_report,
)
from topstep50k.engine.clock import Clock, LookAheadError


def _ts(minutes: int) -> datetime:
    return datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minutes)


def test_aligned_returns_filters_to_common_timestamps():
    closes = {
        "A": [(_ts(0), Decimal("100")), (_ts(1), Decimal("101")), (_ts(2), Decimal("102"))],
        "B": [(_ts(1), Decimal("200")), (_ts(2), Decimal("198")), (_ts(3), Decimal("199"))],
    }
    common_ts, rets = aligned_returns_from_closes(closes)
    # Only ts=1 and ts=2 are present in both; first return needs prior, so only ts=2 emits.
    assert common_ts == [_ts(2)]
    assert len(rets["A"]) == 1
    assert len(rets["B"]) == 1


def test_perfect_correlation_recovered():
    rc = RollingCorrelation(symbols=["A", "B"], window=10)
    rng = np.random.default_rng(0)
    base_returns = rng.normal(0.0, 0.01, size=50)
    snap = None
    for i, r in enumerate(base_returns):
        snap = rc.push(_ts(i), {"A": r, "B": 2 * r + 0.001})  # perfectly correlated, scaled
    assert snap is not None
    assert abs(snap.pair("A", "B") - 1.0) < 1e-9


def test_negative_correlation_recovered():
    rc = RollingCorrelation(symbols=["A", "B"], window=10)
    rng = np.random.default_rng(1)
    for i, r in enumerate(rng.normal(0.0, 0.01, size=50)):
        snap = rc.push(_ts(i), {"A": r, "B": -3 * r})
    assert snap is not None
    assert abs(snap.pair("A", "B") + 1.0) < 1e-9


def test_window_warmup_returns_none():
    rc = RollingCorrelation(symbols=["A", "B"], window=5)
    for i in range(4):
        assert rc.push(_ts(i), {"A": 0.001 * i, "B": -0.001 * i}) is None
    assert rc.push(_ts(4), {"A": 0.005, "B": -0.005}) is not None


def test_clock_guard_blocks_future_push():
    clk = Clock(_ts(10))
    rc = RollingCorrelation(symbols=["A", "B"], window=4, clock=clk)
    # Pushing a timestamp newer than the clock must fail.
    with pytest.raises(LookAheadError):
        rc.push(_ts(11), {"A": 0.001, "B": 0.001})


def test_avg_offdiag_for_3_symbols():
    rc = RollingCorrelation(symbols=["A", "B", "C"], window=6)
    rng = np.random.default_rng(3)
    snap = None
    for i in range(12):
        a = rng.normal(0.0, 0.01)
        b = 2 * a
        c = -a
        snap = rc.push(_ts(i), {"A": a, "B": b, "C": c})
    assert snap is not None
    # corrs: A-B = +1, A-C = -1, B-C = -1; avg offdiag = (1 + -1 + -1) / 3 * 2 / (3*2) = -1/3
    # The avg_offdiag computes sum of off-diagonal entries / (k*(k-1)) -> symmetric counts each pair twice
    # so result = 2*(1 + -1 + -1)/(3*2) = -1/3
    assert abs(snap.avg_offdiag() - (-1.0 / 3.0)) < 1e-9


# ── stream_correlation_report(): static batch ensemble-construction check ──

def test_identical_streams_flagged_at_correlation_one():
    a = np.array([10.0, -5.0, 20.0, -15.0, 5.0, 0.0, 30.0])
    report = stream_correlation_report({"A": a, "B": a.copy()}, threshold=0.6)
    assert len(report.high_corr_pairs) == 1
    ki, kj, c = report.high_corr_pairs[0]
    assert {ki, kj} == {"A", "B"}
    assert c > 0.99


def test_orthogonal_nonoverlapping_streams_not_flagged():
    # A trades on even days only, B trades on odd days only -- zero overlap,
    # like OD (overnight) vs. a day-session strategy.
    a = np.array([10.0, 0.0, -5.0, 0.0, 20.0, 0.0, -15.0, 0.0])
    b = np.array([0.0, 8.0, 0.0, -3.0, 0.0, 12.0, 0.0, -6.0])
    report = stream_correlation_report({"A": a, "B": b}, threshold=0.6)
    assert report.high_corr_pairs == []


def test_zero_variance_stream_does_not_crash():
    a = np.zeros(10)
    b = np.array([1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 2.0, -2.0, 1.0, -1.0])
    report = stream_correlation_report({"flat": a, "active": b}, threshold=0.6)
    assert report.matrix.shape == (2, 2)
    assert report.high_corr_pairs == []


def test_threshold_controls_flagging():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = 0.5 * a + rng.normal(size=200) * 0.3  # moderately correlated
    loose = stream_correlation_report({"A": a, "B": b}, threshold=0.9)
    tight = stream_correlation_report({"A": a, "B": b}, threshold=0.3)
    assert loose.high_corr_pairs == []
    assert len(tight.high_corr_pairs) == 1
