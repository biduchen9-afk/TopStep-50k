"""Pairwise stream-correlation reporting tests."""

from __future__ import annotations

import numpy as np

from topstep50k.analysis.correlation import stream_correlation_report


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
