"""Regime detection.

Currently exposes a Gaussian Hidden Markov Model wrapper that respects
the engine's no-look-ahead contract: training only sees the train fold;
online inference uses a forward-filter (no smoothing past `now`).
"""

from topstep50k.regime.conditioners import (
    DailySessionStats,
    meanrev_low_vol_gate,
    orb_expansion_gate,
    overnight_drift_post_selloff_gate,
    per_day_session_stats,
    rolling_vol,
    trailing_median,
)
from topstep50k.regime.features import DailyFeatures, daily_features_from_bars
from topstep50k.regime.hmm import (
    HMMRegimeFilter,
    HMMRegimeResult,
    RegimeAssignment,
    fit_hmm,
)

__all__ = [
    "DailyFeatures", "daily_features_from_bars",
    "HMMRegimeFilter", "HMMRegimeResult", "RegimeAssignment", "fit_hmm",
    "DailySessionStats", "per_day_session_stats",
    "rolling_vol", "trailing_median",
    "orb_expansion_gate", "meanrev_low_vol_gate",
    "overnight_drift_post_selloff_gate",
]
