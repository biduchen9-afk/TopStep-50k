"""Regime detection.

Currently exposes a Gaussian Hidden Markov Model wrapper that respects
the engine's no-look-ahead contract: training only sees the train fold;
online inference uses a forward-filter (no smoothing past `now`).
"""

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
]
