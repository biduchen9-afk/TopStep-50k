"""Regime detection.

Currently exposes a Gaussian Hidden Markov Model wrapper that respects
the engine's no-look-ahead contract: training only sees the train fold;
online inference uses a forward-filter (no smoothing past `now`).
"""

from topstep50k.regime.hmm import (
    HMMRegimeFilter,
    HMMRegimeResult,
    RegimeAssignment,
    fit_hmm,
)

__all__ = ["HMMRegimeFilter", "HMMRegimeResult", "RegimeAssignment", "fit_hmm"]
