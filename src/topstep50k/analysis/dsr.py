"""Probabilistic / Deflated Sharpe Ratio and Probability of Backtest Overfitting.

These are the inferential layer on top of raw Sharpe. They answer:

  * Is this Sharpe distinguishable from zero given how few samples and
    how non-normal the return distribution actually is? -> PSR
    [ref:bailey_psr]
  * How much should I deflate this Sharpe for the fact that I tried N
    strategy variants and picked the best? -> DSR [ref:bailey_dsr]
  * If I shuffle the in-sample / out-of-sample assignments, how often
    does the in-sample winner also win out-of-sample? Probability of
    Backtest Overfitting. [ref:bailey_pbo]

Everything here is closed-form except PBO, which is a small Monte Carlo
over combinatorial splits of the matrix of strategy-by-period returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class PSRResult:
    sharpe: float          # observed annualised Sharpe
    psr: float             # P(true Sharpe > benchmark)
    n_periods: int
    skewness: float
    kurtosis_excess: float
    benchmark_sharpe: float


@dataclass(frozen=True)
class DSRResult:
    sharpe: float          # observed annualised Sharpe of the SELECTED strategy
    dsr: float             # Probability that the true Sharpe > 0 after deflation
    n_trials: int          # number of strategy variants tried (>=1)
    sharpe_trial_std: float
    expected_max_sharpe: float
    n_periods: int


def _ann_factor(periods_per_year: int) -> float:
    return math.sqrt(periods_per_year)


def _moments(returns: Sequence[float]) -> tuple[float, float, float, float]:
    """Sample mean, stdev (ddof=1), skewness, excess kurtosis."""
    n = len(returns)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    mu = sum(returns) / n
    var = sum((r - mu) ** 2 for r in returns) / (n - 1)
    sigma = math.sqrt(var)
    if sigma < 1e-15:
        return mu, 0.0, 0.0, 0.0
    m3 = sum((r - mu) ** 3 for r in returns) / n
    m4 = sum((r - mu) ** 4 for r in returns) / n
    sk = m3 / (sigma ** 3)
    ku = m4 / (sigma ** 4) - 3.0
    return mu, sigma, sk, ku


def probabilistic_sharpe(
    returns: Sequence[float],
    *,
    benchmark_sharpe_annual: float = 0.0,
    periods_per_year: int = 252,
) -> PSRResult:
    """PSR closed form ([ref:bailey_psr] eq. 6).

    Returns P(true annualised Sharpe > benchmark) given the observed
    sample Sharpe, sample skewness, and sample kurtosis.
    """
    n = len(returns)
    if n < 4:
        raise ValueError("PSR needs at least 4 periods")
    mu, sigma, sk, ku = _moments(returns)
    if sigma < 1e-15:
        return PSRResult(0.0, 0.0 if benchmark_sharpe_annual > 0 else 1.0,
                         n, sk, ku, benchmark_sharpe_annual)
    sharpe_period = mu / sigma
    sharpe_annual = sharpe_period * _ann_factor(periods_per_year)
    bench_period = benchmark_sharpe_annual / _ann_factor(periods_per_year)
    # PSR(SR*) = Phi( (SR - SR*) * sqrt(n-1) / sqrt(1 - sk*SR + (ku/4)*SR^2) )
    denom_sq = 1.0 - sk * sharpe_period + (ku / 4.0) * sharpe_period * sharpe_period
    if denom_sq <= 0:
        # Degenerate; PSR is ill-defined. Be conservative.
        return PSRResult(sharpe_annual, 0.0, n, sk, ku, benchmark_sharpe_annual)
    z = (sharpe_period - bench_period) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return PSRResult(
        sharpe=sharpe_annual,
        psr=float(norm.cdf(z)),
        n_periods=n,
        skewness=sk,
        kurtosis_excess=ku,
        benchmark_sharpe=benchmark_sharpe_annual,
    )


def expected_max_sharpe(n_trials: int, sharpe_trial_std_period: float) -> float:
    """Expected maximum of N i.i.d. standard normals, scaled by the
    cross-trial Sharpe stdev. Closed form from [ref:bailey_dsr]:

        E[max_i SR_i] ~= sigma_SR * ((1-gamma) * Phi^{-1}(1 - 1/N)
                                     + gamma * Phi^{-1}(1 - 1/(N*e)))

    where gamma = Euler-Mascheroni ~= 0.5772.
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649015329
    e = math.e
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * e))
    return sharpe_trial_std_period * ((1.0 - gamma) * a + gamma * b)


def deflated_sharpe(
    selected_returns: Sequence[float],
    *,
    all_trial_sharpes_annual: Sequence[float],
    periods_per_year: int = 252,
) -> DSRResult:
    """DSR ([ref:bailey_dsr]): deflates the selected strategy's PSR by the
    expected max Sharpe under the null that no strategy has skill.

    `all_trial_sharpes_annual` should contain ALL strategy variants tried
    in selection (including the winner). The function computes the
    benchmark Sharpe E[max] and PSR against that benchmark.
    """
    n_trials = len(all_trial_sharpes_annual)
    if n_trials < 1:
        raise ValueError("DSR needs at least one trial")
    if not selected_returns:
        raise ValueError("DSR needs the winner's return series")
    sharpes_period = np.array(all_trial_sharpes_annual) / _ann_factor(periods_per_year)
    if n_trials > 1:
        s_std = float(np.std(sharpes_period, ddof=1))
    else:
        s_std = 0.0
    bench_period = expected_max_sharpe(n_trials, s_std)
    bench_annual = bench_period * _ann_factor(periods_per_year)
    psr = probabilistic_sharpe(
        selected_returns,
        benchmark_sharpe_annual=bench_annual,
        periods_per_year=periods_per_year,
    )
    return DSRResult(
        sharpe=psr.sharpe,
        dsr=psr.psr,
        n_trials=n_trials,
        sharpe_trial_std=s_std * _ann_factor(periods_per_year),
        expected_max_sharpe=bench_annual,
        n_periods=len(selected_returns),
    )


def probability_of_backtest_overfit(
    returns_matrix: np.ndarray,
    *,
    s: int = 16,
    rng: np.random.Generator | None = None,
    metric: str = "sharpe",
) -> float:
    """PBO via combinatorially symmetric cross-validation [ref:bailey_pbo].

    Parameters
    ----------
    returns_matrix : shape (T, N). T = time periods, N = strategy variants.
        Element [t, i] is the (period) return of strategy i in period t.
    s : even integer >= 4, number of equal-sized slices the time axis is
        split into. Recommended 16. We then form C(s, s/2) splits where
        half the slices are "IS" and half "OOS". For each split we pick
        the IS winner and ask whether its OOS rank is below median.
    metric : 'sharpe' or 'mean' to rank strategies. Sharpe is the default.
    rng : if returns_matrix is too big to enumerate C(s, s/2) splits in
        full, an rng (default seeded) is used to sample 1000 splits.

    Returns
    -------
    pbo : float in [0, 1]. The probability that the IS-best strategy is
        below the median strategy in OOS. PBO=0.5 is "no overfitting
        signal at all", PBO->1 means selecting the IS-best is actively
        anti-predictive of OOS.
    """
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2D (T x N)")
    if s < 4 or s % 2 != 0:
        raise ValueError("s must be an even integer >= 4")
    T, N = returns_matrix.shape
    if N < 2:
        raise ValueError("PBO needs at least 2 strategy variants")
    if T < s:
        raise ValueError(f"PBO needs at least s={s} periods, have T={T}")

    # Trim to multiple of s
    T_use = (T // s) * s
    M = returns_matrix[:T_use]
    slices = np.array_split(M, s, axis=0)

    def _metric(arr: np.ndarray) -> np.ndarray:
        # arr shape (rows, N). Return shape (N,)
        if metric == "mean":
            return arr.mean(axis=0)
        # sharpe
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(sd > 1e-15, mu / sd, 0.0)
        return sr

    indices = list(range(s))
    half = s // 2
    splits = list(combinations(indices, half))
    if len(splits) > 2000:
        if rng is None:
            rng = np.random.default_rng(0)
        # subsample
        choice = rng.choice(len(splits), size=1000, replace=False)
        splits = [splits[i] for i in choice]

    logits: list[float] = []
    for is_indices in splits:
        oos_indices = [i for i in indices if i not in is_indices]
        is_data = np.vstack([slices[i] for i in is_indices])
        oos_data = np.vstack([slices[i] for i in oos_indices])
        is_metric = _metric(is_data)
        oos_metric = _metric(oos_data)
        winner = int(np.argmax(is_metric))
        # omega = OOS relative rank of the IS winner, on (0, 1).
        # Convention from [ref:bailey_pbo]: high omega means the IS-best
        # is ALSO good OOS (no overfit). So we rank ascending — worst=1,
        # best=N — and divide by N+1.
        ranks = oos_metric.argsort().argsort() + 1
        omega = ranks[winner] / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1.0 - omega)))
    logits_arr = np.array(logits)
    # PBO = P(logit < 0) = P(omega < 0.5) = P(winner is below median OOS)
    return float((logits_arr < 0).mean())
