"""Pairwise stream-correlation reporting for ensemble construction.

Direct response to a real concern: OD's diversification value came
from trading in a non-overlapping TIME WINDOW (overnight, when every
day-session strategy is flat) -- that's automatic, structural
decorrelation. Now that every candidate strategy trades within the
same RTH window, decorrelation is no longer free; it has to be
checked, not assumed. harness.py's Gate 5 already checks one stream's
correlation against existing streams during promotion, but the
ensemble-construction scripts in this project derive weights from
IS-pass30 alone, with no explicit look at how correlated the streams
actually are with each other before combining them.

This computes the full pairwise correlation matrix of daily-PnL
streams (not just one at a time) and flags any pair above a threshold,
so "these strategies shouldn't be conversing with each other" is
something the ensemble build actually verifies, not something assumed
because the strategies look mechanistically different on paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationReport:
    keys: list                      # stream identifiers, in matrix order
    matrix: np.ndarray               # keys x keys correlation matrix
    high_corr_pairs: list            # [(key_i, key_j, corr), ...] above threshold
    threshold: float
    max_pair: tuple | None           # (key_i, key_j, corr) of the single worst pair


def stream_correlation_report(
    arrays: dict,
    *,
    threshold: float = 0.6,
) -> CorrelationReport:
    """`arrays`: {key: np.ndarray} of daily PnL/returns, all the SAME
    length and aligned to the same day index (e.g. the IS-only arrays
    used for weight derivation). Correlation is computed on the raw
    daily $ series -- zero-days (strategy didn't trade) are included
    deliberately, since a strategy that's flat on days another is
    active is exactly the decorrelation we're checking for.
    """
    keys = list(arrays.keys())
    n = len(keys)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = arrays[keys[i]], arrays[keys[j]]
            if xi.std() > 0 and xj.std() > 0:
                c = float(np.corrcoef(xi, xj)[0, 1])
            else:
                c = 0.0
            mat[i, j] = mat[j, i] = c

    high_pairs = []
    worst = None
    for i in range(n):
        for j in range(i + 1, n):
            c = mat[i, j]
            if abs(c) >= threshold:
                high_pairs.append((keys[i], keys[j], c))
            if worst is None or abs(c) > abs(worst[2]):
                worst = (keys[i], keys[j], c)

    high_pairs.sort(key=lambda t: -abs(t[2]))
    return CorrelationReport(keys=keys, matrix=mat, high_corr_pairs=high_pairs,
                              threshold=threshold, max_pair=worst)


def print_correlation_report(report: CorrelationReport) -> None:
    print(f"  Pairwise correlation matrix ({len(report.keys)} streams, "
          f"flag threshold=|r|>={report.threshold}):")
    label = lambda k: f"{k[0]}/{k[1]}" if isinstance(k, tuple) else str(k)
    header = "".join(f"{label(k):>14}" for k in report.keys)
    print(f"  {'':>18}{header}")
    for i, ki in enumerate(report.keys):
        row = "".join(f"{report.matrix[i, j]:>+14.2f}" for j in range(len(report.keys)))
        print(f"  {label(ki):>18}{row}")

    if report.high_corr_pairs:
        print(f"\n  FLAGGED pairs (|r| >= {report.threshold}):")
        for ki, kj, c in report.high_corr_pairs:
            print(f"    {label(ki)} <-> {label(kj)}: r={c:+.2f}")
    else:
        print(f"\n  No pairs above |r|={report.threshold} -- no flagged "
              f"'conversing' streams.")
    if report.max_pair:
        ki, kj, c = report.max_pair
        print(f"  Worst pair overall: {label(ki)} <-> {label(kj)}  r={c:+.2f}")
