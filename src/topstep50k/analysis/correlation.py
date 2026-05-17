"""Multi-asset rolling correlation, look-ahead safe.

The estimator consumes per-symbol bar streams aligned to a common time
grid (caller's responsibility) and emits a rolling Pearson correlation
matrix at each grid point that has a full window. The matrix at index
t uses only returns from indices [t-window+1 .. t] -- strictly causal.

Use cases:
  * Portfolio-level diversification check: too-correlated open
    positions count as one risk slot for sizing.
  * Regime feature: average pairwise correlation is itself a regime
    signal (correlations cluster up during stress).

For shrinkage-stabilised covariance see [ref:ledoit_wolf_2004]. For
correlation-based clustering / Hierarchical Risk Parity see
[ref:lopez_hrp]. This module is the cheap rolling primitive everything
else builds on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Sequence

import numpy as np

from topstep50k.engine.clock import Clock


@dataclass(frozen=True)
class CorrelationSnapshot:
    ts: datetime
    symbols: tuple[str, ...]
    matrix: np.ndarray  # shape (k, k), symmetric, diag=1

    def pair(self, a: str, b: str) -> float:
        i, j = self.symbols.index(a), self.symbols.index(b)
        return float(self.matrix[i, j])

    def avg_offdiag(self) -> float:
        k = len(self.symbols)
        if k < 2:
            return 0.0
        m = self.matrix.copy()
        np.fill_diagonal(m, 0.0)
        return float(m.sum() / (k * (k - 1)))


def aligned_returns_from_closes(
    closes_by_symbol: dict[str, list[tuple[datetime, Decimal]]],
) -> tuple[list[datetime], dict[str, list[float]]]:
    """Align per-symbol close series to their common timestamps and
    convert to log-returns. Returns (timestamps, returns_by_symbol).

    Only timestamps present in EVERY symbol survive. The first surviving
    timestamp of each symbol contributes no return (no prior).
    """
    if not closes_by_symbol:
        return [], {}
    symbols = sorted(closes_by_symbol.keys())
    common: set[datetime] | None = None
    for s in symbols:
        ts_set = {ts for ts, _ in closes_by_symbol[s]}
        common = ts_set if common is None else common & ts_set
    common_ts = sorted(common or set())
    if len(common_ts) < 2:
        return common_ts, {s: [] for s in symbols}
    # Index closes for fast lookup
    indexed: dict[str, dict[datetime, float]] = {
        s: {ts: float(c) for ts, c in closes_by_symbol[s]} for s in symbols
    }
    rets: dict[str, list[float]] = {s: [] for s in symbols}
    for i in range(1, len(common_ts)):
        prev_ts, cur_ts = common_ts[i - 1], common_ts[i]
        for s in symbols:
            p, c = indexed[s][prev_ts], indexed[s][cur_ts]
            if p <= 0 or c <= 0:
                rets[s].append(0.0)
            else:
                rets[s].append(float(np.log(c / p)))
    # Drop the first index since it had no return.
    return common_ts[1:], rets


class RollingCorrelation:
    """Streaming rolling correlation. Caller pushes one aligned slice of
    returns per timestamp; updates(ts) returns a snapshot if the window
    is full, else None.

    Invariants:
      * snapshot.ts is the timestamp at which the LAST return in the
        window was observed -- never a forward-looking ts.
      * If a Clock is supplied, every snapshot ts is asserted visible.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        window: int,
        *,
        clock: Clock | None = None,
        min_periods: int | None = None,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        if len(symbols) < 2:
            raise ValueError("need at least 2 symbols for a correlation matrix")
        self._symbols = tuple(symbols)
        self._k = len(symbols)
        self._window = window
        self._min = min_periods or window
        self._clock = clock
        # Ring buffer of recent returns; column order matches symbols
        self._buf = np.zeros((window, self._k), dtype=np.float64)
        self._n = 0  # number of pushes so far
        self._cursor = 0

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def push(self, ts: datetime, returns_by_symbol: dict[str, float]) -> CorrelationSnapshot | None:
        if self._clock is not None:
            self._clock.assert_visible(ts, "rolling-correlation push")
        row = np.array([returns_by_symbol[s] for s in self._symbols], dtype=np.float64)
        self._buf[self._cursor] = row
        self._cursor = (self._cursor + 1) % self._window
        self._n += 1
        if self._n < self._min:
            return None
        # Use only the populated rows when buffer is still warming up
        if self._n < self._window:
            data = self._buf[: self._n]
        else:
            # Re-order ring buffer so the OLDEST row is at index 0; not
            # strictly needed for correlation (it is centring + scaling
            # invariant) but avoids surprises for callers who introspect.
            data = np.vstack([self._buf[self._cursor:], self._buf[: self._cursor]])
        mu = data.mean(axis=0)
        centred = data - mu
        # Sample covariance with ddof=1
        denom = max(data.shape[0] - 1, 1)
        cov = (centred.T @ centred) / denom
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        # Build correlation
        outer = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(outer > 1e-15, cov / outer, 0.0)
        np.fill_diagonal(corr, 1.0)
        # Clip numerical drift
        corr = np.clip(corr, -1.0, 1.0)
        return CorrelationSnapshot(ts=ts, symbols=self._symbols, matrix=corr)
