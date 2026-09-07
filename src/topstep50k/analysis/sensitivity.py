"""Parameter-sensitivity sweep with overfit detection.

Why this exists: a parameter grid run against ONE in-sample window is
the classic backtest-mining trap [ref:lopez_pseudo]. This module
combines the grid with walk-forward CV, then reports both the
out-of-sample winner AND the Probability of Backtest Overfitting
[ref:bailey_pbo] computed across the parameter matrix.

Output you should look at, in order:
  1. PBO -- if >> 0.5, the sweep has selection bias; the "winner" is
     almost certainly an artefact.
  2. Per-cell DSR (deflated for the size of the grid) -- if the winner's
     DSR is not > 0.95, the OOS edge isn't statistically distinguishable
     from "best of N noise draws".
  3. Per-cell OOS Sharpe surface -- only after the above two pass.

This module is engine-agnostic. You hand it a `runner(params) ->
list[float]` that returns the OOS daily-return series for those params
(typically one element per test-fold day). It does NOT run the engine
itself -- decoupling lets us use the same sweep on a vector strategy in
a notebook or on the full event-driven backtester in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from topstep50k.analysis.dsr import (
    deflated_sharpe,
    probabilistic_sharpe,
    probability_of_backtest_overfit,
)


@dataclass(frozen=True)
class SweepCell:
    params: dict[str, Any]
    sharpe_annual: float
    psr: float
    n_periods: int
    daily_returns: list[float]


@dataclass(frozen=True)
class SensitivityResult:
    cells: list[SweepCell]
    winner: SweepCell
    dsr_for_winner: float
    expected_max_sharpe: float
    pbo: float
    grid_shape: dict[str, int] = field(default_factory=dict)


def _grid_product(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    names = list(grid.keys())
    combos = []
    for values in product(*(grid[n] for n in names)):
        combos.append(dict(zip(names, values, strict=True)))
    return combos


def sensitivity_sweep(
    grid: Mapping[str, Sequence[Any]],
    *,
    runner: Callable[[dict[str, Any]], Sequence[float]],
    periods_per_year: int = 252,
    pbo_slices: int = 16,
    min_periods: int = 16,
    rng: np.random.Generator | None = None,
) -> SensitivityResult:
    """Sweep `runner` over the cartesian product of `grid`.

    Parameters
    ----------
    grid : { param_name: [values, ...], ... }. All combinations are
        evaluated.
    runner : callable that takes a single combo dict and returns the OOS
        per-period return series. Same length across combos is REQUIRED
        for PBO to work (we stack them into a (T, N) matrix).
    periods_per_year : annualisation factor (252 for daily, 252*23 for
        hourly futures, etc.).
    pbo_slices : `s` parameter for combinatorially symmetric CV. Must
        be even, >= 4. With T returns we need T >= pbo_slices.
    min_periods : skip cells whose returns are shorter than this.

    Returns
    -------
    SensitivityResult with per-cell stats, the OOS winner, the deflated
    Sharpe (for the winner), and the matrix-level PBO.
    """
    combos = _grid_product(grid)
    if not combos:
        raise ValueError("empty parameter grid")
    cells: list[SweepCell] = []
    return_lengths: set[int] = set()
    for combo in combos:
        rets = list(runner(combo))
        if len(rets) < min_periods:
            continue
        return_lengths.add(len(rets))
        psr_res = probabilistic_sharpe(rets, periods_per_year=periods_per_year)
        cells.append(SweepCell(
            params=combo, sharpe_annual=psr_res.sharpe, psr=psr_res.psr,
            n_periods=len(rets), daily_returns=rets,
        ))
    if not cells:
        raise ValueError(
            f"no parameter combo produced >= {min_periods} OOS periods"
        )

    winner = max(cells, key=lambda c: c.sharpe_annual)

    # Deflate the winner's Sharpe by the entire trial population.
    all_sharpes = [c.sharpe_annual for c in cells]
    dsr_res = deflated_sharpe(
        winner.daily_returns,
        all_trial_sharpes_annual=all_sharpes,
        periods_per_year=periods_per_year,
    )

    # PBO matrix needs every cell on the same time grid.
    pbo = float("nan")
    if len(return_lengths) == 1 and len(cells) >= 2:
        T = next(iter(return_lengths))
        if T >= pbo_slices:
            mat = np.array([c.daily_returns for c in cells]).T  # (T, N)
            pbo = probability_of_backtest_overfit(mat, s=pbo_slices, rng=rng)

    return SensitivityResult(
        cells=cells,
        winner=winner,
        dsr_for_winner=dsr_res.dsr,
        expected_max_sharpe=dsr_res.expected_max_sharpe,
        pbo=pbo,
        grid_shape={k: len(v) for k, v in grid.items()},
    )
