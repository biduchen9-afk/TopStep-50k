"""Walk-forward cross-validation.

The audit contract: a fold's TEST window is NEVER visible to the
strategy during selection. Anchored ("expanding") and rolling
("non-overlapping" / "fixed-window") schemes are supported.

Typical use:

    folds = walk_forward_folds(
        all_days, train=180, test=30, step=30, anchored=True,
    )
    for fold in folds:
        params = optimise_on(fold.train_days)   # in-sample only
        result = backtest_on(fold.test_days, params=params)
        record(result)
    aggregate(results)   # OOS performance only

This module produces the fold boundaries. The CALLER is responsible for
slicing the bar stream by those days — the typing here is intentionally
opaque about the price data because that lets us reuse the harness for
any time-keyed dataset (bars, daily returns, signals, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class WalkForwardFold:
    index: int  # 0-based fold number
    train_start: date
    train_end: date  # inclusive
    test_start: date
    test_end: date  # inclusive

    def __post_init__(self) -> None:
        if not (self.train_start <= self.train_end < self.test_start <= self.test_end):
            raise ValueError(
                f"Invalid fold #{self.index}: dates not strictly ordered "
                f"({self.train_start} {self.train_end} {self.test_start} {self.test_end})"
            )

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days + 1

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days + 1


def walk_forward_folds(
    days: list[date],
    *,
    train: int,
    test: int,
    step: int | None = None,
    anchored: bool = True,
) -> list[WalkForwardFold]:
    """Build walk-forward folds over a sorted list of trading days.

    Parameters
    ----------
    days : list of distinct trading dates, ascending.
    train : minimum size of the initial training window (in days).
    test : size of each test window (in days).
    step : how many days to advance between folds. Defaults to `test`
           (non-overlapping test windows).
    anchored : if True, train window grows ("anchored walk-forward").
               If False, train window slides ("rolling walk-forward").
    """
    if step is None:
        step = test
    if train <= 0 or test <= 0 or step <= 0:
        raise ValueError("train, test, step must all be positive")
    if len(days) < train + test:
        raise ValueError(
            f"Need at least {train + test} days, got {len(days)}"
        )
    # Verify strict ordering and uniqueness
    for i in range(1, len(days)):
        if days[i] <= days[i - 1]:
            raise ValueError(f"days must be strictly ascending; "
                             f"{days[i - 1]} >= {days[i]}")

    folds: list[WalkForwardFold] = []
    i = 0
    train_start_idx = 0
    train_end_idx = train - 1
    fold_index = 0
    while True:
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test - 1
        if test_end_idx >= len(days):
            break
        folds.append(
            WalkForwardFold(
                index=fold_index,
                train_start=days[train_start_idx],
                train_end=days[train_end_idx],
                test_start=days[test_start_idx],
                test_end=days[test_end_idx],
            )
        )
        fold_index += 1
        train_end_idx += step
        if not anchored:
            train_start_idx += step
        i += step
    return folds
