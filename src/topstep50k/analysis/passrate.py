"""Realized Combine pass-rate over a long backtest.

The acceptance criterion the user set: a "challenge-ready" strategy
must show > 50% Combine pass-rate when its 4-year daily-PnL series is
diced into 30-45 day windows (the typical pass-window). This module is
the instrument that produces that number.

Two related primitives:

  * `simulate_combine_window(daily_pnl_window, rules, starting_balance)`
    -- runs the strategy's daily PnL series through the Combine rule
    machine (trailing MLL, profit target, consistency) and reports
    pass / fail / breach-reason.
  * `realized_pass_rate(daily_pnl, rules, starting_balance, ...)` --
    slides a fixed-length window across the daily-PnL series with a
    configurable stride and counts how many windows passed.

The simulator is deliberately separate from `analysis/bootstrap.py`
(which resamples blocks). Here we want the LITERAL historical
pass-rate, not a resampled estimate -- "if a trader had started a
Combine on day t, would they have passed by day t+W?"

No look-ahead: each window only consumes daily PnLs whose dates fall
strictly within [start, end]. The MLL state is initialised fresh per
window (each Combine starts from $0 drawdown). Days outside the window
are ignored.

References: TopStep rules per docs/rules_sources.md.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Iterable, Sequence

from topstep50k.rules.topstep import TopstepRules, TrailingMLL


@dataclass(frozen=True)
class CombineWindowResult:
    """Outcome of running one Combine cycle over a daily-PnL window."""

    start_day: date
    end_day: date
    n_days: int
    outcome: str  # 'pass' | 'mll_breach' | 'consistency_fail' | 'no_target'
    final_pnl: Decimal
    days_to_outcome: int  # -1 if outcome is 'no_target' (ran to end)
    breach_day: date | None = None


@dataclass(frozen=True)
class RealizedPassRate:
    n_windows: int
    n_passed: int
    pass_rate: float
    outcomes: dict[str, int]  # outcome -> count
    window_results: list[CombineWindowResult] = field(default_factory=list)
    window_days: int = 0
    stride_days: int = 0

    @property
    def ci_95_low(self) -> float:
        """Wilson 95% lower bound on the pass-rate."""
        return _wilson_ci(self.n_passed, self.n_windows, 1.96)[0]

    @property
    def ci_95_high(self) -> float:
        return _wilson_ci(self.n_passed, self.n_windows, 1.96)[1]


def _wilson_ci(k: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    rad_sq = p * (1 - p) / n + z * z / (4 * n * n)
    rad = (rad_sq ** 0.5) * z / denom
    return max(0.0, centre - rad), min(1.0, centre + rad)


def simulate_combine_window(
    daily_pnl: Sequence[tuple[date, Decimal]],
    *,
    rules: TopstepRules,
    starting_balance: Decimal,
) -> CombineWindowResult:
    """Run one Combine cycle over an ordered daily-PnL series.

    Stops on the first MLL breach. After all days are consumed, applies
    the consistency check. A "pass" requires reaching the profit target
    on or before the last day AND passing consistency.

    `daily_pnl` is a sequence of (date, decimal_pnl) tuples, ascending.
    """
    if not daily_pnl:
        raise ValueError("empty daily_pnl window")
    days_sorted = sorted(daily_pnl, key=lambda x: x[0])
    start_day = days_sorted[0][0]
    end_day = days_sorted[-1][0]
    n_days = len(days_sorted)

    mll = TrailingMLL(starting_balance, rules.max_loss_limit_distance)
    eq = starting_balance
    cumulative: dict[date, Decimal] = {}
    passed_target_on: int | None = None

    for i, (day, pnl) in enumerate(days_sorted):
        eq = eq + pnl
        cumulative[day] = pnl
        # MLL is evaluated against the EOD balance after the trail has
        # ratcheted from the PREVIOUS day. We update the trail at EOD
        # AFTER the breach check, matching the engine's per-day order.
        breach = rules.check_max_loss(eq, mll.anchor)
        if breach is not None:
            return CombineWindowResult(
                start_day=start_day, end_day=end_day, n_days=n_days,
                outcome="mll_breach", final_pnl=eq - starting_balance,
                days_to_outcome=i + 1, breach_day=day,
            )
        mll.update_end_of_day(eq)
        if passed_target_on is None and rules.reached_profit_target(eq):
            passed_target_on = i

    if passed_target_on is not None:
        cons = rules.check_consistency(cumulative)
        if cons is not None:
            return CombineWindowResult(
                start_day=start_day, end_day=end_day, n_days=n_days,
                outcome="consistency_fail",
                final_pnl=eq - starting_balance,
                days_to_outcome=passed_target_on + 1,
                breach_day=None,
            )
        return CombineWindowResult(
            start_day=start_day, end_day=end_day, n_days=n_days,
            outcome="pass", final_pnl=eq - starting_balance,
            days_to_outcome=passed_target_on + 1,
            breach_day=None,
        )

    return CombineWindowResult(
        start_day=start_day, end_day=end_day, n_days=n_days,
        outcome="no_target", final_pnl=eq - starting_balance,
        days_to_outcome=-1, breach_day=None,
    )


def realized_pass_rate(
    daily_pnl: dict[date, Decimal] | Iterable[tuple[date, Decimal]],
    *,
    rules: TopstepRules,
    starting_balance: Decimal,
    window_days: int = 30,
    stride_days: int = 1,
) -> RealizedPassRate:
    """Slide a `window_days`-long Combine window over the daily-PnL
    series with the given stride. Returns aggregated pass-rate plus
    every window's outcome.

    Default stride=1 means every starting trading day is a separate
    Combine attempt (a sliding-window pass-rate). Set stride_days=window_days
    for strictly non-overlapping windows.
    """
    if window_days < 2:
        raise ValueError("window_days must be >= 2")
    if stride_days < 1:
        raise ValueError("stride_days must be >= 1")

    if isinstance(daily_pnl, dict):
        items = sorted(daily_pnl.items(), key=lambda x: x[0])
    else:
        items = sorted(daily_pnl, key=lambda x: x[0])
    if len(items) < window_days:
        raise ValueError(
            f"need at least {window_days} days for one window; have {len(items)}"
        )

    results: list[CombineWindowResult] = []
    for start_idx in range(0, len(items) - window_days + 1, stride_days):
        window = items[start_idx : start_idx + window_days]
        results.append(simulate_combine_window(
            window, rules=rules, starting_balance=starting_balance,
        ))
    n_passed = sum(1 for r in results if r.outcome == "pass")
    counts = Counter(r.outcome for r in results)
    return RealizedPassRate(
        n_windows=len(results),
        n_passed=n_passed,
        pass_rate=n_passed / len(results) if results else 0.0,
        outcomes=dict(counts),
        window_results=results,
        window_days=window_days,
        stride_days=stride_days,
    )
