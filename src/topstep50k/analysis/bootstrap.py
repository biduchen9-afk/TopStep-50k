"""Monte Carlo via stationary block bootstrap.

The stationary bootstrap (Politis & Romano 1994) preserves the
autocorrelation structure of a return series by resampling blocks of
geometrically-distributed length and concatenating them. This is the
right MC technique for backtest returns because daily PnL is rarely
i.i.d. (trends, vol clustering, regime persistence).

API:
    pass_rate, ci, draws = topstep_pass_probability(
        daily_pnl=...,         # observed daily PnL series
        rules=...,             # TopstepRules to score each MC draw against
        starting_balance=...,
        n_draws=10_000,
        block_mean_length=5,   # expected block size in days
        target_n_days=60,      # length of each simulated cycle
        seed=42,
    )

The estimator never accesses the SOURCE of the daily PnL series — it
treats it as an exchangeable summary. So the bootstrap output is only
meaningful if the underlying series is itself OOS (from a walk-forward
test fold).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Sequence

import numpy as np

from topstep50k.rules.topstep import TopstepRules, TrailingMLL


@dataclass(frozen=True)
class BootstrapResult:
    n_draws: int
    pass_rate: float
    ci_low: float
    ci_high: float
    mll_breach_rate: float
    consistency_fail_rate: float
    median_final_pnl: float
    p5_final_pnl: float
    p95_final_pnl: float


def stationary_bootstrap_indices(
    n: int, length: int, block_mean: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano sampling: each step continues the current block
    with probability 1 - 1/block_mean, otherwise jumps to a new random
    start. Wraps circularly."""
    p_new = 1.0 / block_mean
    idx = np.empty(length, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    for t in range(1, length):
        if rng.random() < p_new:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def simulate_cycle(
    daily_pnl_arr: np.ndarray,
    rules: TopstepRules,
    starting_balance: Decimal,
    n_days: int,
    block_mean: float,
    rng: np.random.Generator,
) -> tuple[str, float, dict]:
    """Simulate ONE Combine cycle by resampling daily PnL.

    Returns
    -------
    outcome : 'pass' | 'mll_breach' | 'consistency_fail' | 'no_target' | 'dll_only'
    final_pnl : sum of realised day-PnLs (terminated early on MLL breach)
    diag : extra diagnostics dict
    """
    idx = stationary_bootstrap_indices(len(daily_pnl_arr), n_days, block_mean, rng)
    mll = TrailingMLL(starting_balance, rules.max_loss_limit_distance)
    eq = starting_balance
    daily_pnl: dict[int, Decimal] = {}
    for d in range(n_days):
        day_delta = Decimal(str(daily_pnl_arr[idx[d]]))
        # Intraday MLL check: applied at EOD against the trailing line.
        # For the bootstrap we treat each day's PnL as if realised all at
        # once at EOD — a conservative approximation that matches how
        # most prop traders summarise sessions.
        eq += day_delta
        daily_pnl[d] = day_delta
        breach = rules.check_max_loss(eq, mll.anchor)
        if breach is not None:
            return "mll_breach", float(eq - starting_balance), {"day": d}
        mll.update_end_of_day(eq)
        if rules.reached_profit_target(eq):
            # Profit target hit; check consistency.
            cons = rules.check_consistency({i: v for i, v in daily_pnl.items()})
            if cons is not None:
                return "consistency_fail", float(eq - starting_balance), {"day": d}
            return "pass", float(eq - starting_balance), {"day": d}
    # Ran out of days.
    cons = rules.check_consistency({i: v for i, v in daily_pnl.items()})
    if cons is not None:
        return "consistency_fail", float(eq - starting_balance), {}
    return "no_target", float(eq - starting_balance), {}


def topstep_pass_probability(
    daily_pnl: Sequence[Decimal],
    *,
    rules: TopstepRules,
    starting_balance: Decimal,
    n_draws: int = 10_000,
    target_n_days: int = 60,
    block_mean_length: float = 5.0,
    seed: int | None = None,
    ci_alpha: float = 0.05,
) -> BootstrapResult:
    """Run `n_draws` bootstrap cycles and report pass rate + CI."""
    if not daily_pnl:
        raise ValueError("daily_pnl is empty")
    arr = np.array([float(v) for v in daily_pnl], dtype=np.float64)
    rng = np.random.default_rng(seed)

    outcomes: list[str] = []
    finals: list[float] = []
    for _ in range(n_draws):
        outcome, final_pnl, _ = simulate_cycle(
            arr, rules, starting_balance, target_n_days,
            block_mean_length, rng,
        )
        outcomes.append(outcome)
        finals.append(final_pnl)

    finals_arr = np.array(finals)
    pass_count = sum(1 for o in outcomes if o == "pass")
    pass_rate = pass_count / n_draws

    # Wilson score interval for the pass-rate Bernoulli proportion.
    z = float(np.sqrt(2) * np.sqrt(np.pi)) if ci_alpha == 0.32 else 1.96
    if abs(ci_alpha - 0.05) > 1e-9:
        from math import sqrt
        # Approximate normal z for arbitrary alpha
        z = float(np.abs(np.percentile(np.random.default_rng(0).standard_normal(20000),
                                       100 * (1 - ci_alpha / 2))))
    denom = 1 + z * z / n_draws
    centre = (pass_rate + z * z / (2 * n_draws)) / denom
    radius = z * np.sqrt(pass_rate * (1 - pass_rate) / n_draws + z * z / (4 * n_draws ** 2)) / denom

    return BootstrapResult(
        n_draws=n_draws,
        pass_rate=pass_rate,
        ci_low=float(centre - radius),
        ci_high=float(centre + radius),
        mll_breach_rate=sum(1 for o in outcomes if o == "mll_breach") / n_draws,
        consistency_fail_rate=sum(1 for o in outcomes if o == "consistency_fail") / n_draws,
        median_final_pnl=float(np.median(finals_arr)),
        p5_final_pnl=float(np.percentile(finals_arr, 5)),
        p95_final_pnl=float(np.percentile(finals_arr, 95)),
    )
