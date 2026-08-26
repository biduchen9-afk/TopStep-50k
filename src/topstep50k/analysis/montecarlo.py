"""Monte Carlo pass-rate estimation via stationary block-bootstrap resampling.

`simulate_sequential_accounts` on the single realized OOS path gives ONE
number (e.g. v1: 13/26 accounts = 50.0%) -- but that's one draw from a
distribution, not the distribution itself. With only 26 non-overlapping
account-attempts in the sample, the sampling uncertainty on that 50.0%
is wide: treating it as a simple binomial gives a standard error of
sqrt(0.5*0.5/26) ~= 9.8pp, i.e. a rough 95% interval of roughly
[31%, 69%] -- and that's before accounting for the fact that daily P&L
isn't actually i.i.d. (autocorrelated streaks), which the binomial
approximation ignores entirely.

This resamples the daily PnL series with a stationary block bootstrap
(same block_len=10 convention as evaluation/harness.py's bootstrap
CI-lower gate), preserving within-block autocorrelation -- winning and
losing streaks stay intact as units -- while destroying the SPECIFIC
historical sequencing of those streaks. Each resampled path is run
through simulate_sequential_accounts to get one Monte Carlo draw of the
pass rate; repeating N times builds an actual distribution instead of
trusting the single historical path.

This is NOT a claim that daily returns are i.i.d. or stationary, and it
cannot simulate a regime that never appeared in the sample -- a genuine
new regime shift is outside what any resample of historical data can
answer. It answers a narrower, still useful question: given the exact
mix of winning/losing streaks actually observed, how much does the
specific order those streaks happened to occur in matter to the pass
rate? A pass rate that's stable across resamples is a real property of
the strategy's day-to-day behavior; one that swings wildly between
resamples means the single-path number was largely luck of the draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import numpy as np

from topstep50k.analysis.passrate import simulate_sequential_accounts
from topstep50k.rules.topstep import TopstepRules


@dataclass(frozen=True)
class MonteCarloPassRateResult:
    n_sims: int
    block_len: int
    n_days: int
    pass_rates: np.ndarray             # one draw per simulation
    mean_pass_rate: float
    median_pass_rate: float
    p05: float
    p95: float
    prob_above_50pct: float            # P(a resampled path's pass rate > 50%)
    mean_n_accounts: float             # avg attempts spanned per resampled path


def monte_carlo_pass_rate(
    daily_pnl: dict[date, Decimal],
    *,
    rules: TopstepRules,
    starting_balance: Decimal,
    n_sims: int = 1000,
    block_len: int = 10,
    checkpoint: Decimal = Decimal("1500"),
    seed: int = 42,
) -> MonteCarloPassRateResult:
    """Resample `daily_pnl` n_sims times (stationary block bootstrap,
    wrap-around at the series boundary) and run each resample through
    simulate_sequential_accounts. Returns the distribution of resulting
    pass rates, not just their mean -- see module docstring for why the
    spread matters as much as the center here.

    Each resampled path keeps the ORIGINAL calendar day index (only the
    PnL *values* are reshuffled in blocks) so the Combine's day-boundary
    and consistency-rule mechanics see the same trading-day cadence as
    the real data.
    """
    days = sorted(daily_pnl)
    n = len(days)
    if n < block_len:
        raise ValueError(f"need at least block_len={block_len} days, have {n}")
    values = np.array([float(daily_pnl[d]) for d in days])

    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block_len)  # ceil

    pass_rates = np.empty(n_sims)
    n_accounts = np.empty(n_sims)
    for i in range(n_sims):
        starts = rng.integers(0, n, size=n_blocks)
        blocks = []
        for s in starts:
            if s + block_len <= n:
                blocks.append(values[s : s + block_len])
            else:
                blocks.append(np.concatenate([values[s:], values[: block_len - (n - s)]]))
        resampled = np.concatenate(blocks)[:n]

        synth_daily = {d: Decimal(str(round(float(v), 2))) for d, v in zip(days, resampled)}
        summary = simulate_sequential_accounts(
            synth_daily, rules=rules, starting_balance=starting_balance,
            checkpoint=checkpoint,
        )
        pass_rates[i] = summary.pass_rate
        n_accounts[i] = summary.n_accounts

    return MonteCarloPassRateResult(
        n_sims=n_sims, block_len=block_len, n_days=n,
        pass_rates=pass_rates,
        mean_pass_rate=float(pass_rates.mean()),
        median_pass_rate=float(np.median(pass_rates)),
        p05=float(np.percentile(pass_rates, 5)),
        p95=float(np.percentile(pass_rates, 95)),
        prob_above_50pct=float((pass_rates > 0.5).mean()),
        mean_n_accounts=float(n_accounts.mean()),
    )
