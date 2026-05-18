# ORB / HMM exploration -- findings summary

This document captures the empirical results from the May 2026 ORB +
HMM exploration. It complements `combine_math.md` (which works out the
analytical Sharpe required) by recording what the actual code found.

## TL;DR

  * **No single-asset ES Opening Range Breakout configuration in a
    144-cell grid reached 25% 30-day pass-rate** on the 4-year file
    (Jan 2022 - Apr 2026). The best 30-day pass-rate was **19.2%** at
    `OR=30min, TP=1.0x, opposite-range stop, both directions,
    chop-filter=0.5`.
  * **DSR after deflation = 0.62**, i.e. the best-Sharpe cell is **NOT
    statistically distinguishable from selection bias** across the 144
    trials.
  * **PBO across the grid = 0.701** -- 70% probability that the
    in-sample winner does not generalise out of sample. Bailey's
    threshold for "real" is PBO < 0.3; we are not close.
  * **A 3-state HMM regime overlay made things worse.** Walk-forward
    (2y train / 1y select / 1y+ test): baseline ORB on TEST had
    pass30 = 17.2% and Sharpe 0.42; the HMM-gated version had
    pass30 = 6.3% and Sharpe -0.29. The "good" regime in 2024 had
    reversed sign in 2025-2026.

## Raw artefacts

  * `results/sweep_orb_es_full.txt` -- per-cell output of the 144-cell
    sweep, plus top-15 ranking, deflated Sharpe, and PBO.
  * `results/run_orb_hmm_regime.txt` -- HMM-gated walk-forward run.
  * `scripts/sweep_orb_es_full.py` -- the sweep itself
    (or_minutes x tp_multiple x stop_mode x direction x chop_filter).
  * `scripts/run_orb_hmm_regime.py` -- the HMM gating runner.
  * `docs/combine_math.md` -- analytical companion (what Sharpe the
    rule set actually demands).

## What the sweep matrix shows

Top 15 cells by 30-day realised pass-rate (out of 144 total):

| or | tp | stop | dir | chop | pass30 | pass45 | Sharpe | MaxDD |
|---:|---:|:--|:--|---:|---:|---:|---:|---:|
| 30 | 1.0 | opposite_range | both | 0.5 | 19.2% | 15.0% | +0.19 | $29,502 |
| 60 | 1.0 | opposite_range | both | 0.5 | 18.8% | 18.6% | +0.74 | $35,590 |
| 30 | 1.0 | opposite_range | both | 0.0 | 18.3% | 14.3% | +0.17 | $27,848 |
| 60 | 1.0 | opposite_range | both | 0.0 | 18.0% | 16.5% | +0.68 | $38,138 |
| 60 | 1.0 | opposite_range | long | 0.0 | 17.4% | 19.4% | +0.94 | $19,212 |
| 60 | 1.0 | opposite_range | long | 0.5 | 17.4% | 17.8% | +0.95 | $17,582 |
| 15 | 1.0 | opposite_range | both | 0.5 | 17.4% | 17.5% | +0.29 | $27,895 |
| 15 | 2.0 | opposite_range | long | 0.5 | 17.1% | 16.6% | +0.84 | $20,410 |
| 30 | 2.0 | opposite_range | long | 0.0 | 17.1% | 16.7% | +0.84 | $18,018 |
| 30 | 2.0 | opposite_range | both | 0.5 | 16.8% | 14.9% | +0.75 | $23,328 |

Observations:

  * **Highest pass-rate cells have low Sharpe.** They get there by
    trading both directions with the opposite-range stop, which allows
    a few large winning days to lift the cumulative distribution above
    $3K -- but increases variance enough to keep Sharpe at 0.17-0.74.
  * **Highest-Sharpe cells (~0.9-1.0) all use long-only direction with
    opposite-range stops.** They have moderate pass-rates (11-17%).
  * **The chop-filter helps only weakly.** Going from `chop=0.0` to
    `chop=0.5` (require today's OR width > 0.5x trailing-20 median)
    leaves the metric within a couple of points. Higher chop values
    starve the strategy of trades.

## Why HMM did not help

Three-state Gaussian HMM was fit on 5-d daily features (log_return,
abs_return, range_pct, vol_5, vol_20) on the first 2 years, then
filtered forward across the whole 4-year file (online Rabiner alpha
recurrence, so no look-ahead).

The learned states had interpretable means:

  | state | log_r | abs_r | range | vol_5 | vol_20 | interpretation |
  |---|---|---|---|---|---|---|
  | 0 | -0.003 | 0.003 | 0.009 | 0.008 | 0.009 | quiet drift-down |
  | 1 | -0.012 | 0.012 | 0.022 | 0.012 | 0.012 | high-vol selloff |
  | 2 | +0.008 | 0.008 | 0.014 | 0.010 | 0.010 | quiet drift-up |

On the SELECT window (year 3), regime 1 had positive ORB mean PnL
(+$196/day, 43 days) -- the high-vol selloff regime, where ORB
breakouts work best because there are big intraday moves and ORB
captures their direction. Regimes 0 and 2 had negative means.

But on the TEST window (year 4+), the days the HMM tagged as regime 1
no longer worked the same way. The strategy lost money on TEST under
the filter; baseline (no filter) was better. **The HMM correctly
identified state structure but the conditional payoff of ORB given
state was not stationary.**

## What this means for the project

The audit infrastructure (Tier-0 + Tier-1, the Combine rules, the
analysis stack -- DSR, PBO, bootstrap CIs, the regime filter) is
working correctly: it produced a clear, statistically honest verdict.

The verdict is **ORB on ES, parameterised in any of the natural ways,
is structurally incompatible with the $50K Combine pass criteria.**
The math in `combine_math.md` predicts this (we'd need annualised
Sharpe ~2-3, we have ~0.9 raw, ~0.6 deflated), and the empirical PBO
of 0.70 confirms it.

Productive next directions (in rough order of plausibility):

  1. **Multi-strategy ensemble** -- ORB + a mean-revert with hard stops
     + a momentum overlay, each tiny so the daily-variance budget is
     honoured. Edge comes from uncorrelated streams.
  2. **News / event-driven** -- trade only into known schedule
     (CPI, FOMC, NFP) with pre-defined position sizes. Cuts trade
     count drastically but each trade has positive expectancy.
  3. **VWAP pullback with intraday trend filter** -- mean-revert inside
     trend direction. Higher hit-rate strategies natively pass the
     consistency rule.
  4. **Lower-target challenge first** -- prove the strategy on the
     $25K Combine ($1500 target / $1500 MLL) before scaling.

What we should NOT spend more time on:

  * Further ORB parameter tuning. The grid has been exhausted; deeper
    grids will overfit harder.
  * Naive HMM-regime overlays on top of weakly-edged strategies.
    Regime detection helps when the regime gates a real edge; ORB's
    edge is not large enough for state-conditional gating to recover.
