# Walk-forward inverse-vol weights on the gated ensemble

Follow-up to `regime_conditioning_findings.md`. Same three strategies,
same literature-grounded gates, same TRAIN/TEST split. Only the
sizing rule changed: weights are recomputed at each trading day t
from the previous 252 days of gated daily PnL (strictly causal,
window ends at day t-1).

## Result: marginal Sharpe lift, NO pass-rate improvement

| ensemble on TEST | Sharpe | MaxDD | $50K pass30 | $50K pass45 | $25K pass30 | $25K pass45 |
|:--|---:|---:|---:|---:|---:|---:|
| Baseline FIXED weights | +0.72 | -$7,704 | 8.8% | 23.6% | 28.2% | **33.3%** |
| Gated FIXED weights (prior phase 5) | +1.16 | -$4,734 | 2.7% | 3.5% | 22.6% | 31.6% |
| **Gated WALK-FORWARD weights (new)** | **+1.19** | **-$3,839** | 1.1% | 2.1% | 22.1% | 30.5% |

Walk-forward bought:

  * Sharpe +0.03 (1.16 -> 1.19, +3%).
  * MaxDD reduction: -19% ($4,734 -> $3,839).
  * Pass-rates moved DOWN by ~1pp across the board.

## Why walk-forward did not solve pass-rate

The walk-forward weights converged on essentially the SAME distribution
as the fixed weights derived from baseline TRAIN stdev:

|         | fixed | walk-forward (mean over TEST) | walk-forward (range) |
|:--|---:|---:|---:|
| ORB             | 0.196 | 0.154 | [0.117, 0.194] |
| MeanRev         | 0.662 | 0.720 | [0.614, 0.792] |
| OvernightDrift  | 0.142 | 0.126 | [0.089, 0.199] |

MeanRev's gated stdev is even SMALLER on TEST than it was on TRAIN
(because the gate cuts trade count further), so inverse-vol gives it
EVEN MORE weight under walk-forward (0.66 -> 0.72). The "MeanRev
hyperweighted because of low stdev" problem we identified in the
prior phase 5 writeup got worse, not better, under WF.

**The real binding constraint was never WHEN we set weights -- it
was the OBJECTIVE the weights are optimised for.** Inverse-vol
minimises ensemble VARIANCE. Pass-rate wants high MEAN per-day PnL
(enough to clear $1.5K / $3K cumulative targets in 30 days). Those
are different objectives, and the gated MeanRev has tiny variance
but tinier mean -- so inverse-vol pushes us into a corner where we
are extremely Sharpe-efficient but produce very little dollar
income per day.

## Drop-one sensitivity (DIAGNOSTIC ONLY)

| variant (walk-forward, gated) | Sharpe | $25K pass30 | $25K pass45 |
|:--|---:|---:|---:|
| WF drop ORB | +1.21 | 25.0% | **39.1%** |
| WF drop MeanRev | +1.07 | 22.7% | 24.5% |
| WF drop OvernightDrift | +0.56 | 13.2% | 16.6% |

"WF drop ORB" reaches $25K pass45 = 39.1%, the highest single point
estimate the project has ever produced. This is a SENSITIVITY
RESULT, not a recommendation -- picking the best subset
post-hoc is exactly the overfit trap.

## DSR

  * Trials accumulated this pass: 12
  * Walk-forward gated raw Sharpe (annual): 1.194
  * Expected-max under null (deflation): 0.694
  * DSR: 0.7992

Up from 0.7211 (gated-fixed, prior pass) and 0.7211 (baseline).
Still below the 0.95 statistical-significance threshold.

## What this teaches us

1. **Walk-forward sizing is the WRONG TOOL for this problem.** It
   addresses variance, but pass-rate is a non-linear function of
   the per-day PnL distribution that rewards mean and consistency
   over variance reduction.
2. **Inverse-vol is the wrong WEIGHTING SCHEME for pass-rate.** A
   strategy that wins a tiny amount often is INDISTINGUISHABLE from
   a strategy that loses a tiny amount often, under inverse-vol
   weighting. Both get heavy weights -- but the first contributes
   to pass-rate while the second hurts it.

## Pre-committed next step candidates

Any of these would be honest follow-ups; I would recommend (1).

  1. **Pass-rate-aware sizing.** Replace inverse-vol with a weight
     scheme that directly targets the 30-day rolling cumulative
     distribution: e.g., for each strategy compute its 30-day
     rolling cumulative PnL and weight by the FRACTION OF WINDOWS
     in which that strategy alone would have passed. This directly
     optimises the objective the user actually cares about.

  2. **Kelly-fractional sizing with MLL hard cap.** For each
     strategy compute its empirical edge and variance over TRAIN;
     size by Kelly-fraction subject to an MLL hard constraint
     (no path can cross -$2K in 30 days). More math-heavy but
     directly addresses the rule set's geometry.

  3. **Acknowledge the ceiling.** Walk-forward sizing got us to
     Sharpe 1.19 and $25K pass45 31% on this 3-strategy ensemble.
     The marginal returns from further sizing tweaks are very
     small. Adding a 4th uncorrelated strategy (Paper 2's VWAP-
     pullback shape, for example) is more likely to move the
     needle than more sizing work.
