# Three-strategy ensemble -- findings

Pre-committed parameters, 2y TRAIN / 2.25y TEST split. ALL THREE
strategies included in the headline ensemble (no drop-the-loser
selection). Numbers below are TEST-window only unless flagged TRAIN.

## TL;DR

  * **Correlation structure works.** All three pairwise correlations
    are near zero in TEST: |corr| < 0.15. The strategies really are
    orthogonal sources of PnL.
  * **Drawdown drops dramatically.** ORB-alone TEST MaxDD was $26K,
    OvernightDrift-alone was $53K. The all-three ensemble TEST MaxDD
    is **$7,704** -- 3-7x smaller than any component.
  * **Sharpe lifts modestly.** Ensemble Sharpe (TEST) = +0.72 vs ORB's
    +0.31. But after **deflating for 7 reported trials, DSR = 0.72**
    (need > 0.95). Not statistically significant.
  * **Pass-rate still does not clear 50%.** Headline ensemble:
    pass30 = 8.8%, pass45 = 23.6%. Best single-component:
    OvernightDrift solo pass45 = 21.9%. The drop-MeanRev ensemble
    hits 16.8% / 16.3% -- best 30-day so far on TEST.

## Why MeanRev hurt the 30-day pass-rate

Inverse-vol sizing assigned MeanRev **66%** of the ensemble weight
because its TRAIN-window stdev was only $297/day (vs $1003 ORB,
$1382 OvernightDrift). MeanRev's TRAIN Sharpe was a robust +0.63 --
exactly the OOS-promising-but-fragile pattern. On TEST it collapsed
to +0.07. With 66% of the weight tied to a strategy that now barely
makes money, the ensemble's mean daily return is too small relative
to the (smaller) variance to clear the $3K/30d target often enough.

This is **textbook overfit on a single strategy**, despite the
strategy itself having Bollinger-paper defaults. The TRAIN→TEST
Sharpe collapse from 0.63 to 0.07 is a 90% reduction.

## What the orthogonality bought us

The all-three ensemble's $7,704 MaxDD is genuinely impressive. A
trader running this combination would have survived all 2.25 years
of TEST without a 20%+ account drawdown. The cost is that the
ensemble's per-day mean is small ($16/day) because the strategies
are individually small-edge.

If you trade this ensemble at qty=1 contract on ES, the **typical
30-day window cumulative PnL** is well under $3000 -- which is why
pass30 is only 8.8%. The math from `combine_math.md` predicted
this: we need ~$100/day mean to hit 50% pass-rate, and the ensemble
delivers ~$16/day.

## Honest verdict

The ensemble proves the **diversification thesis** -- combining
orthogonal streams reduces risk by 3-7x -- but does NOT solve the
**edge thesis**: the three strategies' combined gross edge is still
too small for the Combine's $3K/30d target.

To clear pass-rate 50%, you'd need either:

  1. Scale up: trade qty > 1 on ES, accept higher MaxDD. But
     scaling 1x→5x multiplies MaxDD too -- $7,704 × 5 = $38,520 >
     $2K MLL. Doesn't help.
  2. Better components: replace MeanRev (Sharpe 0.07 OOS) with
     something that holds up. But this risks a fishing expedition.
  3. Different rule shape: $25K Combine with $1.5K target halves
     the daily-mean demand to ~$50/day, which the ensemble can
     plausibly hit. This is a structural change, not a strategy
     improvement.

## Drop-one diagnostics (sensitivity, not selection)

| dropped | pass30 | pass45 | Sharpe | MaxDD |
|:--|---:|---:|---:|---:|
| ORB             | 11.8% | 22.9% | +0.58 | $-11,511 |
| MeanRev         | 16.8% | 16.3% | +0.77 | $-17,686 |
| OvernightDrift  |  4.7% | 13.2% | +0.30 |  $-9,365 |
| none (all-three)|  8.8% | 23.6% | +0.72 |  $-7,704 |

  * Dropping MeanRev gives the best 30-day pass-rate (16.8%) and
    Sharpe (+0.77), but DOUBLES the MaxDD ($17.7K vs $7.7K).
    Diversification gain came from MeanRev's near-zero correlation
    with the other two even though its own edge collapsed.
  * Dropping OvernightDrift catastrophically reduces pass-rate
    (4.7%) because OvernightDrift is the largest mean producer.
  * Dropping ORB does the least damage but loses the structural
    intraday-trend exposure.

## What we should NOT do next

  * Re-weight by something else (equal weights, risk-parity with
    correlations, etc.). Choosing weights AFTER seeing the test
    is in-sample selection.
  * Drop MeanRev and call the result the "real" strategy. Same
    problem -- the choice is post-hoc.
  * Re-tune MeanRev parameters. We pre-committed to Bollinger
    defaults; iterating would be the overfit trap by another name.

The honest decision: report this finding and let the user pick the
next direction.
