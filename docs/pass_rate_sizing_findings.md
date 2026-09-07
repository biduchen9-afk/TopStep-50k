# Pass-rate-aware sizing on the gated ensemble

Phase 5 follow-up #2. Same three strategies, same literature-grounded
gates, same TRAIN/TEST split. Only the sizing rule changed: instead
of inverse-vol, weight each strategy by its SOLO pass-rate over the
relevant lookback window.

Pre-committed parameters: lookback 252 days, 30-day pass-rate
criterion, $25K Combine rules for the criterion (small target makes
TRAIN signal non-trivial). No exponents, no power-laws, no sweep.

## Headline

| ensemble on TEST | Sharpe | MaxDD | $50K p30 | $50K p45 | $25K p30 | $25K p45 |
|:--|---:|---:|---:|---:|---:|---:|
| inverse-vol fixed (prior best) | +1.22 | -$4,176 | 2.4% | 3.2% | 22.8% | **30.7%** |
| **STATIC pass-rate** | +1.03 | -$11,171 | **11.9%** | **23.9%** | 24.0% | 27.5% |
| **WALK-FORWARD pass-rate** | +1.08 | -$10,699 | **12.8%** | 21.6% | 20.8% | 23.3% |

**$50K pass30 lifted 5x (2.4% -> 11.9%) and pass45 lifted 7x (3.2%
-> 23.9%) under pass-rate-aware sizing.** This is the first sizing
change in the project that meaningfully moves the harder Combine.

## Where the lift came from

TRAIN solo pass-rates (gated, 30-day window, $25K rules):

| strategy | solo TRAIN pass-rate | static weight | inverse-vol (prior) |
|:--|---:|---:|---:|
| ORB | **44.1%** | 0.483 | 0.196 |
| MeanRev | 14.5% | 0.159 | 0.662 |
| OvernightDrift | 32.7% | 0.358 | 0.142 |

The previous inverse-vol weights gave MeanRev 66% of the ensemble
PURELY because its gated stdev was small. The new weights notice
that ORB and OvernightDrift are the ones actually clearing 30-day
windows on TRAIN, and reweight accordingly.

## The Pareto trade-off

Higher mean PnL ($27K vs $13K total over TEST) AND higher MaxDD
($11K vs $4K). The new ensemble is more aggressive: it clears the
$3K target on $50K more often but uses up more of the $1500 MLL on
$25K. $25K pass45 dropped 3.2pp as a result.

This is a genuine Pareto frontier choice between the two rule sets,
not a free lunch.

## Walk-forward vs static

Walk-forward variant tracks the same weights as static (mean
ORB=0.45, MR=0.10, OD=0.45) -- the TRAIN solo pass-rates are a
stable signal across the 2-year TRAIN window. WF gave slightly
higher Sharpe (+1.08 vs +1.03) and slightly higher $50K pass30
(12.8% vs 11.9%) but slightly lower $25K pass30 (20.8% vs 24.0%).
The two variants are roughly Pareto-equivalent; static is simpler
and more interpretable.

## Drop-one sensitivity (diagnostic, NOT a recommendation)

| variant (static) | Sharpe | $50K p30 | $50K p45 | $25K p30 |
|:--|---:|---:|---:|---:|
| Drop ORB | +1.19 | **25.1%** | **30.0%** | 27.3% |
| Drop MeanRev | +1.00 | 13.6% | 22.2% | 21.8% |
| Drop OvernightDrift | +0.21 | 7.6% | 15.0% | 18.1% |

"Drop ORB" reaches $50K pass30 = 25.1% because removing the largest-
weighted component (which had high TRAIN pass-rate but mediocre TEST
OOS) shifts weight to OvernightDrift, whose TEST OOS strongly
outperformed its TRAIN solo pass-rate. This is pure post-hoc
selection; it would be statistically dishonest to declare it the
recommendation.

## DSR

  * 9 trials this pass
  * STATIC: raw Sharpe 1.025, DSR 0.7421
  * WF:     raw Sharpe 1.082, DSR 0.7713

Up from inverse-vol fixed's 0.7211. Still below the 0.95 threshold,
but the trend is encouraging.

## Project-wide best pass-rates so far

After all the sizing/gating work:

| Combine | best pass30 | best pass45 | how |
|:--|---:|---:|:--|
| $50K  | **12.8%** | **23.9%** | WF / Static pass-rate-aware sizing on full 3-strategy gated ensemble |
| $25K  | **28.2%** | **33.3%** | baseline inverse-vol gating + literature gates |

The $50K result is the clearest empirical win from this phase 5
work; the $25K result remains the inverse-vol baseline.

## What we have NOT done (still on the table)

  * **Kelly-fractional sizing with MLL hard cap.** More math-heavy
    than pass-rate-aware; would directly respect the rule geometry
    instead of approximating it. Likely a marginal lift over what
    we have now.
  * **Add a 4th component** (Paper 2's VWAP-pullback shape, causally
    implemented). Would broaden the ensemble at the cost of more
    code; expected payoff is unclear without running.
  * **Multi-asset rotation.** ES + NQ + GC on the same ensemble
    structure. Most likely real-world lift but biggest plumbing
    effort.

## What we should NOT do

  * Re-tune the lookback window (252) -- it's pre-committed.
  * Sweep over the weight exponent -- ditto.
  * Cherry-pick "drop ORB" because its TEST numbers are higher.
  * Switch the weight criterion to $50K Combine (currently $25K)
    just because the headline lift was on $50K. That would be a
    post-hoc fit to the very metric we're trying to validate.
