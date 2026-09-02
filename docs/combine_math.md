# What the $50K Combine math actually requires

Written after the May 2026 ORB / HMM exploration. This is the napkin
math + the implications for strategy selection.

## The rule set, expressed as a constrained optimisation

Single Combine attempt:

  * Start: $50,000.
  * **Profit target**: reach $53,000 EOD at some point in the cycle.
  * **MLL** (trailing): never let EOD equity touch
    `max(EOD_equities) - $2,000`, where the trail starts at $48,000
    and locks at $50,000 once it climbs that far.
  * **Consistency**: at the moment you cross $53,000, the best winning
    day must be `<= 50%` of cumulative cycle PnL.
  * **Pass window**: 30-45 calendar days (~21-32 trading days).
  * Position cap: 50 micro-equivalents account-wide.
  * No daily-loss limit on the Combine.

To pass, you need to produce a daily-PnL series with all three
properties simultaneously:

  1. **Cumulative > $3,000 within ~21-32 trading days.**
  2. **Path never drops > $2,000 below the running maximum EOD equity.**
  3. **No single day contributes > 50% of the total.**

## What an average-day return needs to be

If the user wants > 50% sliding-window pass-rate over a 4-year file,
the strategy needs to satisfy (1) and (2) on a majority of overlapping
windows. Translate this to mean / stdev of daily PnL:

  * For (1) at 50% pass-rate on 30 days, the median cumulative PnL after
    30 days must be `>= $3,000`, i.e. mean daily PnL `>= $100`.
  * For (2), the per-day standard deviation must be small relative to
    the $2,000 MLL. A rough rule from random-walk maths: a path with
    daily mean `mu` and daily stdev `sigma` has a 50% chance of
    drawing back > k * sigma at some point within N days where
    `k ~ sqrt(2 * log(N))`. For N=30 days that's k ~ 2.6. So we need
    `sigma <= $2,000 / 2.6 ~ $770`.

Combine these:

    target mean   : $100 / day
    target stdev  :   ~$770 / day (or smaller)

  Implied **daily Sharpe = $100 / $770 = 0.13.**

  Annualised, that is `0.13 * sqrt(252) = 2.1`.

Numerator goes up if you target a higher pass-rate. For 75% pass-rate
on 30 days the mean has to land near `$140-$170/day` — implied annual
Sharpe close to **3**.

## Sharpe 2-3 is in the top decile of trading strategies, ever

Reference points:

| strategy | annual Sharpe | source |
|---|---|---|
| Buy-and-hold S&P 500 (last 10y) | 0.7 - 0.9 | bog standard |
| Top decile mutual funds | 0.6 - 1.2 | Morningstar |
| Renaissance Medallion (legendary) | ~3.5 | reportedly |
| Most academic / SSRN strategies, OOS | 0.3 - 0.8 | López de Prado |

So the **minimum** Sharpe needed for the user's pass-rate criterion is
already at the edge of what known strategies achieve out-of-sample.

## ORB's actual measured Sharpe

From the 144-cell sweep on the 4-year ES file (single-asset, qty=1):

  * Best raw Sharpe across the grid: ~0.9 (long-only, fixed-tick stop,
    OR=15-30, TP=2-3x).
  * That's already cherry-picked from 144 trials. After deflating for
    selection bias (Bailey-Lopez de Prado DSR with N=144), the genuine
    Sharpe is likely `0.4 - 0.6`.
  * Translates to an annual mean / stdev ratio of `~0.5`, which means
    daily mean / stdev `~0.03`. With sigma at the ~$800 level required
    by MLL, the mean would be `~$24/day` -- not the $100-$170 we need.

## The constraints fight each other

There's an additional structural issue beyond just "need higher edge."
The three rule constraints push in incompatible directions:

  * To survive **MLL**, you want SMALL per-day variance. Argues for
    tight stops, small position size.
  * To reach **profit target** in 30 days, you need MEANINGFUL per-day
    PnL. Argues for big position size or trades with large expectancy.
  * To pass **consistency**, you need NO outlier winning day. Argues
    against tight stops with large TPs (because the few winners
    dominate) and against trend-following (where a single trend day
    can blow the 50% ratio).

A strategy that has positive expectancy (#2) and tight per-day variance
(#1) tends to be either mean-reversion (which has fat-tail LOSSES that
break MLL) or a high-frequency scalper (which violates the position cap
or needs infrastructure we don't have).

## Strategies that have a structural fit

Looking back at the four families I originally listed:

  | family | pass-rate fit | why |
  |---|---|---|
  | **ORB on ES alone** | Poor (verified ~17-20% best) | trend-day bias kills consistency; chop days kill MLL |
  | **VWAP pullback in trend regime** | Plausible | high hit-rate when filter works; needs strong regime detection |
  | **Mean reversion** | Bad | fat-tail losses break MLL |
  | **Multi-asset ensemble** | Plausible | natural distribution flattening; needs uncorrelated edges |

What I'd try next if we keep going:

  1. **Multi-strategy ensemble**: ORB (small size) + mean reversion
     with tight stops (small size) + a momentum overlay (small size).
     Each component independently sized so total daily-variance budget
     is met; consistency benefits from many small winning days from
     uncorrelated sources.
  2. **Trade only known-regime days**: instead of HMM, use a simpler
     rule -- "trade only if prior day's range > 1.5x trailing-20
     median" -- to skip chop. (The chop filter we added does this; it
     gave a tiny lift on baseline ORB but not enough.)
  3. **Reduce target**: build the strategy against a $25K Combine
     ($1,500 target / $1,500 MLL) where the per-day expectations are
     half as demanding. Then scale if it passes.

## Acceptance criteria for declaring a strategy "challenge-ready"

In light of the above, the gate that the EVGate already encodes is the
right one:

  * Deflated Sharpe Ratio > 0.95 on the OOS test fold.
  * Bootstrap pass-rate > 50% with Wilson-CI lower bound > 40%.
  * PBO across the parameter grid < 0.3.
  * Position-cap and MLL never breached more than 30% of windows in
    the realized 30/45-day evaluation.

If a candidate strategy passes all four with the audit log open for
inspection, declare it challenge-ready. Otherwise iterate.
