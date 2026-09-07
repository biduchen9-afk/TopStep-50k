# $25K Combine -- ensemble pass-rate findings

Same three strategies, same pre-committed parameters, same TRAIN/TEST
split, same TEST-window daily PnL series. Only the rule set used for
the sliding-window pass-rate evaluation changed.

## Side-by-side, TEST window (2024-04 -> 2026-04, 725 trading days)

|                          | $50K pass30 | $50K pass45 | $25K pass30 | $25K pass45 |
|:-------------------------|---:|---:|---:|---:|
| ORB alone                | 15.1% | 11.9% | 12.4% |  8.5% |
| MeanRev alone            |  1.0% |  7.8% | 12.5% | 21.6% |
| OvernightDrift alone     | 18.0% | 21.9% | 16.7% | 19.7% |
| **ALL-THREE ensemble**   |  **8.8%** | **23.6%** | **28.2%** | **33.3%** |
| drop ORB (sensitivity)   | 11.8% | 22.9% | 33.3% | 35.8% |
| drop MeanRev (sens.)     | 16.8% | 16.3% | 15.4% | 14.0% |
| drop OvernightDrift (s.) |  4.7% | 13.2% | 21.6% | 24.1% |

## What changed

  * **Headline ensemble pass30 lifted 8.8% -> 28.2%** -- a 3.2x
    improvement -- and pass45 lifted 23.6% -> 33.3%. The $25K target
    of $1,500 is reachable on far more 30-day windows than the $50K
    target of $3,000.
  * **MeanRev came alive at $25K**. Its TEST Sharpe was only +0.07
    (small edge, low stdev), so it almost never produced the $3K
    needed for $50K pass; but $1.5K target catches its slow grind.
    pass30 jumped from 1.0% to 12.5%.
  * **ORB's pass-rate FELL slightly** at $25K (15.1% -> 12.4%) -- a
    counter-intuitive result. ORB has high stdev ($1,003/day on
    TRAIN); $25K's $1,500 MLL is busted by ORB's big losing days
    before the cumulative reaches $1,500. The tighter MLL hurts
    higher-variance strategies more than the lower target helps them.
  * **OvernightDrift's pass-rate moved less** between rulesets
    because its daily PnL is bimodal (large overnight moves dominate);
    halving target and MLL together leaves the pass distribution
    similar.

## Why we still do not clear 50%

The math from `combine_math.md` for the $25K rule set:

  * Mean per day needed:  $50/day  (= $1,500 / 30 days for 50% pass-rate)
  * Stdev per day needed: $577/day (= $1,500 MLL / 2.6 from random-walk)
  * Implied daily Sharpe : 0.087  -> annual Sharpe ~ 1.4

The all-three ensemble at qty=1 has mean ~$16/day, stdev ~$320/day,
implied daily Sharpe ~0.05 -> annual Sharpe +0.72. **We are about
half the required mean.**

## Could we scale up size to close the gap?

Tempting but tricky. Doubling qty would double both mean AND stdev.
Pass-rate hinges on the ratio mean / stdev (Sharpe), which doesn't
change. Bigger position only helps if the rule's MLL distance is
proportionally bigger -- which is what happens going UP an account
size, not down. Scaling at fixed account size moves us along an
isoquant: bigger mean is offset by bigger MLL bust risk.

Empirically: at the all-three ensemble's TEST MaxDD of $7,704 over
2.25y at qty=1, that's a 30-day-window MaxDD distribution whose
upper tail already touches $1,500. Doubling size would put most
30-day windows in MLL-breach territory.

## Honest verdict for the $25K Combine

The $25K Combine is **materially more passable** than $50K at the
same edge level:

  * Ensemble pass45 = **33.3%** is the strongest TEST-window pass-rate
    we have produced in this project.
  * 1 out of 3 sliding 45-day windows clears the $1.5K target without
    busting the $1.5K MLL.

It is **not** at 50% in our setup. To cross 50% at $25K with this
ensemble we would need ~2x the per-day mean PnL, which from a
fixed-target / fixed-MLL ratio standpoint means a fundamentally
larger edge -- not just a sizing trick.

## What I am NOT doing next

Same overfit discipline as before:

  * Not picking "drop ORB" as the answer just because its $25K pass30
    (33.3%) is the highest in the table. The selection was made AFTER
    seeing the test result.
  * Not re-tuning inverse-vol weights to favour low-stdev strategies.
    The current weights came from TRAIN. Changing them post-hoc would
    leak TEST signal.
  * Not exhaustively trying every micro/mini contract permutation to
    pick the size that maximises pass-rate. That is the same trap.

## Possible next steps (your call)

  1. **Stop here.** The headline finding -- 33.3% pass45 on $25K --
     is the project's strongest result and stands as documented.
  2. **Walk-forward / bootstrap CI on the headline ensemble** so the
     33.3% is reported with a confidence interval (e.g. Wilson-CI
     lower bound) and is robust to which days are sampled.
  3. **Try a fourth, momentum-style component** to bump the mean.
     This grows trial count (DSR deflation) and risks overfit; needs
     pre-committed parameters and possibly a walk-forward holdout.
  4. **Replace ES with a portfolio of micros** (MES, MNQ, MCL, MGC)
     so the ensemble can size more finely against the $1,500 MLL.
     This is plumbing more than research.
