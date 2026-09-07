# Path-aware sizing on the multi-asset gated ensemble

Follow-up to `multi_asset_findings.md`. Same 9 streams, same gates, same
pass-rate-aware weights. Only the per-day position-size MULTIPLIER changed.

## Hypothesis

The Combine objective is path-dependent: cumulative buffer must avoid
-MLL and reach +target within a 30/45-day window. The prior pipeline
sized every day the same (qty=1), ignoring the path. Path-aware sizing
should help by scaling DOWN when the buffer is in danger and UP when
ahead of pace.

## Result: it does NOT help. Baseline wins.

Apples-to-apples comparison using PROPER TopStep rules (trailing MLL +
consistency check, via `simulate_combine_window`):

| variant | $50K p30 | $50K p45 | $25K p30 | $25K p45 |
|:--|---:|---:|---:|---:|
| **Baseline (qty=1, no scaling)** | **30.6%** | **44.2%** | **36.2%** | **45.2%** |
| SYM  {0.5, 1.0, 1.5} (original pre-commit) | 25.0% | 35.7% | 29.0% | 37.0% |
| SYM2 {0.5, 1.0, 2.0} (aggressive attack)   | 19.1% | 29.1% | 21.6% | 30.2% |
| DEF  {0.5, 1.0, 1.0} (defend-only)         | 30.3% | 43.5% | 35.6% | 44.6% |
| DEF2 {0.25, 1.0, 1.0} (aggressive defend)  | 30.0% | 43.0% | 35.3% | 42.6% |

Thresholds were mechanically derived from rule parameters (defend if
cum <= -M/2, attack if cum >= +T/2) — no data tuning. Both symmetric
variants make things significantly worse; both defend-only variants are
statistically indistinguishable from baseline (within 1pp).

## Why path-aware sizing fails here

The TopStep trailing MLL is itself a path-dependent constraint that
already encodes the geometry path-aware sizing was trying to exploit.
Specifically:

1. **Attack mode (1.5x or 2x) ratchets the peak faster.** Once peak EOD
   balance hits a new high, the MLL anchor ratchets up by the same
   amount, capped at starting balance. Larger up-days produce higher
   peaks, which produce a tighter trail, which produces less room for
   the next drawdown. The strategy that "deserved" 1.5x because it was
   ahead now has 1.5x the equity and 1.5x the danger of a normal
   pullback violating MLL.

2. **Consistency rule punishes attack-mode wins.** Best-day PnL must be
   <= 50% of total cycle PnL. A 1.5x day disproportionately becomes
   "the best day" and forces the rest of the window to produce 2x its
   contribution — a constraint that wasn't there before.

3. **Defend mode (0.5x) doesn't help because losses were already
   bounded.** Each underlying strategy has tight stops (ORB 40 ticks,
   MR 15 ticks, OD held overnight at known max ~30 ticks). Halving
   exposure on bad days saves a few hundred dollars; the strategy
   wasn't dying from gradual bleed anyway.

The first effect dominates. The baseline (uniform sizing) is already
near-optimal under TopStep's path-dependent rules because the rule
machine itself rewards stable per-day variance.

## Outcome decomposition (illustrative, $50K pass30)

SYM variant outcomes show the MLL effect directly:

| outcome | baseline | SYM 0.5/1.0/1.5 |
|:--|---:|---:|
| pass | 213 | 174 |
| mll_breach | ~250 | **246** (similar) |
| consistency_fail | ~80 | **100** |
| no_target | ~150 | 176 |

The +24 consistency failures and +26 no-target outcomes are direct
evidence that attack-mode wins shift the win distribution toward
fewer-but-larger days — failing both the consistency rule and the
"reach target somewhat consistently" requirement.

## What this teaches us

1. **TopStep's rules are NOT separable from sizing.** The trailing MLL
   creates a coupling between current size and future room. Naive
   position sizing rules that work in unconstrained backtests
   (Kelly, anti-Martingale, pyramiding) actively backfire here.

2. **The path-dependent geometry was already exploited.** The pass-rate-
   aware WEIGHTS we derived in the prior phase already implicitly
   solved for "low-drawdown high-mean" sizing at the cross-section
   level. There is no additional benefit from a within-window dial.

3. **Defense doesn't help when losses are bounded.** Strategies that
   already have tight stops do not get safer by scaling down. Defense
   helps when underlying losses are unbounded (e.g., martingale,
   leveraged carry); ours aren't.

## DSR

Counted 5 new trials (4 variants + the build-scaled time series for
DEF). Cumulative project trials: ~19. DEF's "winner" DSR of 0.99 is
misleading — DEF's pass-rate is WORSE than baseline by 0.5-1pp; the
DSR was computed against a Sharpe-only criterion which doesn't reflect
the path-dependent objective. Honest reporting: this is a null result.

## Conclusion: path-aware sizing is not the unlock

The remaining gap (44.2% → 50%+) won't come from sizing tweaks on the
current 9 streams. The MLL geometry is the binding constraint. The
realistic paths forward:

1. **More independent streams.** Each new uncorrelated stream raises
   ensemble mean without raising per-day variance proportionally. The
   ES→ES+NQ→ES+NQ+GC progression added 20pp of pass-rate; another
   asset (e.g., CL, ZB) plausibly adds another 3-5pp.

2. **Longer history.** Lowers DSR deflation penalty without adding
   trials. Currently 4 years; 10 years cuts the penalty by ~1.6x.

3. **Accept the ceiling.** 44.2% pass45 on $50K with DSR ~0.61 is a
   defensible, honest result. The pipeline is anti-overfit by
   construction.

## Pre-committed next options (none more compelling than the others)

- **Volatility targeting at portfolio level.** Single hyperparameter
  (target daily stdev). MIGHT help because it normalises across regime
  changes, but per the path-aware lesson the MLL trail will probably
  punish high-vol days regardless of normalisation.

- **Kelly-fractional sizing.** Math-heavy, more knobs to defend,
  and shares the same coupling-to-MLL problem.

- **Add a 4th asset (CL or 6E).** Most promising. Continues the
  observed multi-asset scaling.
