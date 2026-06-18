# Replication of the two ChatGPT-generated NQ strategy papers

Both papers were tested on the actual NQ 1-minute file from the
project's data release (2022-04-07 to 2026-04-10, 1,406,058 bars).
The implementations in `src/topstep50k/strategy/volume_profile_mr.py`
and `vwap_std_fractal.py` follow the papers' written rules faithfully
but with **strictly causal session statistics** — running VWAP, running
volume-weighted STD (paper 1) and running expanding-STD-of-hlc3 (paper
2). No future bars are used at any decision point. Engine fills at
next-1-min-open (no same-bar cheating).

## Headline: neither paper's edge replicates

### Paper 1 — Volume-Profile Mean Reversion (5-min)

| metric | paper claim | causal replication |
|:--|---:|---:|
| Trades | 7,813 | **18,178** |
| Win rate | **66.8%** | **33.9%** |
| Avg win (pts) | +38.3 | +22.0 |
| Avg loss (pts) | -16.8 | -14.2 |
| Profit factor | **4.59** | **0.79** |
| Net PnL | +156,232 pts ≈ **+$3.1M** | **-35,081 pts ≈ -$701K** |
| Sharpe (daily $) | ~15+ (inferred) | **-4.58** |
| Max DD | ~-$6.2K | **-$703K** |

The strategy is a **loser** in causal form. Win rate 33.9% on this
geometry (target = frozen VWAP, stop = 2σ from VWAP) is consistent
with a NO-EDGE fade against the prevailing 2022-2026 NQ trend.

### Paper 2 — VWAP + StdDev + Williams Fractal (15-min, RR=3:1)

| metric | paper claim | causal replication |
|:--|---:|---:|
| Trades | 212 | **33** |
| Win rate | **69.8%** | **24.2%** |
| Avg win (pts) | +21.5 | +28.8 |
| Avg loss (pts) | -7.65 | -9.15 |
| Profit factor | **2.81** | **1.01** |
| Net PnL | +$53,895 | **+$30** |
| Sharpe (daily $) | **~12.58** | **0.03** |

A **24.2% WR with RR=3:1 is exactly the geometric null** — if entry
direction were random, target=3×stop ⇒ WR=25%. My causal version
returns the null. The paper's 69.8% is 45 percentage points above
random, which on a 3:1 setup is implausible without information leak.

## Where the inflation likely comes from

The strategies' rules are NOT inherently flawed — both market-profile
mean-reversion and VWAP-pullback-with-fractal-confirmation are
established setups in the literature. The issue is the ChatGPT-
generated implementation's likely look-ahead. My current suspects,
ranked:

1. **Whole-session VWAP / STD instead of running.** This is the
   classic ChatGPT bug. If the reported VWAP at every intraday
   decision uses ALL bars in the day (including future ones not yet
   observed at decision time), the strategy effectively knows where
   price will revert TO. Paper 1's "Compute the VWAP using 5-minute
   close prices weighted by volume" is silent on running vs whole-
   session, and Paper 1's MR strategy is exactly the kind that would
   massively benefit from this leak. Win-rate inflation from ~34% to
   ~67% is roughly what whole-session VWAP would produce.

2. **Sharpe computed on trade days only.** Paper 2's "Sharpe ≈ 12.58
   using daily PnL" — with 212 trades over ~1,000 trading days, if
   the daily PnL series is restricted to the ~212 days that had
   trades, the mean/stdev ratio is artificially elevated. A
   properly-computed all-calendar-day Sharpe on the same trades
   should be 3-4x smaller.

3. **Williams fractal "confirmed two bars later" used directly at T.**
   I tested this explicitly with a leaky variant (`scripts/
   diagnose_paper2_lookahead.py`). It produced MORE trades (101 vs
   33) but **WORSE** WR (20.8%) — so the fractal bug is NOT the
   primary inflation source. Eliminated.

4. **Stop/target same-bar resolution.** Both papers assume "stop is
   hit first if both stop and target are in the same bar." That's
   conservative for the strategy — it can only HURT win rate, not
   help it. So this isn't the bug.

## What I am NOT doing next

- I will not iterate on parameter tweaks until the strategy "matches"
  the paper. Backing into the claimed numbers by adjusting band
  thresholds, fractal definitions, or proximity tolerances is exactly
  the overfit / reverse-engineering trap. Once I have a clean causal
  implementation, the answer is what it is.

- I will not import ChatGPT-generated trade lists or PnL series as
  "validation." The only valid replication is rule-text -> code -> data.

## What we LEARNED that is still useful

- The strategy **logic** in both papers is implementable cleanly and
  produces a well-defined null behavior. They can serve as
  **template** strategies for future iteration if we add real edge
  (e.g., regime gates, asymmetric thresholds, multi-asset overlay).

- Paper 2's structure (trend bias + pullback to band + fractal
  confirmation) is the same shape as "VWAP-pullback" strategies in
  the broader literature, which DO have modest documented edge
  (Sharpe ~0.5-1.0). The Sharpe-12 claim is the outlier; the
  underlying setup is not.

- The causality test for VWAP/STD should be ADDED to our standard
  audit suite. We now have unit tests in
  `tests/unit/test_volume_profile_mr.py` that fail loudly if anyone
  re-introduces look-ahead.

## Practical next steps (your call)

a. **Use the Paper 2 SHAPE in our ensemble** (VWAP pullback + fractal,
   causal, with realistic Sharpe expectations of 0.3-0.8). It has
   different correlation structure than ORB/MeanRev/OvernightDrift
   and could add ensemble breadth.

b. **Demand the notebook.** If you can obtain the ChatGPT code, I can
   point to the exact line where the look-ahead lives. That has
   educational value but doesn't change the strategy's actual edge.

c. **Move on.** Both papers are flagged as ChatGPT-output, and
   ChatGPT backtest reports are well-known to inflate. The 4 days
   between these papers and a faithful replication is well-spent --
   we have a clean reusable VWAP-MR and VWAP-Fractal infrastructure
   and we know exactly what they're worth.
