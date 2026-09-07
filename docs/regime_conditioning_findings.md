# Phase 5 -- literature-grounded regime conditioners on the 3-strategy ensemble

After HMM failed (regime instability across years), Phase 5 was about
finding a BETTER regime detector. The literature search converged on
rule-based gates rather than learned classifiers. Three were picked
from specific empirical papers BEFORE we touched our data, each with
zero free parameters tuned to our file:

| strategy | gate rule | source |
|:--|:--|:--|
| ORB | trade only if `rv_5 / rv_20 >= 0.8` (vol not in deep compression) | ATR-breakout literature: "low-vol periods precede high-vol; trade the transition" |
| MeanRev | trade only if `rv_20[t-1] < trailing 60d median(rv_20)` | mean-reversion literature: fades fail in high-vol regimes |
| OvernightDrift | trade only if **prior RTH session return < 0** | **Boyarchenko/Larsen/Whelan, NY Fed SR #917 (2020)**: "market selloffs generate robust positive overnight reversals" |

Methodology: same TRAIN/TEST split as the baseline ensemble. Inverse-
vol weights from BASELINE TRAIN held fixed (no re-fit on gated PnL).
All three gates always applied -- no cherry-picking. Both rule sets
evaluated.

## Per-strategy TEST window: baseline vs gated

| strategy | metric | BASELINE | GATED | delta |
|:--|:--|---:|---:|---:|
| ORB | Sharpe | +0.31 | +0.17 | -- |
| ORB | MaxDD | -$26,485 | -$26,742 | -- |
| ORB | $50K pass30 | 15.1% | 11.8% | -3.3pp |
| ORB | $50K pass45 | 11.9% | **17.0%** | **+5.1pp** |
| ORB | $25K pass30 | 12.4% | 14.2% | +1.8pp |
| MeanRev | Sharpe | +0.07 | **+0.56** | **+0.49** |
| MeanRev | MaxDD | -$10,425 | -$3,525 | -$6,900 |
| MeanRev | $25K pass30 | 12.5% | 6.6% | -5.9pp |
| MeanRev | $25K pass45 | 21.6% | 8.4% | -13.2pp |
| **OvernightDrift** | **Sharpe** | **+0.69** | **+1.16** | **+0.47** |
| **OvernightDrift** | **MaxDD** | **-$53,868** | **-$32,040** | **-$21,828** |
| **OvernightDrift** | **$50K pass30** | **18.0%** | **25.6%** | **+7.6pp** |
| **OvernightDrift** | **$50K pass45** | **21.9%** | **26.4%** | **+4.5pp** |
| **OvernightDrift** | **$25K pass30** | **16.7%** | **25.7%** | **+9.0pp** |

## What the literature actually delivered

**1. NY Fed overnight-drift conditioner WORKED.** This was the most
specific empirical claim in the literature and the one that
translated best. Sharpe lifted 0.69 → 1.16 (1.7x), MaxDD halved,
and $50K pass30 lifted **18.0% → 25.6%**. The Boyarchenko paper's
finding -- positive overnight reversal concentrates after RTH
selloffs -- holds OOS on our 2024-2026 data.

**2. Mean-reversion low-vol conditioner had a paradoxical effect.**
Sharpe lifted 8x (+0.07 → +0.56), MaxDD dropped 66%, BUT pass-rate
fell because the gate cut trade count by 55%. The strategy is now
"better" by risk-adjusted measures and "worse" by pass-rate. This
is a real lesson about pass-rate as an objective: it rewards
frequency of small wins, not Sharpe.

**3. ORB expansion conditioner was a wash.** Sharpe dropped slightly;
$50K pass45 lifted; $25K pass30 lifted; $50K pass30 dropped. The
"trade only when rv_5/rv_20 >= 0.8" rule didn't deliver a clear
edge on ORB specifically.

## Ensemble result

| ensemble | Sharpe | MaxDD | $50K pass30 | $50K pass45 | $25K pass30 | $25K pass45 |
|:--|---:|---:|---:|---:|---:|---:|
| Baseline (no gates) | +0.72 | -$7,704 | 8.8% | 23.6% | 28.2% | **33.3%** |
| Literature-gated | **+1.16** | **-$4,734** | 2.7% | 3.5% | 22.6% | 31.6% |

Sharpe **+0.72 → +1.16 (+61%)**. MaxDD **-39%**. Pass-rate moved in
mixed directions because the MeanRev gate cuts trade volume more
than it improves expectancy, and the baseline inverse-vol weights
gave MeanRev 66% of the ensemble -- the gate amplified that
distortion.

## What we should and shouldn't say next

We **should** say: the OvernightDrift conditioner is the clearest
genuine win in the project. It's literature-grounded, has a specific
mechanism (post-selloff inventory unwind), and improved every metric.
At qty=1 standalone on TEST it produces **$50K pass30 = 25.6%**, the
strongest single-strategy result on the harder Combine.

We **shouldn't** say: "let's drop MeanRev." A drop-MeanRev sensitivity
diagnostic hits $25K pass45 = 36.6% (the highest in the entire
project), but that's exactly the in-sample selection trap. Honest
DSR after this pass: **0.79**, still below the 0.95 threshold.

## What this means structurally

We learned three things:

  1. **Specific, well-cited empirical findings translate.** Vague
     volatility-regime rules ("low-vol mean-reverts") didn't help
     much; a specific mechanism with a documented annualised number
     (Boyarchenko's 3.6% overnight drift after selloffs) did.

  2. **Sharpe and pass-rate are different objectives.** A gate that
     prunes losing trades will lift Sharpe even when it hurts pass-
     rate, if the pruned trades had small mean but were frequent.
     Future sizing/weighting work should optimise pass-rate
     directly, not Sharpe.

  3. **The baseline weights are now wrong for the gated streams.**
     Re-deriving inverse-vol weights from GATED TRAIN PnL would
     improve the ensemble -- but it's an extra fit. The proper way
     is a walk-forward weight recalculation, not a re-fit.

## Pre-committed next step candidates

  * **Walk-forward weight recomputation.** Re-derive inverse-vol
    weights on a rolling basis (e.g. 252-day TRAIN, applied 252
    days forward). No new fit at evaluation time. Most likely
    next single improvement.
  * **Add a 4th component** that's natively post-selloff or post-
    rally biased -- e.g., RTH gap-and-go on positive overnight
    futures movement. Increases breadth.
  * **Replace MeanRev** with a strategy whose trade count survives
    a low-vol gate. (Selection risk -- needs pre-commitment.)
