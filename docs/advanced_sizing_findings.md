# Vol-targeting and Kelly-fractional sizing: both null results

Follow-up to `path_aware_sizing_findings.md`. Tested the two remaining
classical sizing rules on the 9-stream multi-asset gated ensemble. Same
streams, same gates, same pass-rate-aware weights. PROPER TopStep
rules throughout (`simulate_combine_window`).

## Headline

| variant | $50K p45 | Sharpe | vs baseline (p45) |
|:--|---:|---:|---:|
| **Baseline (qty=1)** | **44.2%** | **+1.66** | (reference) |
| 0.25 × k* (quarter-Kelly) | 0.0% | -1.66 | -44.2pp |
| 0.5 × k* (half-Kelly) | 0.0% | -1.66 | -44.2pp |
| 0.75 × k* | 0.0% | -1.66 | -44.2pp |
| 1.0 × k* (full Kelly) | 0.3% | -1.66 | -43.9pp |
| vol-target L=10 | 41.3% | +2.03 | -2.9pp |
| vol-target L=20 (main) | 41.9% | +2.11 | -2.3pp |
| vol-target L=60 | 42.6% | +2.11 | -1.6pp |

## Why Kelly failed catastrophically

The pass-rate-aware weights were derived to maximise SOLO pass-rate on
TRAIN, not to maximise mean PnL. Because TRAIN had limited tradeable
opportunities for some streams (e.g., GC/MeanRev with 5% TRAIN pass30),
the weighted ensemble's TRAIN mean came out NEGATIVE:

  * mu_TRAIN = -$3.29/day  (sigma_TRAIN = $426.91/day, Sharpe -0.12)
  * mu_TEST  = +$83.14/day (sigma_TEST  = $795.80/day, Sharpe +1.66)

Kelly's full multiplier:

    k* = mu * B / sigma^2 = -3.29 * 50000 / 426.91^2 = -0.902

Kelly literally said "**short the strategy**" based on TRAIN. Applied
to TEST, this inverted the sign of every day's PnL and produced near-zero
pass-rate across the board. Even with a fractional 0.25 × k*, the
strategy ran backwards.

This is not a flaw of Kelly — it's a flaw of using Kelly with parameters
estimated on data that doesn't match the deployment distribution. **The
regime shift between TRAIN and TEST broke the parametric assumption.**
Vol-targeting (non-parametric, only uses recent vol) didn't fail this
way for the same reason.

## Why vol-targeting failed (more interestingly)

Vol-targeting raised SHARPE (+1.66 → +2.11) but lowered PASS-RATE
(44.2% → 42.6% best). This is the cleanest demonstration of the
objective-mismatch finding from earlier phases:

  * **Vol-targeting succeeds at its actual objective**: reducing
    variance contribution from high-vol regimes. Sharpe is exactly what
    that wins — and it did.
  * **Pass-rate is NOT Sharpe.** Pass-rate is a non-linear function of
    per-day MEAN. The vol-targeted series has effective mean scaling of
    0.80-0.99 (because the trader is below 1.0 on most days). Lower
    mean × same MLL distance → fewer windows clear the target in 30/45
    days.

The headline number — **Sharpe +2.11 is the project's highest, but it
came with a pass-rate decrease** — is the cleanest piece of evidence we
have that the Combine objective is fundamentally not the Sharpe
objective and that no smooth variance-reduction trick will solve it.

## DSR

  * Trials added this pass: 8
  * Cumulative project trials: ~27
  * Winner (highest $50K p45): vol-target L=60 (42.6%)
  * Winner raw Sharpe: 2.110
  * Expected-max under null: 2.841 (DSR=0.12 — NOT significant)

The DSR penalty is now severe because we have accumulated 27 trials over
the project. Even the strong vol-target Sharpe of 2.11 is below the
selection-bias ceiling of 2.84. This is the discipline tax of being
honest about how many things we've tested.

## What this teaches us

1. **Kelly with parametric inputs is brittle under regime shift.** Our
   TRAIN/TEST split crossed a meaningful market regime change (mu went
   from -$3 to +$83). Parametric methods amplify TRAIN-side noise into
   TEST-side disasters. Non-parametric or robust-statistics approaches
   would have safer behavior here, but none are pass-rate-targeted.

2. **The Sharpe-vs-pass-rate gap is now measured.** Vol-targeting buys
   Sharpe at the cost of pass-rate. We CAN make the strategy look
   better by classical metrics — and choose not to, because those
   metrics are not the target.

3. **No size-only rule beats baseline.** Five sizing approaches tested
   across two writeups (symmetric path-aware, defensive path-aware,
   constant multiplier from Kelly, three vol-targeting lookbacks):
   none improve $50K pass45 over the simple qty=1 baseline. **The
   right move on the sizing axis is to LEAVE IT ALONE.**

## What remains to test (and what doesn't)

Genuine unexplored axes:

  * **Add a 4th asset (CL crude or 6E euro FX).** BLOCKED on data
    acquisition. Empirically the only axis that has lifted pass-rate
    (+20pp from ES→ES+NQ+GC).
  * **Longer history.** BLOCKED on data. Would lower DSR penalty.

Axes I will NOT keep iterating on:

  * Further sizing tweaks. Five attempts, none improved pass-rate. The
    MLL geometry binds; the Sharpe-vs-pass-rate gap is structural.
  * Strategy parameter retuning. Would burn DSR trials without
    obvious upside.
  * Regime gate retuning. Already literature-grounded; retuning would
    be reverse engineering.

## Honest conclusion

**The pipeline's current ceiling is $50K pass45 = 44.2%** with
DSR=0.61, on the multi-asset gated ensemble with pass-rate-aware
weights and uniform sizing. Crossing 50% on the existing data is not
achievable through algorithmic changes within the anti-overfit
discipline. Crossing 50% requires either:

  1. New independent return streams (4th asset, or genuinely
     decorrelated strategy shape), or
  2. Longer history to deflate the DSR penalty enough that we can
     credibly claim significance for the existing 44.2%.

Both are data-acquisition tasks. The pipeline itself is done.
