# Pre-registration 10 (committed BEFORE the runs): hybrid funded sizing - volatility rule before the first payout, Kelly after it

Context. Pre-registration 9, Part C tested Kelly sizing for the whole funded stage; no setting passed the "not greedy" rule. The near misses (Kelly 3x/1x, 4x/1x) raised the mean and shrank the worst farm but paid fewer accounts and delayed the first payout by about six months, because the trailing-mean gate keeps the size at 1 micro when the trailing mean is not positive. The first-payout phase is therefore left exactly as it is (current volatility rule, 1.2 x cushion / ATR), and Kelly is applied only AFTER the first payout, where the account has already paid once and the question is how hard to push the cushion that is left.

## What changes
- Stage 1 (until the first payout): unchanged, `a1 = 1.2` x cushion / ATR, first payout at balance >= $2,000 (balanced preset).
- Stage 2 (after the first payout): contracts = f2 x cushion x mu / sigma^2 with mu and sigma^2 per micro over the previous 120 sessions (causal; 1 micro when the trailing mean is not positive), the same unit stream as pre-registration 9 (`kelly_sizing.kelly_unit`).
- Engine: `run_funded_firm(..., atr2=None)` takes an optional stage-2 sizing unit; `FirmPlan.funded_atr2` passes it. Defaults reproduce every earlier result exactly (tested).
- Combine sizing, payout rules, costs, the Topstep default split (3 noise-area + 2 gap-VWAP accounts, one Combine open at a time), RTH-only trading and dev-only selection (farms ending <= 2022-12-31) are unchanged.

## Trials (3, all declared here)
f2 in {1, 2, 3} x full Kelly. Compared with the current rule (a1 = 1.2, a2 = 0.8 x cushion / ATR). No other cell is run for selection. (Part C already used 12 looks; this brings the Kelly family to 15, which is reported.)

## Metrics and acceptance (same rule as pre-registration 9, applied to the new cells)
Ten 3-year dev farms (start years 2011..2020, Topstep default split). A cell replaces the current setting only if ALL hold:
1. dev mean net rises AND dev 25th-percentile net rises;
2. the worst dev farm is not more than 25% deeper;
3. the share of funded accounts ever paid does not fall by more than 5 percentage points;
4. (new) the median days from the first Combine to the farm's first payout is not more than 10% longer.
Among cells that pass, the highest dev p25 is chosen; if none passes, the current rule stays. Reported, not gating: payouts per paid account, number of payouts per paid account, share of funded accounts that die, seen starts (2021, 2023, 2025-01, 2025-07), full-history net.

## Checks after the run (descriptive; do not select anything)
- Edge-removed placebo (dev-fitted mean edge removed) for the current rule and the best hybrid cell.
- Stage 1 sanity: with `f2` set so that stage 2 equals the current stage-2 unit, the result must equal the current rule exactly (unit test).
- If a cell passes: the same cell with Kelly windows 60 and 250 sessions (stability only) and the self-funded pipeline (section 20, Part B) with the cell in place of the current rule.

## Honest limits
Dev farms are 10 overlapping 3-year windows (not independent); the 2023+ years have been looked at many times; the hybrid is a post-hoc idea following the Part C near misses, so even a pass is "plausible, not proven" and would need a frozen forward test.
