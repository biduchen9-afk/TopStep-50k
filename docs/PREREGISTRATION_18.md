# Pre-registration 18: forward test v2 (supersedes the kill rules and units of PR15; written 15 October 2026, before any post-2026-07-02 bar is loaded)

Council 1 (`docs/COUNCIL_1.md`) found that PR15 could not confirm anything: in raw dollars its second kill rule is unit-ambiguous (the dollar scale of the same strategy doubled over the development period), and a zero-edge strategy survives both rules most of the time. This version fixes the units, states what the test can and cannot do, and drops everything that would need real money.

## What is frozen
Unchanged from PR15: the baseline (nq_noise, nq_vwap, nq_gap30 sleeves as in `final_strategy.py`; the Topstep default farm of 3 noise-area + 2 gap-VWAP slots, one Combine at a time, Combine size round(9,600 / ATR$), funded size 1.2 x cushion / ATR before the first payout at a $2,000 balance and 0.8 after; flat by 15:55 ET; entries from 09:30 ET; costs $1.22 + 1 tick per micro round trip), the bar-by-bar engines in `paper/` (parity-tested against the back-test, `test_paper.py`), the data source and the data validity checks (1) to (5) of PR15. Rules of the engine now come from Topstep's own pages (`docs/TOPSTEP_RULES.md`). No parameter may change during the test.

## Units (the change)
All verdict-bearing statistics are in **R units**: R = daily P&L per micro divided by the ATR in dollars per micro (the unit the farm sizes on). The reference distribution is the development period (2010-06 to 2022-12-31, repaired every-session data) with the sleeves **re-charged at today's cost-to-ATR** (`retest_cost.rebase`, 1 tick), because the forward period trades at today's price level and cost structure; the dollar-cost version of the development period (cost 2.5% of ATR in 2013) is not the right yardstick for a test run at 0.2% of ATR. The pooled statistic is (3 R_noise + 2 R_gap) / 5 per day, the account-average R of the farm's mix.

## Verdicts (there is no 'confirmed')
* **INVALID**: a data check fails (reported as invalid, not as a failure of the strategy).
* **STOP** (the strategy is not confirmed; stop and investigate), any of: (ii) after at least 60 sessions the pooled cumulative R is below the 5th percentile of equal-length development windows; (iii) after at least 120 sessions both sleeves have a negative cumulative R.
* **CONTINUE** otherwise, including any result above the median or above the 90th percentile. No window, however good, is a confirmation.
Checked weekly from 60 sessions on; the test runs at least 250 sessions (about one year) before any statement beyond 'not stopped'.

## What the test can detect (`run_forward_power.py`, `results/forward_power.csv`, 20,000 block-bootstrap paths of development R, 21-day blocks, re-based costs)
Chance that the kill rules have fired by session 60 / 120 / 250, by the size of the true edge relative to the development edge:
| true edge | stopped by 60 | by 120 | by 250 | rule (iii) alone at 120 | pooled R positive at 120 |
|---|---|---|---|---|---|
| 100% | 3.2% | 12.0% | 16.1% | 9.8% | 80% |
| 75% | 4.3% | 17.0% | 24.0% | 14.5% | 74% |
| 50% | 6.4% | 23.8% | 33.9% | 20.5% | 66% |
| 0% | 11.6% | 39.9% | 56.9% | 36.5% | 49% |
So the rules stop a strategy with no edge only 57% of the time within a year, and wrongly stop one with the full development edge 16% of the time. The likelihood ratio of a STOP is about 3.5 for 'no edge' against 'full edge' after 250 sessions. The test is a monitor with a late alarm, not evidence of an edge. (Development-window percentiles in R: 5th percentile of the pooled 60-session sum -2.09 R, of the 120-session sum -1.88 R, of the 250-session sum -0.47 R; median 1.10, 2.29, 4.66 R.)

## Gates that are not measurable here
Realised cost per micro against the model's $1.72 and the liquidation fill need real fills; paper trading cannot show them (the sleeves fill at the next bar's open). No real-money pilot, purchase or top-up is planned or authorised; the gates stay open and the model's cost and fill assumptions stay labelled assumptions. Farm-level results are descriptive only (unchanged from PR15 section C, with the re-based reference distributions when available).

## Operation
`python -m paper.forward_test --bars data_new/nq_lse_1m.pkl` (report in `results/forward_test_report.json` with `verdict`, `R_units`, `kill_criteria_triggered`, `gates`); the live harness `python -m paper.run --source lse ...`. **Blocked at present:** the container cannot reach the London Strategic Edge host (`api.londonstrategicedge.com`, `data-ws.londonstrategicedge.com`), and a read-only call with the user's key was denied by the permission classifier; the host must be allowed in the environment's network settings and the key supplied as the environment variable `LSE_API_KEY` for a run that is allowed to make the call. Until bars after 2026-07-02 exist there is no forward test, only the machinery check on stand-in data (`--warm-until`).
