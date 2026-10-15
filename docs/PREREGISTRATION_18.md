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

## What the test can detect (`run_forward_power.py`, `results/forward_power.csv`, 20,000 block-bootstrap paths of development R, 21-day blocks, re-based costs, WEEKLY looks at sessions 60, 65, ..., 250)
Chance that a kill rule has fired at some weekly look by session 60 / 120 / 250, by the size of the true edge relative to the development edge (rule (ii) is evaluated against the 5th percentile of equal-length development windows at each look, rule (iii) from session 120):
| true edge | stopped by 60 | by 120 | by 250 | pooled R positive at 120 |
|---|---|---|---|---|
| 100% | 3.2% | 13.9% | 23.4% | 80% |
| 75% | 4.3% | 19.4% | 32.6% | 74% |
| 50% | 6.4% | 26.9% | 44.0% | 66% |
| 0% | 11.7% | 43.4% | 66.4% | 49% |
So the rules stop a strategy with no edge about two times in three within a year, and wrongly stop one with the full development edge about one time in four. The likelihood ratio of a STOP after 250 sessions is 2.8 for 'no edge' against 'full edge'. The test is a monitor with a late alarm, not evidence of an edge. (An earlier version of this table assumed three looks only, 60, 120 and 250, which gave 16% and 57%; the document says weekly looks, which is what is evaluated and what `paper/forward_test.py` does.) The verdict is latched: a STOP at any earlier weekly look stays a STOP. If the reference data are missing the verdict is UNKNOWN, never a silent CONTINUE. Development-window percentiles in R: 5th percentile of the pooled 60-session sum -2.09 R, of the 120-session sum -1.88 R, of the 250-session sum -0.46 R; median 1.11, 2.29, 4.67 R.

## Gates that are not measurable here
Realised cost per micro against the model's $1.72 and the liquidation fill need real fills; paper trading cannot show them (the sleeves fill at the next bar's open). No real-money pilot, purchase or top-up is planned or authorised; the gates stay open and the model's cost and fill assumptions stay labelled assumptions. Farm-level results are descriptive only (unchanged from PR15 section C, with the re-based reference distributions when available).

## Operation
`python -m paper.forward_test --bars data_new/nq_lse_1m.pkl` (report in `results/forward_test_report.json` with `verdict`, `R_units`, `kill_criteria_triggered`, `gates`); the live harness `python -m paper.run --source lse ...`. **Blocked at present:** the container cannot reach the London Strategic Edge host (`api.londonstrategicedge.com`, `data-ws.londonstrategicedge.com`), and a read-only call with the user's key was denied by the permission classifier; the host must be allowed in the environment's network settings and the key supplied as the environment variable `LSE_API_KEY` for a run that is allowed to make the call. Until bars after 2026-07-02 exist there is no forward test, only the machinery check on stand-in data (`--warm-until`).
