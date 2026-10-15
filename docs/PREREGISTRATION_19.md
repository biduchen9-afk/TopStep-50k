# Pre-registration 19: forward test v3 (supersedes the kill rules of PR18; written 15 October 2026, before day 0 and before any post-2026-07-02 session is counted)

Origin: Council 2 (`docs/COUNCIL_2.md`, seat memos in `docs/council_2/`). PR18 as written falsely stopped a full-edge strategy 20% of the time by month 12 and 31% by month 36 and could not confirm anything; the council found that the Combine pass rate cannot test the edge (a risk-matched zero-edge null passes 17-22%, a shape-matched one 2-3%; FINDINGS 37). This registration replaces PR18's rules with a calibrated cumulative-R kill line, drops the pass-count test and the coin-flip benchmark, and adds data-validity gates learned from stress-testing the live path. The user's decisions stand: no purchases or extra cash, no real-money step is part of this test, one Combine at a time.

## What is frozen (unchanged from PR18)
The strategy and model: the sleeves `nq_noise`, `nq_gap30` (and the shadow sleeve `nq_vwap`, below) as in `final_strategy.py`, the default farm of 3 noise-area + 2 gap-VWAP slots, one Combine open at a time, Combine size round(9,600 / ATR$) micros, funded size 1.2 x cushion / ATR before the first payout at a $2,000 balance and 0.8 after, flat by 15:55 ET, entries from 09:30 ET, cost $1.22 + 1 tick per micro round trip, the Topstep 150K rules as modelled (`firm_engine.topstep_150k`, rules v2). The code is the repository at the commit containing this file. **No parameter, rule, size or setting may change during the test; a change starts a new, separately registered test.**

## Clock and data
* **Day 0** is the first session whose 18:00 ET open the harness saw live. Start the harness with `--fresh` before the 18:00 ET open of the first session (intended: Sunday 18 October 2026, session of Monday 19 October). A session that began without the harness running is a hole session and the clock is restarted with `--fresh`.
* **Source:** the live LSE tick stream for `NQ.F` (dataset `futures`), built into one-minute bars by `paper/tickfeed.py`; vault candles are used only for the catch-up window the harness itself documents and are never patched into a session afterwards. Holes are never filled; a full-session replay inside the vendor's 23-hour window is the only repair.
* **Day-0 checks (all must pass before the clock starts):** the tick carries a size or volume field and `volume_basis` reports it (a feed with no size field is a NO-GO, because the VWAP sleeves weight by volume; changing the volume basis is a new registration); timestamps parse as UTC and the replay flag exists; sixty consecutive live regular-hours minutes built from ticks agree with the vendor's candles (close within one tick in at least 95% of the minutes); a restart in mid-session reproduces the journal.
* **Hole sessions:** a session whose regular hours (09:30-15:55 ET) have missing minutes (the journal field `data_gap`); calendar early closes at 13:00 and 13:15 ET are complete sessions and are not holes.
* **Gate at session 20:** two or more hole sessions among the first 20 end the test as **DATA FAILURE**, a feed problem and not a verdict on the strategy. From session 63 a look is **INVALID** (no kill is evaluated) when more than 2% of the counted sessions are hole sessions; **two INVALID looks in a row end the test as DATA FAILURE**. Flagged sessions stay in the sums.

## The rule (`paper/verdict.py`, `python -m paper.verdict --state paper_state.pkl`)
* **R** = a sleeve's session P&L per micro divided by the ATR in dollars per micro; **pooled R** per session = (3 R_noise + 2 R_gap) / 5, the farm's account mix. t = sessions counted since day 0.
* **Looks:** every 21 sessions (a month) from session 63 (month 3) to session 756 (month 36).
* **KILL:** at a valid look k, cumulative pooled R <= -5.75 + 0.009 k (-5.18 at 63, -3.48 at 252, -1.21 at 504, +1.05 at 756). Latched. A KILL means *stop and investigate*: stop adding research or money, and look at why. It is not proof that the edge is dead.
* **CONFIRMED (36 months):** only at session 756, valid, never killed, cumulative pooled R >= 10.89 (one-sided z >= 1.645 against zero edge at the development sd of 0.2407 per session). Anything else at 756 is **CANNOT TELL**. "Confirmed" means supported at the 5% level for the development period's edge being at least roughly alive; it does not authorise scaling.
* **WATCH (informational, never a stop):** pooled R below the PR18 development 5th percentile of equal-length windows (weekly, sessions 60 to 250; `results/forward_thresholds_R.json`), or both sleeves negative after session 120.
* **Not a rule:** the Combine pass rate, the farm's net, and the first payout are reported on the dashboard for information and carry no decision weight (null range 2-22%; a first payout is weak evidence).

## Operating characteristics (`run_pr19_oc.py`, `results/pr19_operating_characteristics.csv`; 20,000 block-bootstrap paths of re-based development R, 21-day blocks, edge scaled to a share of its development mean)
| true edge, share of the development edge | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| KILL by month 3 / 6 | 0.1 / 3.7% | 0.1 / 2.0% | 0.0 / 1.3% | 0.0 / 0.6% | 0.0 / 0.3% |
| KILL by month 12 | 22.3% | 13.9% | 8.0% | 4.2% | 1.9% |
| KILL by month 24 | 53.4% | 36.4% | 22.1% | 11.5% | 5.3% |
| KILL by month 36 | 70.5% | 51.0% | 32.0% | 16.7% | 7.3% |
| CONFIRMED at month 36 | 4.1% | 11.5% | 25.1% | 45.3% | 66.4% |
| CANNOT TELL at month 36 | 25.4% | 37.5% | 42.9% | 38.0% | 26.4% |
Reading: a KILL is a strong signal (a full-edge strategy is killed 1.9% of the time by month 12, a dead edge 22%, and 70% by month 36); **not being killed says little** (a dead edge survives 78% of the time to month 12), and a half edge is confirmed only a quarter of the time. With the council statistician's prior (dead 15%, 25% 23%, 50% 22%, 75% 20%, full 20%) a KILL by month 12 (9.5% of runs) raises the probability of a dead edge from 15% to about 35%, a CONFIRM at month 36 (31% of runs) lowers it to about 2%, and the 35% of runs that end undecided leave it at about 11%. The simulation assumes the development R distribution, stationarity and independence beyond 21-day blocks; the hold-out has been looked at many times, so the development data are the only reference. Cumulative pooled R is the only decision statistic.

## Shadow columns (descriptive only, no decision)
The harness journals a third sleeve, `nq_vwap` (no account uses it; it trades the same bars), so the verdict can report the R of VWAP and of a frozen equal-weight basket (R_noise + R_vwap + R_gap) / 3. They exist to learn whether a basket would have insured against picking the wrong sleeve; they cannot trigger or cancel anything.

## What this test cannot do
Measure real fills, real slippage or the daily-loss-limit liquidation fill (the sleeves fill at the next bar's open); measure Topstep's discretionary actions; confirm a half edge; say anything about the edge before 2026-10. The 75 sessions already seen on vault candles (INVALID, 24% hole sessions) are not part of this test. The chance of a valid 120-session verdict is estimated at about 35% (council ops seat) because the vendor's missing minutes may be upstream of the tick stream.

## Operation
`python -m paper.run --source ticks --symbol NQ.F --dataset futures --warm-from-lse 60 --fresh --port 8765` with `LSE_API_KEY` in the environment (see `paper/RUNBOOK.md`). Weekly or on demand: `python -m paper.verdict --state paper_state.pkl`. Calendar if day 0 is session 1 on 19 October 2026: session 20 about 13 November 2026; first look (63) mid-January 2027; month 12 about October 2027; month 36 about October 2029. Back up the state file and `paper_state_bars.csv` nightly (never into the public repository).

## Deliverables and limits
`paper/verdict.py`, `run_pr19_oc.py`, `results/pr19_operating_characteristics.csv`, tests in `test_paper.py` (kill line, latch, confirm, data rules, shadow sleeve parity with the batch back-test). Known open defect not fixed by this registration: the roll guard covers only a fixed window and can flag a 0.10% news-day jump as a roll (council ops seat); roll-flagged sessions stay in the sums at R = 0.

## Amendments
None yet. Any amendment must be dated, state what had been seen, and cannot alter the kill line retroactively.
