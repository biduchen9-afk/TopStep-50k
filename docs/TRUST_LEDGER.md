# Trust ledger (15 October 2026): which numbers of this project can be relied on, and which cannot

Written after the user asked, with no further data available: "can we still trust the entire project and what the numbers showed us?" Every figure below is quoted from a committed result file or a FINDINGS section; the only new computations are the start-year tables in section 3 (from `results/pr17_A_farms_rebased1.csv` and `results/many_starts_data_raw_live_v2_w756.csv`, no engine run). Nothing here selects or adopts anything.

## 0. Short answer
* **The machinery is trustworthy.** Signals, account engine, evaluator and cost re-basing were each re-derived by independent reviewers and matched. The Topstep rules the engine uses are now first-hand text (`docs/TOPSTEP_RULES.md`) apart from the items in section 5.
* **The measurements on the 2010-2022 history are accurate descriptions of that history**, but that history is about four independent three-year windows, and one of them (starts in 2016-2018) supplies most of the profit. The hold-out (2023 to 2026-07-02) is in section 7b: it agrees with the re-based cost level, flips which sleeve is strong, and shows farms started since mid-2024 flat to negative.
* **No dollar forecast is trustworthy.** The headline moved $17.5k, $9.3k, $12.7k, $38.8k as errors and the cost basis were corrected; the sign is set by assumptions the data cannot test (section 5).
* **Whether the edge still exists is unknown.** The 2023-26 data were looked at many times, the last 18 months are the weakest of ten windows, and the one forward look (75 sessions) is INVALID on data completeness and could not have confirmed anything anyway.

## 1. Tier A: verified (high trust)
| Item | Evidence | Where |
|---|---|---|
| No look-ahead in the three sleeves | independent event-driven re-implementation matches all 4,058 tradable days x 3 sleeves; cutting the data at every decision time changes no entry or exit (2,057 cut times, 4,988 entries) | FINDINGS 27 |
| Account engine (Combine and funded) | equals an independent reference on 18,000 random Combine starts, 18,000 funded starts and 32,960 real Combine starts | 27 |
| Costs, ATR units, timestamps, DST, fills | verified by hand / on raw data: $1.22 + 1 tick = $1.72 per micro round trip, $2.00 per point per micro | 27 |
| Evaluator (`retest.py`) | a separate reviewer reproduced all 247 farm nets exactly and the rates $24.755 / $43.409 / $29.278 per Combine session | 28b |
| PR17 engine edits, cohort accounting, regime bootstrap, R definition | reviewer: bit-identical at defaults, bootstrap reproduced to the last digit, own exact counterfactual matched ($38.8k mean, 87% positive, p25 $9.9k) | 30b |
| Edge-removed placebos lose money | the payout rules alone are not a free lunch (about -$56k in the original full-history run with the edge removed; the audit confirmed the sign, not that figure) | FINDINGS 2, 27 |
| Hedging conflict | the two groups hold opposite positions in 1 of 4,123 tradable sessions (7 of about 206,000 shared minutes) | 30 |
| Bar-by-bar live harness equals the back-test | author-written parity tests (`test_paper.py`); not separately reviewed by an independent agent | 27, 32 |
| Core Topstep rules | first-hand text: target $9,000, trailing loss $4,500 locking at $0, 150 micros, 55% consistency, optional $3,000 daily limit (forced break, not breach), $229 less $30 = $199, activation $0, payout rule (5 winning days of $150, 50%, cap $5,000, minimum $125, floor to $0, positive net profit since the last payout), up to 5 XFAs | `docs/TOPSTEP_RULES.md` |
| Zero-edge worlds rarely reproduce the baseline | baseline mean $12.7k occurs in 1.7% of zero-edge resampled histories (median $2.9k: 9%) | council statistician |

Caution on this tier: independent reviewers found real errors four times (hindsight roll-skip filter; 3:2 pooling of the paid share; `ok.head(5)` truncating the farm tests; cost cache on unrepaired data plus the cost-only re-basing). None was found by the author. "Verified" means "the reviewers who looked found nothing more", not "no error remains".

## 2. Tier B: measured on the development history, as modelled (true as measurements, conditional on 2010-2022)
Baseline 3 noise-area + 2 gap-VWAP, repaired every-session data, rules v2, dev costs ($1.72 per micro), 247 overlapping 3-year farms started every 10 sessions:
| Statistic | Value |
|---|---|
| Realised 3-year farm net: mean / median | $12.7k / $2.9k |
| p10 / p25 / p75 / p90 | -$8.5k / -$4.7k / $25.4k / $52.6k |
| Share of farms positive / worst farm | 60% / -$11.6k |
| Cohort net (every payout of Combines bought in the window): mean / median / p25 | $32.1k / $7.6k / -$4.7k |
| Calendar days to the farm's first payout (median) | 167 (p25 / p75 / p90: 89 / 302 / 432 in the council's run) |
| Funded accounts paid | 48.2% (farm level, unfinished counted unpaid); 53.8% (account level, farm mix) |
| Combine pass rate: noise-area / gap-VWAP | 21.4% / 28.8% |
| Rate (profit per Combine session): noise / gap / farm mix | $24.8 / $43.4 / $29.3 |
| Peak cash at risk: median / p90 / max | $4.7k / $10.3k / $11.6k |
| P(cash line ever below -$2k / -$5k / -$10k) | 78% / 47% / 12% |
| Strategy edge (net, daily R): noise-area | Sharpe 0.46, t 1.64 (0.59 without 2017-18); gross Sharpe 1.28 |
| Strategy edge (net, daily R): gap-VWAP | Sharpe 0.97, t 3.46, picked from 156 variants (deflated 0.57-0.82) |

Uncertainty the data themselves give (statistician, regime-resampled bootstrap of 13 calendar years): baseline mean 90% interval [-$3.6k, +$42.6k], P(mean > 0) 0.86, P(median > 0) 0.76; block-bootstrap interval of the mean [-$1.0k, +$28k]. The 247 farms are about 5 independent draws (integrated autocorrelation 49-57 starts).

## 3. Where the profit comes from (new tables from the stored farm files)
Mean 3-year farm net by calendar year of the start (each year has 25-26 farms; 2020 has 2, 2010 has 14):
| Start year | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| as modelled (dev costs) | $2.6k | $0.1k | -$5.8k | -$6.8k | -$8.1k | $10.1k | $43.0k | $54.8k | $24.0k | $8.2k | $18.8k |
| exact re-basing, 1 tick | $27.5k | $26.7k | $19.0k | $8.6k | -$6.4k | $26.3k | $113.7k | $106.8k | $47.6k | $13.9k | $26.9k |

Non-overlapping blocks (farm mean / median):
| Starts in | 2010-12 | 2013-15 | 2016-18 | 2019-20 |
|---|---|---|---|---|
| as modelled | -$1.7k / -$1.8k | -$1.4k / -$5.6k | $40.4k / $31.5k | $8.9k / $5.1k |
| exact re-basing | $23.8k / $22.3k | $9.7k / $7.3k | $89.1k / $99.1k | $14.8k / $10.0k |

Without the starts of 2016-2018 (170 of 247 farms):
| | mean | median | p25 | share positive |
|---|---|---|---|---|
| as modelled | $0.2k | -$1.8k | -$6.2k | 42% |
| exact re-basing | $16.0k | $19.8k | $3.4k | 81% |
Starts in 2016-2018 hold 72% of the summed net on the exact basis (31% / 28% / 13%) and, as modelled, essentially all of it (council: 99%). Read this as: **as modelled, the farm is profitable only if a 2017-18-type expansion falls inside the window; on the re-based cost it is positive in every non-overlapping block, but with a spread from $10k to $89k.** The shuffle test (strategy seat) says the start-year dispersion is what a stationary edge produces in a convex farm (P = 0.2-0.45), so concentration is not proof of regime dependence either.

## 4. Tier C: counterfactuals (trust only as far as their assumption holds)
Exact re-basing = the same percentage moves at today's price level (P&L, intraday low and ATR scale by 1/f, per-contract dollar costs unchanged; reference price 22,123, f = min(1, price / reference)). It assumes the percentage edge does not depend on the price level and that one tick of slippage still holds at NQ 30,000.
| Basis (3+2, 247 / 222 farms) | mean | median | p25 | positive | worst | first payout |
|---|---|---|---|---|---|---|
| as modelled | $12.7k | $2.9k | -$4.7k | 60% | -$11.6k | 167 days |
| exact, 1 tick (headline of the re-based world) | $38.8k | $22.8k | $9.9k | 87% | -$9.5k | 139 days |
| exact, 2 ticks | $37.8k | $23.0k | $7.0k | 86% | -$7.8k | 139 days |
| exact, 3 ticks | $30.9k | $24.4k | $7.8k | 82% | -$8.0k | 141 days |
Path-noise floor: about $0.5k on the farm median and $2-3k on the mean, so differences smaller than that between rows are not effects. Evidence for and against the assumption: real cheap-cost starts give +$32.5k (2021) and +$26.5k (2023) over 756 sessions, but -$6.8k for the 2025-01 start (378 sessions); costs ate 51% / 29% of the gross edge in dev and 11% / 12% in 2023+.
Rules scenarios on the exact basis (reference $38.8k / $22.8k / $9.9k): all accounts forced onto the Consistency path $6.5k / $5.3k / -$4.0k (68% positive, paid share 22% against 50% at farm level, first payout about 300 days); payouts capped at 5 per account $30.7k / $21.6k; capped at 3 $21.2k / $18.0k; $30 per payout $38.2k / $22.0k; Standard-path price $36.9k / $20.6k; XFA ladder flat at 30 micros no change; combined adverse (2-tick cost, 2-tick overshoot, $30 per payout, 5 payouts) $26.6k / $18.8k / $5.8k (87% positive). Liquidation overshoot of 0-4 ticks per contract: median $22.0-22.8k on the exact basis, but on dev costs the median goes $2.9k, $2.3k, $2.3k, -$1.7k, -$1.9k, so the as-modelled sign depends on the fill and the re-based one does not.
Split menu (exact basis): 3+2 $38.8k / $22.8k / $9.9k; 2+3 $40.4k / $16.4k / $7.6k; gap x5 $31.8k / $20.1k / $8.9k. P(beats 3+2) 0.70 / 0.66 / 0.59 (2+3), 0.48 / 0.57 / 0.54 (gap x5); neither passes the switch rule. **Reading: the data cannot say 3+2 is best, only that nothing tested is reliably better.**

## 5. What no amount of auditing can settle
1. **Price-independence of the edge** (drives the whole difference between the $2.9k and the $22.8k median).
2. **Edge persistence**: the last 378 sessions (from 2025-01-16) have a re-costed Sharpe of 0.43, the weakest of ten non-overlapping 18-month windows (z = -1.1 against the full-sample mean); decay and bad luck cannot be separated (SE of an 18-month Sharpe about 0.8). The 2023-26 data are "seen" and soft.
3. **Topstep's discretionary conduct rules** (forced Consistency path, Responsible Trading Program, account stacking, call-up to Live): the one scenario that moves the result most ($38.8k to $6.5k).
4. **Liquidation fill, real slippage and cost per micro**: need real fills; paper trading cannot show them.
5. **Rules not read in full**: the 150K XFA scaling ladder (image only on Topstep's page), FOMC-window entry rules, payout processing time, activation lag (a one-session lag cuts the paid share to 46.8% and the median days to 194 in the older model), the Terms-of-Use wording "only one Account".
6. **Selection**: about 5,000 configurations were tried over the project; the baseline layout is the user's, not a pre-registered winner.
7. **Single vendor, single container**: the 2010-2026 Databento file exists only in `/home/user/data_raw` (2.7 GB, never in git). Keep the original download.

## 6. Tier D: numbers not to use
$17.5k mean / $108k full-history net / $87k / $232.7k (hindsight roll-skip, unrepaired splices); 61.5% paid and 78 days (3:2 pooling, farm mix is about 50-54% and 55 days); "80% paid within 90 days" (not reachable at the farm mix); 96% positive and p25 $17k (cost-only re-basing, dev contract caps); the audit-selected gap cell (z 2.16; the account-level acceptance gates fire on a zero-edge grid in 29-34% of resamples) and the frontier pairs (46,656-pair dev optimum, the split-half check does not carry); any ranking of splits or sizings below about $5k; "walk-forward positive in 9 of 11 years" (pre-audit data); the old forward-test rules of PR15 (superseded by PR18).

## 7. Forward evidence (what exists)
LSE `NQ.F` one-minute candles after 2026-07-02: 75 sessions, 18 of 74 completed sessions (24%) with holes in regular hours, so the pre-registered verdict is **INVALID** (the clause allows 2%). Descriptive only: pooled R +1.82 (58.7th percentile of equal-length development windows re-charged at today's cost), noise +1.99 R, gap +1.56 R, +1.37 R without 24 flagged sessions; no STOP at any weekly look; paper farm +$2,107 (6 Combines, 2 funded, 1 payout). A zero-edge strategy has a 50% chance of a positive pooled R at 120 sessions, so this says almost nothing. The forward test can only kill: by session 250 it has stopped 23.4% of full-edge paths and 66.4% of zero-edge paths (likelihood ratio of a STOP 2.8); 80% power for the gap sleeve alone needs about 6.6 years, for noise-area about 29.
The user reports (15 October) that LSE no longer serves more futures history. This conflicts with `results/lse_history_probe_2026-10.json` (FINDINGS 33: `NQ.F` 1d, 1h and 1m candles from 2016-05-29, six sample days complete). Unverified here (the container cannot reach LSE); the probe should be repeated by the session that holds the key before anything relies on either statement.
**Correction to FINDINGS 33's last paragraph:** even if the 2016-2026 LSE minute history is complete, it is the same CME NQ price history as the Databento 2016-2026 file, so it is a second *vendor* (a data-quality and roll cross-check, and a possible warm-up source), not a second *sample* of the market. The only independent evidence is data after 2026-07-02, and it accrues one session per trading day.

## 7b. The hold-out (2023-01-02 to 2026-07-02): what it shows and how far it can be trusted
Added after the user asked why the farm tables stop at 2022. The 247 farms are 3-year windows that must fit inside the development period (dev = up to 2022-12-31), so the hold-out years (about 880 sessions, 3.5 years) were never in them; they were reported only as four "seen" farm starts (FINDINGS 28). `run_holdout_view.py` (descriptive, nothing selected, outputs `results/holdout_view_*`) puts them side by side with dev, same evaluator, same baseline cells.
**It is out-of-sample for the parameters, but not clean and not large.** (a) It was looked at many times: dev-versus-hold-out rank correlation was computed over 4,752 configs, the a2 = 0.4 NQ result ($53k) was picked after seeing it, and the gap-VWAP idea was found after seeing the 2025 failure (FINDINGS 6, 27). (b) It is about one independent 3-year window: the farm starts below are every 10 sessions and overlap. (c) It contains two regimes, a strong 2023-mid-2024 and a weak mid-2024 onward.
Account level, Combine started every 3 sessions, funded accounts followed to the end of the data (2025 starts are not shown: too little follow-up):
| | starts | pass | paid | rate ($ per Combine session) |
|---|---|---|---|---|
| noise-area, dev 2011-22 | 987 | 22.6% | 54.3% | 31.3 |
| noise-area, hold-out 2023-24 | 168 | 35.1% | 74.6% | 105.1 |
| gap-VWAP, dev 2011-22 | 987 | 29.0% | 65.4% | 41.6 |
| gap-VWAP, hold-out 2023-24 | 168 | 21.4% | 66.7% | 3.5 |
(The dev rates differ slightly from the $24.8 / $43.4 of FINDINGS 28 because funded accounts here run to 2026 rather than being cut at 2022.) The ranking of the two sleeves flips: the sleeve with the weak dev evidence (noise-area, t 1.64) is strong in 2023-24 and the sleeve with the strong dev evidence (gap-VWAP, t 3.46, picked from 156 variants) is close to zero.
Farm level, baseline 3 + 2, one Combine at a time, farms started every 10 sessions, as modelled (the dev cost of $1.72 per micro is about today's real cost-to-ATR in these years, so "as modelled" is the real cost environment in the hold-out) and on the exact re-based basis. Windows of 378 and 504 sessions, because a 756-session window does not fit after 2023-01:
| Window | Farms | Mean | Median | 25th pct | Positive | Worst |
|---|---|---|---|---|---|---|
| 504, dev only, as modelled | 272 | $7.0k | $1.2k | -$4.5k | 55% | -$9.6k |
| 504, dev only, exact re-based | 272 | $20.2k | $12.1k | $1.6k | 78% | -$10.3k |
| **504, hold-out starts, as modelled** | 41 | $16.5k | $16.3k | $12.3k | 95% | -$2.9k |
| 378, dev only, as modelled | 285 | $4.1k | $0.2k | -$3.5k | 54% | -$8.2k |
| 378, dev only, exact re-based | 285 | $13.1k | $9.3k | -$1.3k | 71% | -$8.2k |
| **378, hold-out starts, as modelled** | 53 | $10.2k | $14.5k | $3.4k | 77% | -$6.6k |
Windows that straddle 2022/2023 (37 and 50 farms) are all positive and are in neither group. By start quarter the 378-session hold-out farms read: 2023 Q1-Q4 $12.4k / $14.9k / $16.5k / $14.2k; 2024 Q1-Q2 $14.7k / $14.8k; **2024 Q3 $0.6k, 2024 Q4 -$4.2k, 2025 Q1 -$6.6k** (one farm). The 504-session farms started in 2024 Q3 (2 farms) are -$2.9k.
**Reading.** (1) The level of the real hold-out farms ($14.5k and $16.3k medians) matches the re-based dev ($9.3k and $12.1k) and is far above the as-modelled dev ($0.2k and $1.2k): the one piece of out-of-sample evidence on the cost re-basing agrees with it (weakly: it is a single regime, and its good start-years coincide with a strong noise-area stretch). The council strategy seat reports the same from the gross side: noise-area gross Sharpe 1.28 in dev and 1.31 in the seen years, net 0.62 and 1.16, i.e. the cost, not the edge, moved (not re-run here). (2) It does not settle the edge: the hold-out is two good start-years followed by roughly 12 months in which new farms are flat to negative, and one window cannot tell decay from bad luck. (3) The hold-out disagrees with dev on which sleeve carries the farm, so the 3 + 2 split remains a risk preference, not a measured optimum.
**What to trust.** A hold-out should outrank in-sample tuning, but this one is small and was used too often to be the only thing trusted. Use it as a check on the dev conclusions (here it passes on level, fails to confirm on persistence) and keep the forward data from 2026-07-02 on as the only fully clean test.

## 8. Is another large audit worth it?
Not a repeat of the five-agent audit: signals, engine, evaluator and the re-basing were each independently reproduced and the remaining uncertainty (section 5) is about assumptions and future data, which reviewers cannot settle. What has **not** been independently reviewed since the last audit: (a) the live harness code written afterwards (`paper/tickfeed.py`, the live guards in `paper/runner.py`, `--warm-from-lse`, `lse_csv_to_bars.py`, `check_lse_overlap.py`) and the FINDINGS 32 descriptive numbers (the real-feed guards, the 0.4% roll-jump threshold and the early-close rule, were written with the feed's data already in view; the 370-bar and 2% clauses are from PR15); (b) a numbers-to-source reconciliation of FINDINGS 27-33, the dashboard datasets and this ledger. A two-agent audit of (a) and (b) is proportionate. Anything else would be new analysis (for example, measuring whether the sleeves' R edge differs by price level inside the development data), which needs a pre-registration first.

## 9. Errata after Council 2 (15 October 2026; details in FINDINGS 37)
The following statements above or in the chat explanation are withdrawn or weakened: the coin-flip benchmark of 14-16% (a risk-matched coin passes about 20% recently; the null range is 2-20%, so pass rate cannot test the edge); 'the recent real pass rate is no better than a coin flip'; the count of consistency-blocked Combines (about 3 events, not 10); the hold-out as support for the cost re-basing and the 'ranking flip' (one window and inside one standard error); 'the 2025 weakness is edge decay' (shortfall 1.1-1.5 standard errors, Bayes factor for a break 0.4-1.2). The exact re-basing headline ($38.8k / $22.8k) depends on the arbitrary reference price (6,000 gives $17.7k / $11.9k; 30,000 gives $45.3k / $32.0k). Tier B and Tier C numbers are unchanged as measurements; the readings of them are weaker than written.
