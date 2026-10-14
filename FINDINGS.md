# Findings and summary: is a Topstep Combine / funded account "farm" a positive-value option?

Status: research complete for now, **no forward test run, no live money**. Everything below is backtest or simulation on NQ/ES/GC
1-minute data (Databento, 2010-06 to 2026-07-02, GitHub release v1.0.0). Code for the final strategy: [`final_strategy.py`](final_strategy.py).

## 1. The theory
Pay Combine fees, pass the Combine, collect Express Funded Account payouts, let accounts die, open new Combines, repeat.
Losses are capped at the fees while payouts are not, so the farm behaves like a cheap option. The question was whether it is
positive in expectation after all fees for a strategy with a modest Sharpe, and how much Sharpe is needed.

## 2. Verdict
**The structure is real; positive expected value is not established.** Treat it as a regime-dependent lottery ticket, not an investment.
- A no-edge control does *not* earn money over the full history on the selected rules (about -$56k with the strategy's mean edge removed), so the payout rules alone are not a free lunch.
- With the strategy's edge the farm made about +$108k over 16 years, but four years (2018, 2021, 2022, 2024) carry essentially all of it; removing them leaves about +$2k. Start years 2011-15 averaged slightly negative.
- After haircutting for the number of things tried, the best estimate of the true Sharpe is about 0.3-0.5; the farm's break-even Sharpe is about 0.44. Expected value is therefore about zero, with a fat right tail in volatile eras.
- Recent regime is unfavourable: a start in mid-2025 loses (-$1.5k) and a start in 2026 lost about $12k.
- Independent reviewers (quant, statistician, rules researcher) all recommended against scaling. Evidence is about 5 independent farm windows.

## 3. The strategy (see `final_strategy.py` for the exact code path)
Three NQ micro sleeves, flat by 15:55 ET, signals at bar close, fill at next open, $1.22 all-in per micro round turn + 1 tick slippage:
1. **Noise-area momentum** (Zarattini-Aziz-Barbon style).
2. **VWAP+ATR momentum** (k=0.3 ATR band around the 18:00-anchored VWAP, 0.2 ATR stop, exit on VWAP re-cross).
3. **Gap + first-30-minute filtered VWAP** (same entry, only when the overnight gap and the first-half-hour move agree): dev Sharpe 0.89, 2023+ 0.87 (NQ only; ES/GC versions were weak).

Static dev-only weights (fit on 2010-2022). Combine risk R=9,600 per ATR unit; funded risk = a x cushion above the MLL floor (a1=0.4, a2=1.6); first payout at balance >= $3,000 then every eligible payout; 5 funded accounts max; "hyst" refill (open no Combines once full until 3 slots empty).

Topstep rules modelled: $9,000 target, $4,500 MLL (real-time, locks at $0), 55% Combine consistency, optional $3,000 DLL, flat by 15:55 ET, 21-session billing, Standard payout path (5 winning days >= $150, 50% of balance, $5,000 cap, $125 minimum), MLL resets to $0 after each payout, 90/10 split, XFA scaling ladder, 5 funded accounts max.

## 4. Key results (net = payouts - all fees, realistic micro costs)
| Farm started | Previous headline (NQ noise only, R=6,400) | Final (dev-selected) | Final + pass-rate throttle |
|---|---|---|---|
| 2010-06 (full history) | +$87k | **+$108k** (payouts $170k, fees $62k, peak cash outlay $16k) | +$63k |
| 2023-01 | +$2k | +$19k | +$24k |
| 2024-01 | -$8k | +$15k | +$12k |
| 2025-01 | -$16k | +$5k | +$5k |
| 2025-07 | -$9k | -$1.5k | -$2.5k |

- Walk-forward (re-pick settings each year from earlier data, 3-year farms): positive in 9 of 11 start years; 2023-26 starts +$4.7k vs -$6.1k for the previous headline; 2016-26 mean $15.9k vs $18.8k (the previous headline did better in strong years such as 2018).
- Dev-vs-hold-out rank correlation across all 4,752 configs: 0.16 (2023+), 0.30 (2025+): selecting on dev adds modest value. Only about 7% of grid configs are positive for 2025+ starts.
- Slippage of 2-3 ticks barely changes farm net (path-dependent, not monotonic). A 30% haircut on payouts leaves the full history positive (+$57k) but 2023+ near +$6k and 2025+ about -$1k; break-even payout haircut is about 64% (full), 42% (2023+), 26% (2025+).
- Break-even edge: about 35% of the observed edge over the full history, about 80% in the 2023+ regime.

## 5. Why the edge faded in 2025-26 (and what could not be done about it)
- In dollars per micro, NQ noise-area P&L in 2025 was about the same as in 2024 (about $15/day) but dispersion roughly doubled and 2025 profit came from a handful of days.
- Trailing variance ratio, return autocorrelation and range had no predictive power for the next day's return: the fade could not be forecast.
- Sizing by trailing strategy volatility instead of ATR did not help on dev data; larger Combine risk (R about 8-9.6k) did, because Combines resolve faster.
- Switching sleeves on/off by trailing Sharpe was worse than static weights on dev data. A pass-rate throttle and a "buffer" refill rule cut fees and losses in weak periods but also give up payouts; they limit damage, they do not create profit.
- More concurrent Combines hurt: with 5 funded slots as the bottleneck, forcing more open Combines lowered dev net from about $22k to about $9k (and to about $0.7k at the extreme). Raising `max_evals` to 20 changed nothing under the default rule.
- Monthly payouts are not achievable: only about 17% of months since 2012 contain a payout (about 26-28% with a2=0.4); payouts cluster because all accounts trade one stream on one path. Mixing ES/GC accounts raises coverage to about 39% but makes dev net negative.

## 6. What went wrong along the way (corrections)
- Early costs were too light ($2.50 per mini-equivalent): corrected to $1.22 per micro + 1 tick; NQ noise-area Sharpe 0.89 -> 0.56.
- I said "coin-flip loses under dynamic sizing": wrong for the early headline plan (the zero-edge control then also earned money in recent years). I also quoted a no-edge control of +$17k for 2023+ that was one noisy 10-seed average; over 24 seeds it is about +$8k with a spread of roughly -$36k to +$46k, and a coin flip is a poor null because it is far more volatile than the strategy.
- The hold-out (2023+) was never clean: the a2=0.4 NQ result of about $53k was picked after seeing it, and the gap-VWAP idea was found after seeing the 2025 failure.
- The early "unresolved Combine" bug billed thousands of days (now capped at 252 sessions).
- Audit-found issues were fixed in `real_engine.py`, `farm_real.py` and `portfolio.py`; 38 tests pass (`python -m pytest -q`).

## 7. Practical blockers (from web-search snippets only; **not verified on Topstep's own pages**, re-check before spending money)
- Per third-party summaries of the API terms, trading must originate from your personal device; VPS/remote servers are prohibited. This constrains full automation across many accounts.
- No Topstep text found banning identical orders on your own accounts, but also no explicit permission; cross-account hedging is prohibited. Topstep has a native trade copier for Combine and XFA accounts. Get written confirmation from support.
- Max 5 active funded accounts, no stated limit on Combines; everything under one profile.
- Cambodia eligibility/payout options could not be confirmed.
- Topstep's own risk disclosure reportedly lists a 16.8% Combine pass rate and 33% of funded traders receiving a payout for 2025; our modelled Combine pass rate is about 22-24%.
- 2026 payout rules were reportedly changed (90/10 split, 50% of balance, $150 winning day); the model already uses these.

## 8. If this is ever tested live (not done)
- Freeze the configuration; do not retune. Paper-trade first (costs nothing, highest information per dollar).
- Then buy one Combine at a time, never two on the same day; cap total net outlay at about $3k.
- Kill if the 12-month Sharpe is below -0.5, realised costs exceed 1.5x the model, or the cap is hit. A good 12 months is not a green light: Sharpe 0.8 vs 0 needs roughly 7-10 years of data to confirm; scale only after 504+ days with Sharpe >= 0.5 and fills matching the model.
- Real money buys execution and payout-friction information, not statistical power.

## 9. Repo map
| What | Where |
|---|---|
| Final strategy + farm config | `final_strategy.py` |
| Rules engine, farm simulator | `real_engine.py`, `farm_real.py`, `chain_real.py`, `contract_eval.py`, `reference_impl.py` (independent re-implementation used in the audit) |
| Sleeves | `intraday_strategies.py`, `academic_strategies.py`, `vwap_variants.py`, `portfolio.py`, `switching.py`, `risk_units.py` |
| Studies in order | `run_real_study.py`, `run_sweep.py`, `run_holdout.py`, `run_rinse_repeat.py`, `run_dynamic_*`, `run_farm_*`, `run_corrected.py`, `run_vwap_variants.py`, `run_regime_study.py` -> `run_regime_ext.py` -> `run_regime_rank.py` -> `run_regime_holdout.py` -> `run_regime_walkforward.py`, `run_monthly_payout.py`, `run_monthly_diversified.py`, `diag/` |
| Results | `results/*.csv`, `results/dashboard_v2/*.json` (dashboard data) |
| Tests | `test_valuation.py`, `test_engine_audit.py` |
| Data (not in git) | `/home/user/data_raw` (Databento NQ/ES/GC; GitHub release v1.0.0) |

Trial counts (so the amount of searching is visible): 156 VWAP-variant trials (`results/vwap_variant_trials.csv`), 300+ earlier sleeve variants (`results/sleeve_sweep.csv`), 4,752 farm configurations (`results/regime_dev_grid*.csv`), 4 switching rules, 2 risk units, 3 refill rules.

## 10. Per-funded-account check (`diag/funded_payout_rate.py`, descriptive, includes seen data)
Combine pass rate about 21%. Of funded accounts, about 86% reach at least one payout (82% for activations before 2023, all of them from 2023), median about 138 calendar days from activation to first payout, about 1.8 payouts per funded account ($3.3k paid on average) at post-payout size 1.6 and about 3.3 payouts ($3.8k) at 0.4. So the funded side mostly works; the cost side (about 4.7 Combines bought per funded account) and the regime are what decide the sign.

## 11. Can mechanics alone improve it? (existing data only; DEV-selected, 2010-2022)
Scripts: `combine_policy.py` (state-dependent Combine sizing; neutral policy checked equal to the engine in `test_combine_policy.py`), `run_combine_policy_study.py` (108 policy x post-payout-size combinations), `run_funded_payout_policy.py` (36 combinations), `run_combine_risk_boundary.py`. Same dev objective as before (12 three-year windows).
- **State-dependent Combine sizing does not help.** Early-phase size-up, de-risking near the target (lock-in), and a cushion cap on risk near the loss floor all lowered the dev objective. De-risking makes Combines slower (more billing cycles) without raising the pass rate enough; the fixed-risk Combine was already near the top.
- **Pass rate is bought with time.** R=6,400 gives a 26% pass rate but ~42 days to pass; R=9,600 gives 21% and ~29 days; the extra billing cycles cancel the higher pass rate.
- **Combine risk R has an interior optimum with a cliff:** dev objective is $6.4k (R=6,400), $10.2k (9,600), $11.2k (12,800), then collapses ($-3.2k at 16,000, worse beyond). R=12,800 also did better on the hold-out starts but sits next to the cliff, so the final config keeps R=9,600 (the gap is within noise).
- **Funded side:** withdrawing less than the 50% maximum, or requiring a higher balance before later payouts, lowered dev net. The existing choice (w=0.5, no extra trigger) is the dev best. Post-payout size 1.6 vs 0.4: 1.6 slightly higher dev net, 0.4 more payouts per funded account (3.3 vs 1.9) and more paid per funded account ($4.0k vs $3.0k).
- **Conclusion:** the farm mechanics are already close to the best of the tested space. Remaining upside has to come from the strategy's edge (not available without new, honest data) or from lower real-world costs; further tuning of the same data would mostly add overfitting.

## 12. Pre-registration 2: time to first payout and one new idea (`docs/PREREGISTRATION_2.md`, committed before running)
**Part A - under 90 calendar days to the first payout (Combine start to payout eligibility, excluding Topstep's 1-3 day approval).** `run_time_to_payout.py`, `diag/time_to_payout_holdout.py`, `diag/farm_first_payout.py`.
- Baseline (first payout at balance >= $3,000, funded size 0.4): median 224 days among paid accounts on dev starts (159 on 2023+ starts); only 2-5% of all Combine starts reach a payout within 90 days. Most of the time is the funded stage (median about 190 days from activation on dev; the Combine itself takes about 22 sessions).
- Getting the median under 90 days needs aggressive funded sizing (a1 >= 1.2) and a low trigger ($500-1,500): median about 55-65 days, but only 53-68% of funded accounts ever get paid and the dev objective drops to between -$13k and +$0.0k (feasibility bar was $5k), with 2023 and 2025 farm starts losing money for the t1=1,500/a1=1.2 case.
- Best feasible (dev objective >= $5k) was t1=2,000, a1=0.8: median 105 days on dev (186 on 2023+ starts), farm net about 60% of baseline. **The <90 day goal is not met as a typical outcome without giving up most of the expected value.** Even the fast configs reach a payout within 90 days for only 6-10% of all Combine starts; at farm level the first payout of any account comes within 90 days in 5-25% of farm starts (accounts share one market path, so a second Combine does not help much).
- Consistency path (3 days, 40% rule) was slower and less reliable (47% of funded accounts paid vs 71% on the standard path).

**Part B - turn-of-month long sleeve (NQ, ES; 770 active days each).** Rejected under the pre-registered rule: dev Sharpe on TOM days -0.8 (NQ) / -0.9 (ES), -0.35 / -0.38 over all days; correlation with the basket about -0.03 / -0.05 (a diversifier, but without an edge on dev). It looked strong on 2023+ (Sharpe about 2.0 on TOM days), but adopting it because of that would be selecting on the hold-out, so it is not used.

## 13. Pre-registration 3: first-payout balance, scaling, "green in all years" (`docs/PREREGISTRATION_3.md`, `run_payout_scaling.py`, `diag/`)
**What "Combine to payout" time means here:** days from buying the Combine to the first payout becoming eligible = days to pass the Combine (median about 22 sessions, roughly 30 calendar days) + days from activation to the first payout. Topstep's own 1-3 business-day approval and transfer time is not included.

**Does waiting make the payout bigger and safer?** Half true. Averaged over the grid (dev farms 2011-2020 starts):
| first-payout balance | first payout | paid per funded account | accounts with a 2nd payout | P(funded account ever paid) | median days start to payout |
|---|---|---|---|---|---|
| $1,500 | $1.0k | $2.3k | 56% | 90% | 134 |
| $3,000 | $1.6k | $4.0k | 80% | 84% | 210 |
| $4,000 | $2.0k | $4.4k | 80% | 81% | 251 |
| $6,000 | $2.9k | $5.5k | 91% | 73% | 367 |
| $8,000 | $3.9k | $6.5k | 89% | 72% | 490 |
Waiting gives a bigger first payout, a bigger cushion and more follow-up payouts, but fewer funded accounts ever get a first payout (more time to die first). Net rises up to about $3-4k then flattens while the worst years get worse.

**Pre-registered selection (84 configs, robust "green" objective, dev only):** first-payout balance $4,000, pre-payout size 0.4, post-payout size 0.8 (the rule was: highest 25th percentile of net across ten 3-year dev farms, subject to mean >= $10k and >= 6 of 10 green). Dev: mean +$20.1k, 7 of 10 start years green, worst -$7.0k; it beat the old baseline (3000/0.4/1.6: mean +$13.0k, p25 -$3.7k). Red start years: 2013, 2014, 2015 (the low-volatility years). **No configuration is green in all years**; what holds in every year is a bounded downside (worst 3-year farm about -$7k) against +$30-70k in good regimes.
- Full history: net +$134k (payouts $173k, fees $39k). Seen starts: 2021 +$100k, 2023 +$41k, 2024 +$28k, 2025-01 +$12k, 2025-07 +$5.7k.
- **Fast alternative** (balance $1,500, size 0.8/0.8): median 88 days on dev (under the 90 day goal), mean +$15.3k, 7 of 10 green, but only 75% of funded accounts get paid, paid per funded account $2.2k, and seen starts are weaker (2023 +$6k, 2025-01 +$0.4k, 2025-07 -$6.4k). Its placebos are less clean (reversed trades roughly break even on dev), so it depends less on the edge and more on volatility.
- **Placebos behave** (`diag/placebo_payout_configs.py`): with the strategy's dev-fitted mean edge removed, the winner loses (dev mean -$7.6k, 2 of 10 green); with every trade flipped it also loses (-$6.8k, 1 of 10 green; 2021/2023/2025 starts -$45k/-$26k/-$11k). So the profit is the edge, not an artefact of the rules. The coin-flip control is not a fair null (it is far more volatile than the strategy and makes money with the winner config from seen starts); the edge-removed version is the one to quote. A "shift P&L by one session" check turned out to be uninformative (a near-iid daily series is unchanged by shifting) and is not used.
- **Independent leakage audit** (separate reviewer, new code only): gap30 filter, noise-area and VWAP signal timing, dev-only weights (rebuilt from data truncated at 2022-12-30: identical), farm bookkeeping, and the Combine policy (identical to the engine, 0 mismatches over about 1,300 starts) were all CLEAN. The reviewer re-derived the farm net with its own loop on the reference engine and matched exactly: 2015-01-02 start -$6,169 (zero payouts, $6,169 of fees), 2023-01-03 start +$41,068.
- Caveats the reviewer could not close: the sleeve parameters (k=0.3, s=0.2, the gap30 filter) cannot be proven to have been chosen on dev data only, the dev-fitted weights collapse to a corner (noise 0.5, vwap 0.5, gap30 2.0), and the roll-day skip uses hindsight (not exploitable). 2023+ is still not a clean hold-out, and the 2013-15 and 2025-26 results show the regime risk.
