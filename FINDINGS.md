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

## 14. Engine check, pass vs payout time, other prop firms, and a self-funded multi-firm ladder
Pre-registered in `docs/PREREGISTRATION_4.md` (committed before the runs). Code: `firm_engine.py` (rule-parameterised eval/funded engine; the Topstep spec reproduces `real_engine` exactly, `test_firm_engine.py`),
`firm_specs.py`, `multi_firm.py` (cash-gated multi-firm simulator, `test_multi_firm.py`), `run_firm_comparison.py`, `run_multi_firm_ladder.py`, `run_ladder_unlock.py`, `test_farm_caps.py`, `diag/pass_and_payout_times.py`.

**Engine check.** The farm never exceeds 5 funded accounts (`max_funded=5`, enforced and tested over 24 randomised farms with every refill rule: `test_farm_caps.py`). A Combine that passes while all 5 slots are busy waits up to 21 sessions and is then wasted (an assumption: Topstep's real rule for a passed Combine with no free XFA slot is unverified).

**Time to pass and time to payout, separately (Topstep, final config, starts every 3 sessions):**
| Period | Combine pass rate | Calendar days start to pass (median, 25-75%) | Funded accounts ever paid | Days activation to first payout (median, 25-75%) | Start to first payout, paid accounts |
|---|---|---|---|---|---|
| 2011-2022 | 22% | 32 (20-56) | 83% | 210 (113-382) | 261 (192-470) |
| 2023-2026 (seen) | 27% | 34 (21-60) | 100% | 117 (103-259) | 165 (134-288) |
Topstep's 1-3 business-day approval is not included.

**Other firms - rules found by web search (2026-10-14).** Every number comes from search snippets and third-party reviews; WebFetch to the firms' own pages was blocked, so **nothing is verified against the firm itself**. Assumptions are listed in `firm_specs.py`. Rule risks the simulation cannot enforce:
- FundedNext Futures: the official help article bans copying between FundedNext accounts, even your own; micro-scalping rule (trades under 10 s); bank-transfer payouts unavailable for Cambodia (crypto/Rise only); account ends after the 5th payout; current price not found (used $250 one-time).
- FTMO Futures: Beta (launched Aug-Sep 2026), max 3 sim-funded accounts, automation policy conflicting; Pro has a hard $2,000 daily loss limit.
- Alpha Futures: **bots/EAs/AI banned on all plans**; counterparty risk (the NinjaTrader/Tradovate contract was terminated in July 2026, Premium accounts closed, payouts in batches); delisted by a comparison site.
- Take Profit Trader: copy trading banned on PRO, automation banned or restricted (uncertain), a $4,500 buffer must stay in the account after every payout, 80/10 split, list price $360/month.
- Cambodia eligibility was not found for any firm; get it in writing.

**Standalone results per firm (same NQ basket at every firm; selection by the pre-registered rule on ten 3-year dev farms, 2011-2020 starts; costs are Topstep's for all firms):**
| Firm | chosen R / a1 / a2 / first-payout balance | dev mean net (3y farm) | worst | green start-years | net per fee $ | pass rate | days start to pass | P(funded ever paid) | days activation to first payout | full-history net | seen starts 2021 / 2023 / 2025 | edge-removed placebo (dev mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Topstep 150K | 12800 / 0.4 / 0.8 / $4000 | $18,644 | $-7,363 | 7/10 | 2.00 | 17% | 27 | 78% | 211 | $137,733 | [91892, 32212, 6149] | $-9,832 |
| FTMO Growth 150K | 9600 / 0.8 / 0.4 / $3000 | $11,319 | $-19,236 | 6/10 | 1.00 | 23% | 54 | 70% | 27 | $66,024 | [30207, 10561, -3385] | $-12,088 |
| FTMO Pro 150K | 6400 / 0.4 / 0.4 / $4000 | $2,784 | $-10,619 | 4/10 | 0.19 | 13% | 41 | 100% | 945 | $38,906 | [15671, -1061, 6358] | $-11,467 |
| FundedNext Flex 150K | 6400 / 0.4 / 0.4 / $3000 | $11,592 | $-8,619 | 5/10 | 1.34 | 25% | 60 | 77% | 210 | $110,993 | [63669, 24962, 2901] | $-7,591 |
| Alpha Advanced 150K | 12800 / 0.4 / 0.8 / $4000 | $2,029 | $-34,408 | 4/10 | 0.12 | 20% | 52 | 80% | 209 | $39,231 | [47672, 10463, 3278] | $-24,972 |
| Alpha Standard 150K | 6400 / 0.4 / 0.4 / $3000 | $4,673 | $-13,960 | 6/10 | 0.68 | 31% | 49 | 80% | 246 | $72,600 | [62603, 12436, -37] | $-15,463 |
| Take Profit Trader PRO 150K | 6400 / 0.4 / 1.6 / $7000 | $10,981 | $-24,340 | 5/10 | 0.68 | 31% | 49 | 74% | 429 | $98,401 | [19615, -3549, -5677] | $-31,394 |
- Every firm is profitable on the dev farms and every edge-removed placebo loses, so the money comes from the strategy's edge. None beats Topstep on dev (best net per fee dollar: Topstep 2.0, FundedNext 1.3, FTMO Growth 1.0, Alpha Standard and Take Profit Trader 0.7, FTMO Pro 0.2, Alpha Advanced 0.1).
- Topstep's selected risk here is R=12,800 (the pre-registered rule; the earlier cliff at R>=16,000 means 9,600 is the safer choice).
- 2023+ starts (seen): pass rates 10-52%, days to pass 25-119, days from activation to first payout 41-216. FTMO Growth is the quickest to a first payout (41 days) but pays out less reliably (70% ever paid on dev).

**Self-funded ladder (`run_multi_firm_ladder.py`, `run_ladder_unlock.py`).** Rule implemented as you described: out-of-pocket money only for the first Topstep account, running ONE account at a time (Combine, pass, funded) until the first payout arrives; after that no new money, a purchase needs cash >= price + 3 billing cycles of monthly fees (+ a cash threshold before a further firm is added); one Combine at a time is always kept running while cash allows.
- **Out of pocket: median about $1.4k, worst about $2.6-3.8k** across the dev farms (about 5 Combines at ~$300 each until one passes, plus waiting).
- **Time to the first payout from the first Combine is long in quiet regimes:** 109-429 days on seen starts, but 1,471 days (4 years) from a 2010 start.
- Topstep alone on this rule from 2010: net +$114k from $2.4k out of pocket (vs +$138k unconstrained). Adding the other firms immediately (no cash threshold) hurt: the extra purchases drain the bankroll in droughts and the farm can run out of cash with nothing alive ("ruined"; 2 of 3 seen starts for the same-basket mix).
- With a cash threshold of $3-6k before adding firms: same-basket ladder improves on Topstep-only over long dev horizons (about $86-98k vs $47k, 4 runs) and on 3-year dev farms ($9.6-10.0k vs $9.0k), but is worse on seen starts (2021: $17-26k vs $31.5k; 2023: $17-18k vs $23.2k). Different-sleeves mix: better on the 2021 start ($148k vs $60k), worse on 2023, ruined in 3 of 9 dev 3-year farms.
- **Read this as: scaling to more firms adds variance, not a reliable gain.** All accounts trade the same signals, so their drawdowns and droughts coincide, the other firms cost more per payout than Topstep, and the self-funding rule has an absorbing state (cash below one fee with nothing running).
- Recommended order if you ever scale (by dev net per fee dollar, rules permitting): Topstep, FundedNext, FTMO Growth. Alpha (bot ban, counterparty risk) and Take Profit Trader (copy/automation limits, high fee, buffer) look worst.

## 15. Shortening the wait for the first payout (`docs/PREREGISTRATION_5.md`, `run_payout_speed.py`, `diag/`)
**Why the wait is long.** For funded accounts that do get paid, the balance trigger is the last condition to arrive in 97% of cases at the default size (`diag/what_blocks_the_first_payout.py`): at 0.4 x cushion the position is about 3 micros, which earns about $15 per micro per day, so reaching $4,000 takes about 136 sessions (about 190 calendar days). The 5 winning days of $150 arrive after about 26 sessions. The wait is therefore size x edge, not a rule quirk, and in quiet regimes (2011-2017) the first payout takes years or never comes (a 2010 start waited 1,471 days under the single-account rule).

**The trade-off (dev farms, 3-year, start years 2011..2020; time = calendar days from Combine start to first payout among paid accounts):**
| Preset | pre-payout size / first-payout balance | median days | funded accounts ever paid | first payout | dev mean net | worst 3-year farm | seen starts 2022 / 2023 / 2025-01 / 2025-07 | full-history net (fees) |
|---|---|---|---|---|---|---|---|---|
| safe (default) | 0.4 / $4,000 | 258 | 82% | $2.0k | $20.1k | -$7.0k | $70k / $41k / $12k / $6k | $134k ($39k) |
| balanced | 1.2 / $2,000 | 64 | 63% | $1.8k | $22.7k | -$22.8k | $24k / $12k / -$0.5k / -$16k | $130k ($111k) |
| fast | 2.4 / $6,000 | 62 | 34% | $3.9k | $38.1k | -$14.2k | $5k / $14k / $31k / -$18k | $194k ($151k) |
- **Farm level (what you actually wait for):** days from the first Combine to the first payout of the farm. safe: median 297 (dev) / 254 (2023+), 3-6% of farms paid within 90 days. balanced: 160 / 120, 19-25% within 90 days. fast: 212 / 218, 11-12% within 90 days. So even the quick presets leave a 4-5 month median wait, and under 90 days is the exception.
- **Cost of speed:** funded accounts mostly die (balanced: all 115 funded accounts died, fast: all 129) and fees rise 3-4x; the worst 3-year farm gets deeper; the three most recent start dates are weaker for balanced and mixed for fast (the 2025-07 start loses $16-18k vs +$6k for safe). The edge-removed placebo loses for all three (balanced -$14.0k, fast -$15.6k dev mean), so the profit is still the strategy's edge. (The pre-registered rule picked the fast point; the "trades flipped" placebo was dropped as invalid here because it understates intraday lows.)
- **What did not help:** buying several Combines at once (k0 = 2-8) or spacing them 5-21 sessions apart did not shorten the wait (all accounts share one market path); changing the lead firm was not faster (dev median days to first payout: Topstep 342, FTMO Growth 430, FundedNext 552, Take Profit Trader 576); the Topstep consistency path was slower.
- **Under the single-account out-of-pocket rule** the faster presets cost more up front (out-of-pocket median $1.0k safe, $1.8k balanced, $2.4k fast; worst case $2.0k / $6.2k / $8.0k).
- **Conclusion:** the wait can be cut about 4x per account (about 60 days) and about 2x for the farm, but only by accepting more fees, less reliable payouts and worse recent starts. There is no setting that is fast, cheap and robust at once with a Sharpe near 1 edge. The default stays "safe"; `PRESETS` in `final_strategy.py` hold the other two.

## 16. ES / GC components, one-at-a-time buying, and the big / moderate / conservative lists (`docs/PREREGISTRATION_6.md`, committed before running)
Balanced is now the default preset (`final_strategy.py`). Code: `xs_sleeves.py`, `run_es_gc_search.py`, `run_tiers.py`, `multi_firm.py` (`Policy.global_one_eval`).

**ES and GC: nothing found.** 60 pre-declared trials (ports of the NQ ideas plus prior-day-range breakout, overnight-range breakout, COMEX-open opening-range breakout, gap fade), realistic costs, selection on 2010-2022. **No trial reached the acceptance bar** (dev Sharpe >= 0.5, >= 60% positive dev years, hold-out Sharpe > 0): the best dev Sharpe was 0.39 (ES gap-filtered VWAP; 0.17 on 2023+), the median trial was -0.61, and about 0.65 is what the best of 60 pure-noise trials would show. So the pool for the lists stays NQ-only; ES/GC give no usable component on this data. Full list: `results/es_gc_trials.csv`.

**One Combine at a time overall (your rule), simple results.** Fees fall, but so does everything else, because a funded slot fills slowly (about 5 Combines per pass at 20% pass rate, one after another). Balanced on all five slots, buying one at a time: full-history net +$41k (payouts $107k, fees $65k) vs +$130k with the earlier multi-Combine rule (fees $111k). So the "no extra fees" rule saves about $46k of fees and gives up about $88k of net.

**The three lists** (1 big risk on NQ gap-VWAP at the "fast" sizing, 2 moderate on the NQ basket at "balanced", 2 conservative on the NQ basket at "safe"; one Combine at a time overall, the next one goes to the list with the most empty slots):
| Case | dev mean 3-yr farm | worst | green | first payout (median days) | seen starts 2021 / 2023 / 2025-01 / 2025-07 | full-history net (fees) | months with a payout | most out of pocket |
|---|---|---|---|---|---|---|---|---|
| 5 slots safe | $5.8k | -$5.2k | 5/10 | 364 | $60k / $23k / $2k / -$3k | $77k ($42k) | 24% | $3.0k |
| 5 slots balanced (default) | $4.4k | -$6.9k | 7/10 | 218 | $29k / $11k / $2k / -$4k | $41k ($65k) | 26% | $6.7k |
| 5 slots fast | $9.6k | -$8.5k | 7/10 | 371 | $47k / $11k / $7k / -$4k | $61k ($66k) | 20% | $5.5k |
| **3 lists 1 / 2 / 2** | $3.3k | -$11.7k | 6/10 | 218 | $112k / $13k / $5k / -$4k | **$111k ($61k)** | **32%** | **$2.7k** |
| 3 lists, no big | $3.5k | -$9.9k | 5/10 | 218 | $33k / $15k / $2k / -$4k | $47k ($62k) | 26% | $10.7k |
| (exploratory) conservative adds ES VWAP | $2.1k | -$11.9k | 6/10 | 218 | $101k / $4k / $2k / -$4k | $33k ($58k) | 24% | $1.7k |
| (extra rule) each list buys its own Combine, 3 open at once | $21.8k | -$15.3k | 6/10 | 218 | $73k / $12k / $5k / -$11k | $154k ($82k) | 40% | $8.0k |
- Monthly net correlation between the lists: big vs moderate 0.10, big vs conservative 0.13, moderate vs conservative 0.31 (low, because of different sizing, different components for the big list and different timing). Full-history net by list: big $62k, moderate $34k, conservative $16k.
- On the full history and the seen starts the 3 lists beat 5 balanced slots (+$111k vs +$41k, payouts in 32% of months vs 26%, less out of pocket), but on the pre-registered dev metric (3-year farms from a cold start) they do not (dev mean $3.3k vs $4.4k, worst -$11.7k vs -$6.9k). A one-at-a-time farm needs 2-3 years just to fill its slots, so the 3-year dev farms understate it; the full-history advantage is partly the seen years. Treat the improvement as plausible, not proven. The gain mostly comes from the big list at "fast" sizing and from mixing sizes, not from new markets.
- Allowing three Combines open at once (one per list) gives +$154k, but that is a different rule from the one you asked for.

**What the current NQ strategy is, by the pre-registered rule** (conservative if >= 75% of funded accounts are paid, moderate if 50-75%, big risk if < 50%): at "safe" sizing 82% -> **conservative**, at "balanced" (the default) 63% -> **moderate**, at "fast" 34% -> **big risk**. The strategy itself (NQ only, one trend/momentum family, positively skewed, regime dependent) is a single risk factor: even the conservative list is only conservative about how fast the account is pushed, not about what it trades.

## 17. ES / GC again: SSRN and arXiv ideas, regular trading hours only (`docs/PREREGISTRATION_7.md`, committed before running)
**Overnight rule.** Nothing is traded or held outside 09:30-15:55 ET. `audit_trade_hours.py` records the entry and exit time of every trade (`intraday_strategies._trace`) and fails if any falls outside: the three NQ sleeves of the strategy (earliest entry 09:31 ET, latest exit 15:55 ET), the ES and GC ports and all 44 new trials pass; the one GC opening-range idea anchored at the 08:20 ET COMEX open (entries from 09:21 ET) fails and the sleeve code now refuses any entry anchor before 09:30. Overnight prices are still used as signal inputs (gap, overnight range, the 18:00-anchored VWAP) but no position exists then.

**Literature search** (two subagents, abstracts and snippets only, nothing verified): the best-documented ideas are the ones already in use (noise-area momentum, VWAP trend, last-half-hour momentum); recent papers report decay (a June 2026 SSRN replication finds Sharpe 0.39 out of sample for the noise-area rule; Rosa 2022 finds last-half-hour momentum gone out of sample) and several arXiv null results (Mesfin, 2026: 14 OHLCV signal families on MNQ and MGC, none passed walk-forward after costs). Nothing published tests a US-hours-only GC rule net of costs.

**44 pre-declared trials on ES and GC** (L1 last-half-hour momentum incl. the COMEX 13:00-13:30 window for gold, L2 opening-move reversal, L3 lunch drift, L4 5-minute opening-candle breakout, L5 volatility-gated opening-range breakout, L6 same-half-hour continuation, L7 ES/NQ-to-GC cross-asset, L8 narrow-range-day breakout). **None was accepted:** best dev Sharpe 0.10, median -0.66, no trial reached 0.5 (`results/literature_trials.csv`). Together with the 60 trials of section 16 that is 104 looks at ES/GC (best-of-104 pure-noise Sharpe is about 0.7) with nothing found, in line with the published null results.

**Consequence.** On this data and these costs, ES and GC give no component for the lists. The lists can differ only by sizing and by NQ sleeve mix; real cross-market diversification would need other instruments (for example RTY, YM, CL, ZN, 6E), which means new historical data and another counted trial budget.

## 18. The user's 60-strategy / 50-model list: 144 pre-declared trials (`docs/PREREGISTRATION_8.md`, `docs/STRATEGY_TRIAGE.md`; committed before running)
Code: `daily_features.py` (causal ADX, ER, Hurst, variance ratio, RSI, Bollinger/Keltner squeeze, ATR percentile, ...), `sleeves_ext.py` (intraday-native ideas and intraday-only versions of the daily systems), `ml_meta.py` (numpy-only walk-forward logistic / L1-logistic / gradient-boosted stumps), `run_extended_search.py` (deflated Sharpe and block bootstrap), `run_overlays.py`. All 144 trials traded only 09:30-15:55 ET (checked on the trades taken).

**Result: no trial was accepted** (rule: deflated Sharpe >= 0.95, >= 60% positive dev years, hold-out Sharpe > 0, block-bootstrap p < 0.05). Also none reached the "weak evidence" band. Details (`results/extended_trials.csv`):
- 44 of 144 trials had a positive dev Sharpe, 33 were positive on both dev and 2023+, and 8 had dev Sharpe >= 0.5, which is about what 144 zero-edge trials give by chance (about 6% of them clear 0.5 when the Sharpe standard error is about 0.32).
- Intraday-only versions of the user's daily systems and the intraday-native ideas on **NQ, ES and GC**: median dev Sharpe -0.24 / -0.31 / -0.32. Best singles: NQ open-drive continuation (#31) dev 0.43, hold-out 0.88, positive in 69% of dev years, bootstrap p 0.07 (suggestive only); GC two-day sweep reclaim (#12) dev 0.66, hold-out 0.30; ES squeeze ignition (#8) dev 0.50, hold-out -0.51.
- Regime gates on the existing NQ sleeves (median dev 0.21, hold-out 0.74): the noise-area sleeve gated by ATR percentile > 60 (dev 0.81, hold-out 1.03) or ADX > 25 (0.69, 1.24) looks better, consistent with the edge living in high-volatility trending periods, but does not survive the haircut.
- ML meta-labeling (walk-forward, out-of-sample only): the gap-VWAP sleeve with a logistic filter reached dev 1.23 / hold-out 1.03 and with boosted stumps 1.15 / 0.62 (base sleeve about 0.89 / 0.87); the 0.55-threshold version failed hold-out (-0.87). The pre-registered deflated Sharpe uses the observed cross-trial dispersion (SD of Sharpe 0.55), which puts the best-of-144 noise level at 1.47, so these score 0.23 / 0.17. Using only sampling error the noise level would be about 0.86 and these two would score "weak evidence". They are improvements to a sleeve that was itself chosen by earlier searching, so they are candidates for a forward test, not findings.
- Validation tools (deflated Sharpe, stationary block bootstrap) are in place; the Monte Carlo reshuffle through the rule state machine was reserved for accepted ideas (none).

**Trade-management overlays on the existing NQ basket** (reported as the change in dev farm value, not counted as trials; balanced default; `results/overlays.csv`):
| Overlay | dev mean 3-yr farm | p25 | worst | green | seen starts 2021 / 2023 / 2025-01 / 2025-07 | full-history net |
|---|---|---|---|---|---|---|
| baseline | $22.7k | $2.5k | -$22.8k | 7/10 | $63k / $12k / -$0.5k / -$16k | $130k |
| daily-loss breaker at -$2,400 / -$2,000 / -$1,500 | $21.7k / $20.1k / $24.1k | $0.7k / $3.6k / $13.1k | -$20.5k / -$21.6k / -$6.7k | 7 / 9 / 8 | worse on 2023+ starts | $115k / $107k / $120k |
| daily profit cap $3,600 / $2,700 / $1,800 (Combine) | $10.2k / -$8.4k / -$38.8k | negative | much worse | 7 / 2 / 0 | worse | $64k / -$53k / -$223k |
| max 1 / 2 entries per sleeve per day | $30.9k / $33.9k | $14.6k / $10.6k | -$9.7k / -$1.4k | 9 / 9 | $71k / $58k on 2021 but $0.7k / -$14k on 2023 and -$14k on 2025-01 | $169k / $169k |
- A daily profit cap (the "consistency cap") destroys the edge: the strategy is positively skewed and capping the big days removes most of the profit. Do not use it.
- A tighter daily-loss breaker and an entry limiter improve the 3-year dev farms and the worst case, but both are weaker on the most recent starts (2023, 2025), so they are plausible, not proven; the entry limiter is the more convincing of the two (a mechanism: fewer re-entries after stop-outs, lower costs).
