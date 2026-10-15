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

## 19. Diversifying the strategies across funded accounts, and Combine pass rate by sleeve (`run_account_diversification.py`, `diag/pass_cost_and_placebo.py`)
**Pass rate by sleeve (Topstep rules, dev starts every 3 sessions 2010-2022; 2023+ in brackets; median calendar days to pass; `results/pass_rate_by_sleeve.csv`).** Pass rate rises as risk R falls but takes much longer (one $199 bill per 21 sessions):
| Sleeve | R=3,200 | R=6,400 | R=9,600 (default) | R=12,800 |
|---|---|---|---|---|
| gap-VWAP | 40% (42%), 133 d | 34% (25%), 46 d | 29% (17%), 36 d | 26% (14%), 33 d |
| NQ basket | 35% (67%), 148 d | 26% (37%), 42 d | 21% (23%), 30 d | 16% (15%), 25 d |
| noise-area | 29% (56%), 80 d | 22% (22%), 30 d | 19% (24%), 22 d | 16% (21%), 20 d |
| VWAP+ATR | 20% (36%), 45 d | 18% (21%), 23 d | 15% (15%), 13 d | 13% (11%), 13 d |
| ES VWAP (gated) / GC noise (gated) | 21% / 12% | 17% / 11% | 13% / 10% | 11% / 8% |
**Combine fees per funded account** (the pass rate that matters; dev): gap-VWAP $1,022 at R=9,600 (best), noise-area $1,219 at 6,400, VWAP+ATR $1,256 at 6,400, NQ basket $1,274 at 6,400. The gap-VWAP sleeve has the best pass rate and the lowest cost per funded account on development data; its 2023+ pass rate at R=9,600 (17%) is below the noise-area sleeve's (24%).
**Across firms** (section 14, each firm's own best configuration, dev): Alpha Standard and Take Profit Trader 31%, FundedNext 25%, FTMO Growth 23%, Topstep 17-22%; by net per fee dollar Topstep is still first.

**Diversifying across accounts is possible and it helps on development data.** Instead of five accounts running the same 3-sleeve basket, each group of accounts runs ONE sleeve (so each account is a high-variance option, while the book is diversified). One Combine at a time, balanced sizing, ten 3-year dev farms:
| Case | dev mean | worst | green | months with a payout | correlation between groups | full-history net (fees) | seen starts 2021 / 2023 / 2025-01 / 2025-07 |
|---|---|---|---|---|---|---|---|
| 5 accounts, same basket | $4.4k | -$6.9k | 7/10 | 26% | n/a | $41k ($65k) | $29k / $11k / $2k / -$4k |
| 3 noise-area + 2 gap-VWAP accounts | $17.5k | -$12.4k | 7/10 | 48% | 0.08 | $233k ($69k) | $37k / $26k / -$6k / -$4k |
| 2 noise-area + 1 VWAP + 2 gap-VWAP | $16.0k | -$10.1k | 6/10 | 47% | 0.04 | $209k ($73k) | $44k / $18k / -$6k / -$4k |
With the other rule (each group buys its own Combine) the split gives dev mean $13-15k and full-history net $90-95k, with fees $88-132k.
- Monthly net correlation between the groups is 0.04-0.08, payouts arrive in 47-48% of months instead of 26%, and the dev mean is about 4x higher. Mechanism: a single sleeve per account is more volatile than the basket, and a funded account is an option that gains from variance, while the groups are nearly uncorrelated with each other.
- **Placebo:** with the strategy's edge removed the split still loses (dev mean -$2.7k and -$3.3k, against -$7.9k for the same-basket farm), so the profit is still the edge; the split just shrinks the losses without the edge.
- Caveats: the split is a post-hoc comparison (three cases, two buying rules, six looks); the two most recent starts (2025) are slightly worse than the same-basket farm; the full-history gain includes seen years.

## 20. Topstep default split, best split at FundedNext and FTMO, the scaling pipeline, Kelly sizing, days to payout (`docs/PREREGISTRATION_9.md`, committed before running; `run_firm_splits.py`, `run_scaling_pipeline.py`, `run_kelly.py`, `run_days_to_payout.py`, `build_dashboard_v4.py`)
**Default.** The Topstep default is now 3 noise-area accounts + 2 gap-VWAP accounts, balanced sizing (`final_strategy.ACCOUNT_GROUPS`). The pre-registered rule would have picked "5 gap-VWAP" for Topstep (highest 25th-percentile farm); the split was fixed by the user instead.

**Best split per firm (Part A; ten 3-year dev farms, one Combine at a time within the firm; `results/firm_splits.csv`).** Rule: highest 25th-percentile net among splits with a positive mean.
| Firm | Rule's pick | dev mean | p25 | worst | other splits (dev mean) |
|---|---|---|---|---|---|
| Topstep (5 funded) | user's: 3 noise + 2 gap | $17.5k | -$1.6k | -$12.4k | same basket x5 $4.4k, 2 noise + 3 gap $16.1k, 5 gap $8.3k, 5 noise $6.2k |
| FundedNext Flex (5, rules ignored by the user) | 5 gap | $1.9k | -$3.2k | -$4.8k | 5 noise $4.1k (p25 -$5.7k), 2 noise + 3 gap $3.6k, same basket $1.4k |
| FTMO Growth (3) | 1 noise + 1 VWAP + 1 gap | $5.2k | -$5.6k | -$12.6k | 1 noise + 2 gap $3.5k, same basket $1.2k, 3 gap $0.7k |
The firm effect dominates the split effect: the same strategies are worth $17.5k per 3-year farm at Topstep, $2-4k at FundedNext and $1-5k at FTMO. The FundedNext pick is weak (its mean is the lowest of the six candidates; the p25 rule favours the lowest-variance split), and splits differ less than the spread between farms.

**Scaling pipeline (Part B; Topstep -> FundedNext -> FTMO, out of pocket only for the first Topstep account, one account at a time until its first payout, then every purchase from payouts with a 3-cycle reserve, one Combine open across everything, new firm unlocked when cash passes a threshold; `results/scaling_pipeline.csv`).**
- Rule-chosen threshold: $3,000 (fewest ruins over four long dev runs and nine 3-year farms: 7, against 8 for $6,000 and $10,000; the means are within noise).
- Topstep only, dev 3-year farms (starts 2012-2020): mean net $4.9k, 4 of 9 green, 5 of 9 end ruined after $1.4k-$3.6k put in; the four winners net $7.0k-$21.5k. Adding FundedNext and FTMO gives $3.7k / $4.5k / $5.1k for thresholds $3k / $6k / $10k: no reliable gain, because the next firm is reached only after the lead has already paid out and the FTMO rung is practically never reached (cash and the sequential unlock both have to be satisfied).
- Full history from 2010: -$4.0k to -$4.1k for Topstep only and for thresholds $3k and $10k (ruined after the 2012-2016 weak years), but +$64k at $6k, a single path that survived to 2018. The same rule on 2018 and 2020 starts pays the first account in 33-34 days and then compounds ($65.7k from a 2018 start at $3k). This is a bimodal lottery on a $1.4k-$4.2k bankroll, not a stable edge; it is the same absorbing-ruin structure as in section 14.
- Seen starts (Topstep only): 2021 -$0.5k (ruined), 2023 +$22.8k, 2025 -$3.8k (no payout yet by the end of the data).

**Kelly sizing (Part C; funded stage only, Combine sizing unchanged; `results/kelly_study.csv`; `kelly_sizing.py`).** Contracts = f x cushion x (trailing 120-session mean) / (trailing variance), per micro, causal; 1 micro when the trailing mean is not positive.
- The current balanced size is about 3-4x full Kelly before the first payout and 2-3x after it (median of the trailing estimates: noise-area 4.1x / 2.7x, gap-VWAP 3.0x / 2.0x). This is deliberate: a funded account that dies only costs fees, so the payoff is capped below and the useful size is far above Kelly.
- No setting passed the pre-registered "not greedy" test (mean AND p25 up, worst farm no more than 25% deeper, share of funded accounts paid no more than 5 points lower). Current: mean $17.5k, p25 -$1.6k, 54% of funded accounts paid, farm's first payout after a median of 152 days.
| Setting | dev mean | p25 | worst | accounts paid | median days to farm's 1st payout | verdict |
|---|---|---|---|---|---|---|
| ATR 1.8 / 0.8 | $20.0k | -$4.0k | -$14.9k | 33% | 169 | worse p25, fewer paid |
| ATR 1.2 / 1.2 | $10.2k | -$7.3k | -$12.5k | 51% | 152 | worse mean and p25 |
| Kelly 3x / 1x | $23.4k | +$0.1k | -$10.5k | 46% | 346 | fewer paid (-8 pts) |
| Kelly 4x / 1x | $19.9k | -$1.1k | -$11.8k | 37% | 345 | fewer paid |
| Kelly 3x / 3x | $30.2k | -$4.9k | -$10.5k | 47% | 346 | worse p25 |
(all other Kelly cells in the csv are rejected for a lower mean and/or worse p25.) The near misses (3x/1x, 4x/1x) raise the mean and shrink the worst farm but pay fewer accounts and delay the first payout by about six months (the trailing-mean gate keeps size at 1 micro in 26-39% of sessions), so Kelly is a tool for bigger payouts per paid account, not for earlier ones. With the edge removed, Kelly 3x/1x and 2x/1x still lose on dev (-$3.4k / -$3.6k, against -$2.7k for the current rule), so the Kelly gains on development data are edge, not rules.
- A combination "volatility rule before the first payout, Kelly after it" was NOT tested (the funded engine takes one sizing unit for both stages and it was not in the pre-registered grid); it is the natural next candidate and would need its own pre-registration.

**Days to payout (Part D; `results/days_to_payout_accounts.csv`, `results/days_to_payout_farms.csv`).**
- One account (sequential Combine starts every 3 sessions, balanced, calendar days): pass takes a median 29-32 days, activation to first payout 24-35 days, so about 60-80 days from buying the Combine to the first payout for an account that gets there (old basket x5 62 days, new default 78 days; the gap-VWAP sleeve is slower, 116 days, because it trades only on gap days). 7 Combines are bought per paid account (pass rate 21-23%, 61-69% of funded accounts get paid).
- Whole farm, one Combine at a time, calendar days from the first Combine to the first payout of any account (ten 3-year dev farms): old basket x5 median 218 days (none within 90 days); new default 152 days unconstrained, 254 days in the self-funded pipeline (2 of 10 within 90 days, range 33-459). So the per-account time is unchanged by the split, the farm's first payout is not reliably faster, and the < 90 days goal is not met in the typical case.
- Fee check (ten 3-year dev farms of the default split): 551 Combines bought at $248 on average (Combine plus activation share), 112 funded, 60 of them ever paid; fees $2.3k per paid account against $5.2k paid out per paid account, so about $2.9k net per paid account. The first payout alone (about $1.8k) does not cover the fees; profit relies on the later payouts of the same account.

**Dashboard.** Added: the Topstep default variant (and its edge-removed control) as the default view, the split table per firm, the month-by-month pipeline from six start dates, the Kelly table and the days-to-payout table. Full history of the default split in the unconstrained farm: net $232.7k (fees $68.9k, payouts $301.6k, funded accounts 59, deepest cash dip $5.2k); with the edge removed the same farm nets -$0.2k over the full history (dev placebo mean -$2.7k). Starts: 2012 $228k, 2015 $226k, 2018 $68k, 2019 $42k, 2021 $37k, 2023 $26k, 2024 $12k, 2025-01 -$6k, 2025-07 -$4k, a fade that matches the regime finding.

**Caveats.** FundedNext and FTMO rules come from search snippets (the user chose to ignore FundedNext's own rules); the pipeline's cash gating means the low-bankroll ruin states dominate small samples (9 farms); the Part A pick for FundedNext is weak; everything after 2022 was seen before this section was run.

## 21. Hybrid funded sizing: volatility rule until the first payout, Kelly after it (`docs/PREREGISTRATION_10.md`, committed before running; `run_hybrid.py`, `results/hybrid_study.csv`)
Idea (after the Kelly near misses in section 20): keep the first payout exactly as fast as today (1.2 x cushion / ATR until the first payout) and use Kelly only for the cushion left afterwards. Engine: `run_funded_firm(..., atr2=...)` and `FirmPlan.funded_atr2` give a separate sizing unit for the stage after the first payout (defaults reproduce earlier results; unit test `test_stage2_unit_changes_only_post_payout_sizing`). Three declared trials: f2 = 1x, 2x, 3x full Kelly (120-session causal estimates). Same acceptance rule as section 20, plus "median days to the farm's first payout not more than 10% longer".
| Setting (Topstep default split, ten 3-year dev farms) | dev mean | p25 | worst | green | accounts paid | payouts per paid account | median days to 1st payout | seen 2021 / 2023 / 2025-01 / 2025-07 | full history | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| current: 1.2 / 0.8 x cushion / ATR | $17.5k | -$1.6k | -$12.4k | 7/10 | 54% | 3.10 | 152 | $37k / $26k / -$6k / -$4k | $233k | current |
| ATR 1.2, then Kelly 1x | $11.5k | -$6.2k | -$11.6k | 5/10 | 50% | 3.04 | 152 | $38k / $27k / -$6k / -$4k | $97k | rejected: lower mean, worse p25 |
| ATR 1.2, then Kelly 2x | $12.6k | -$7.4k | -$12.7k | 5/10 | 47% | 2.74 | 152 | $11k / $5k / -$6k / -$4k | $243k | rejected: lower mean, worse p25, fewer paid |
| ATR 1.2, then Kelly 3x | $15.3k | -$7.7k | -$12.7k | 4/10 | 48% | 2.36 | 152 | $17k / $12k / -$6k / -$4k | $146k | rejected: lower mean, worse p25, fewer paid |
- **No cell passes.** The first payout is as fast as the current rule by construction (152 days at the farm level), but every hybrid cell has a lower dev mean and a worse bad-quartile farm, and the larger Kelly multiples also leave fewer accounts paid and fewer payouts per paid account. The 2021 and 2023 starts are much worse for 2x and 3x ($5-17k against $26-37k); the two 2025 starts are unchanged (no payout yet in those farms).
- **Descriptive checks (not used for selection).** Edge removed: current -$2.7k, hybrid 3x -$1.2k (both negative, so the dev profit is edge, as before). Kelly window for the 3x cell: 60 sessions $8.7k, 120 $15.3k, 250 $14.8k; all below the current $17.5k. The full-history figures swing a lot between cells ($97k to $243k) with no relation to the dev ranking, which is what a path-dependent sum of a few large accounts looks like; they are not evidence for any cell.
- **Likely reason (not tested).** The Kelly size is driven by a 120-session trailing mean whose standard error is of the same order as the mean itself, so after the first payout the size follows recent luck (large after a good stretch, 1 micro after a bad one, in 26-39% of sessions) while the volatility rule sizes from the cushion and the current range only. A cleaner test of this would be Kelly with a long, shrunk estimate of the mean; that is a new idea and would need its own pre-registration.
- **Trials.** The Kelly family now has 15 looks (12 in section 20, 3 here) and none was accepted. The current sizing (balanced, 1.2 before and 0.8 after the first payout, x cushion / ATR) stays. This also closes the question "can sizing add payout without being greedy": the sizing rule is not where the remaining gain is; the structural changes (account split by strategy) were.

## 22. Why fewer funded accounts are paid, and can it go back above 80% without losing much profit? (`docs/PREREGISTRATION_11.md` incl. its Part C and D addenda; `run_paid_share.py`, `run_paid_share_c.py`, `run_paid_share_d.py`; `results/paid_share_*.csv`)
**Where the drop came from** (share of funded accounts that ever get a payout; sequential Combine starts every 3 sessions, dev 2010-2022, pooled noise x3 + gap x2):
| Case | accounts paid | days start -> first payout (one account) | mean 3-year farm |
|---|---|---|---|
| old safe, basket x5, size 0.4, first payout $4,000 | 82% | 259 | $5.8k |
| basket x5, balanced (1.2, $2,000) | 68% | 58 | $4.4k |
| default split, balanced | 62% | 78 | $17.5k |
| noise-area x5 alone / gap-VWAP x5 alone, balanced | 57% / 66% | 56 / 118 | $6.2k / $8.3k |
| default split with the safe setting | 82% | 276 | $6.7k |
The ~80% was the "safe" preset. Moving to "balanced" (the user's choice, for a first payout in about 60 days instead of 260) costs about 14 points; splitting into single-strategy accounts costs about 6 more (the noise-area accounts are the more volatile and are paid in 57% of cases). The farm-level share quoted in sections 20-21 (54%) is lower again because it counts accounts still running when a 3-year farm ends as unpaid (even the old safe setting is 50% on that measure).

**Part B, 30 common settings (a1 x first-payout balance x a2): no cell qualifies** (qualify = paid >= 80%, dev mean >= 85% of current, p25 not more than $1,000 below current, worst farm not > 25% deeper). The paid share is set by the size before the first payout and the first-payout balance, and so is the profit, in the same direction: 86% paid at size 0.4/$2,000 but a mean farm of $3.1k (18% of current); 82% at 0.4/$4,000 with $6.7k-$10.4k (38-59%) and 276 days per account; 80% at size 0.6/$2,000 with $8.5k (49%); 72% at 0.8/$1,500 taking 30% with $10.9k (62%). Roughly, going from 62% to 82% paid costs about 40-60% of the profit and makes the first payout 3-4x slower.
**Part C, a smaller first payout (post-hoc, 18 cells): the one almost-free gain.** First payout at balance $1,000 instead of $2,000 at the current size: paid 66% (+4 points), mean farm $18.0k (+3%), p25 -$1.4k (current -$1.6k), farm's first payout after 128 days (current 152), 2021/2023 starts $43k/$22k (current $37k/$26k). Not a pass (still far below 80%) and within the noise of a 48-cell search, so "plausible, not proven"; withdrawing 30% instead of 50% did not help.
**Part D, two tiers (post-hoc, 12 cells): the closest.** The profit sits in the noise-area accounts ($15.8k of the $17.5k per 3-year farm) while the gap-VWAP accounts earn $1.7k and are paid 100% of the time at size 0.4. Noise-area accounts as current with first payout $1,000, gap-VWAP accounts at size 0.4: pooled paid 81% (noise-area 61%, gap-VWAP 100%; 75% for 2010-16 starts, 86% for 2017-22 starts), mean farm $17.0k (97% of current), p25 -$3.0k (current -$1.6k), worst farm unchanged, 6/10 green (current 7/10), first payout 104 days per account (current 78) and 128 days for the farm (current 152), seen starts 2021 $35k, 2023 $21k, 2025 -$6k/-$4k (current $37k, $26k, -$6k/-$4k). It misses the declared rule only on p25 (by $370). Gap size 0.6 gives 75% paid with $18.1k.
- **Read this honestly:** the 81% is reached by composition. The two gap-VWAP accounts are almost always paid, but the three noise-area accounts, where the money comes from, are still paid about 61% of the time and still mostly die. The pooled share is cosmetic for profit: it says nothing new about whether the profit-making accounts survive.
- Edge-removed placebo (descriptive): current -$2.7k, two-tier -$4.6k dev mean, both negative (the profit is edge). Full-history nets swing from $65k to $250k between neighbouring cells (the same $200k+ noise-area cluster is either hit or missed when the Combine purchase dates shift), so full-history nets were not used.
- Looks: 60 declared cells (30 + 18 + 12), the last 30 after seeing the first results; no cell passed the pre-registered rule. The default stays balanced 1.2 / $2,000 for all five accounts; the "$1,000 first payout" and the "two-tier" settings are candidates for a frozen forward test, not adoptions.

## 23. Scaling the funded risk size before the first payout (`docs/PREREGISTRATION_12.md` incl. its round-2 addendum; `run_scaling_size.py`, `run_scaling_size2.py`, `results/scaling_size_grid.csv`, `results/scaling_size_grid2.csv`)
**Why this is new.** The Topstep MLL trails the PEAK balance and locks at $0 only once the peak passes $4,500, above every first-payout balance tested. Before the first payout the cushion is therefore $4,500 minus the current drawdown from the peak: the size (a x cushion / ATR) already shrinks in a drawdown but does not react to profit, recent results or time. Engine: `FundedPlan.scale` (`("ladder", ((balance, a), ...))`, `("perf", L, f_up, f_down)`, `("ramp", d, a_start)`), applied before the first payout only; None reproduces every earlier number (tests). Chosen with the user: scale before the first payout; triggers = profit milestones, recent performance, time ramp; judged by "paid >= 80% and profit near current". Default split, a2 = 0.8, first payout $2,000 x 50%.

**Round 1 (36 cells + baseline): nothing qualifies.** Time ramps (12 cells) top out at 67% paid, at 43% of the profit (the ramps that keep the profit have 52-61% paid); recent-performance scaling (8) reaches 74% paid only by losing 70% of the profit (the base-1.2 versions keep the paid share at 60-63% and one, last 10 days x1.25 / x0.75, raises the mean to $24.5k); "start big, step down" ladders (8) give 62-70% paid with 27-51% of the profit and the shallowest worst farm (-$7.7k). The ladders that START SMALL and STEP UP (8) are the only family with both a high paid share and a high profit.
**Round 2 (30 more cells, declared after seeing round 1; only the up-ladder refined).** One cell meets the headline rule: start 0.4, step to 1.2 at +$500, to 1.6 at +$1,000:
| | today (constant 1.2) | 0.4 -> 1.2 at $500 -> 1.6 at $1,000 |
|---|---|---|
| accounts paid (pooled; noise-area / gap-VWAP) | 61.5% (57% / 66%) | 81.0% (65% / 97%) |
| mean 3-year farm / p25 / worst | $17.5k / -$1.6k / -$12.4k | $21.0k (120%) / -$2.5k / -$12.5k |
| farms started 2011-15 / 2016-20 | -$3.0k / +$38.0k | -$4.8k / +$46.8k |
| paid, 2010-16 starts / 2017-22 starts | 54% / 68% | 71% / 90% |
| days to first payout: one account / farm | 78 / 152 | 210 / 156 |
| seen starts 2021 / 2023 / 2025 | $37k / $26k / -$6k | $33k / $11k / -$6k |
| edge-removed placebo, dev mean | -$2.7k | -$0.6k |
It FAILS the stability check declared in advance (it must beat the baseline in both halves of the dev farms and keep >= 75% paid in both halves of the account-level starts): the 2011-15 farms are worse than today's (-$4.8k vs -$3.0k) and 2010-16 starts are 71% paid. The 2023 start is also much weaker ($11k vs $26k). One qualifying cell out of 66 is also about what chance alone would give. **Not adopted; the default stays constant 1.2.**
- **Price of the paid share.** Every start-small ladder waits longer per account (150-240 days from Combine start to first payout against 78), because the first stage is slow; at the farm level the qualifying cell is unchanged (156 vs 152 days) since other accounts pay meanwhile, but the larger-profit ladders are not (318-350 days).
- **The gains are regime-dependent.** All 30 round-2 up-ladders do worse than today in the 2011-15 farms (-$4.2k to -$6.7k vs -$3.0k), and in the 2016-20 farms they range from +$24k to +$74k (today +$38k; the high-top ladders far above, the low-top ones below): the extra size is a bet on a working edge. This is the same convexity that makes the funded account an option, and it is what the placebo and the weak 2011-15 / 2023 starts warn about.
- **The best profit candidate (post-hoc, not the target of this test):** start 0.8, step to 2.0 at +$500: paid 62% (unchanged), mean $30.7k (+76%), p25 -$1.8k, worst -$12.9k, 105 days per account and 152 for the farm (unchanged), seen starts 2021 $63k and 2023 $45k (today $37k / $26k), 2011-15 farms -$4.1k vs -$3.0k, edge-removed placebo -$2.0k. It misses the older "not greedy" rule (section 20: mean AND p25 up) by $124 on p25. 126 looks in total on this question (60 in section 22, 66 here), so this is "plausible, not proven".
- **What to do with it:** nothing in the default. Two frozen configurations are worth a forward test when new data exist: the paid-share candidate above and "0.8 -> 2.0 at +$500" for profit; both with kill criteria written first.

## 24. Scaling the risk size in the Combine and after the first payout (`docs/PREREGISTRATION_13.md`; `run_scaling_combine.py`, `run_scaling_after.py`, `scaling_common.py`; `results/scaling_combine_grid.csv`, `results/scaling_after_grid.csv`)
Triggers as in section 23 (profit milestones, recent performance, time ramp; plus the number of payouts after the first payout). Engine: `run_eval_firm(..., scale)` / `FirmPlan.eval_scale` (Combine contracts x a factor from the Combine's own balance, recent P&L or time) and `FundedPlan.scale2` (the post-payout multiple); both default to None (tests show neutral settings reproduce the constant sizes exactly). Acceptance (declared before running): dev mean AND p25 rise, worst farm not > 25% deeper, share of funded accounts paid not down > 5 points, beats today's mean in both halves of the dev farms (starts 2011-15 and 2016-20); Combine cells also need Combine fees per funded and days to the farm's first payout within +10%. Default split, ten 3-year dev farms.

**Combine (22 cells + baseline R = 9,600): nothing passes, and no cell even has a higher dev mean than the constant size** (baseline $17.5k, p25 -$1.6k, pass rate 22.9%, 28 days to pass, $1,146 of Combine fees per funded account).
| Setting | pass rate | days to pass | fees per funded | mean farm | p25 | worst |
|---|---|---|---|---|---|---|
| start at 0.5x, full size after 5 sessions | 22.3% | 29 | $1,202 | $17.0k | -$6.3k | -$13.2k |
| halve the size once up $3,000 | 24.9% | 41 | $1,369 | $16.0k | -$7.3k | -$10.7k |
| start at 1.5x, normal after 5 sessions | 22.1% | 29 | $1,201 | $13.8k | -$0.6k | -$10.7k |
| constant 1.25x (R = 12,000) | 20.7% | 27 | $1,231 | $13.6k | -$0.8k | -$7.8k |
| x1.25 after 10 profitable days, x0.75 after 10 losing days | 22.6% | 28 | $1,203 | $11.1k | -$1.4k | -$10.4k |
| double the size once up $6,000 | 19.0% | 28 | $1,367 | $7.1k | -$5.8k | -$11.6k |
Scaling UP with profit in the Combine lowers the pass rate (16-20% against 22.9%) and raises the fees per funded account by 17-38%: with the trailing loss limit, profit does not make the account safer, so extra size only adds breaches. Scaling DOWN after a profit milestone raises the pass rate (to 23-25%; x0.25 to 25%) but makes the pass 1.3-3.3x slower and lowers the mean farm. Recent-performance scaling and ramps lower the pass rate or the profit. The constant R was chosen on this same dev data, so the baseline starts with a selection advantage over every scaled cell.

**After the first payout (37 cells + baseline a2 = 0.8): nothing passes, and none has a higher dev mean than constant 0.8** (baseline $17.5k, p25 -$1.6k, 84% of funded accounts die, 3.10 payouts and $5.2k paid per paid account).
- **Constant levels are a sharp peak at 0.8:** 0.4 $10.9k, 0.6 $12.1k, 0.8 $17.5k, 1.0 $14.2k, 1.2 $10.2k (section 20). The 0.8 is the winner of an earlier dev grid, so $17.5k is probably overstated and the surface around it is nearer $12-14k; all the scaled cells (best $17.2k) sit inside that spread.
- **The closest cell:** start at 0.4 after the payout, step up to 1.2 only once the balance reaches $3,500: mean $17.2k (-1.7%, i.e. $305 short), p25 -$0.25k (today -$1.64k), worst -$11.2k (today -$12.4k), 67% of funded accounts die (84%), 3.76 payouts and $5.2k per paid account (3.10 and $5.2k), 53.9% paid (53.6%), farms started 2011-15 -$1.9k (today -$3.0k) and 2016-20 $36.2k (today $38.0k), seen starts 2021 $50k and 2023 $32k (today $37k and $26k), 2025 unchanged. It fails the declared rule on the mean (by $305) and on the 2016-20 half (by $1.7k); its edge-removed placebo is -$5.2k (today -$2.7k), so some of its dev mean may be structure rather than edge. Its full-history net is $83k against $233k, which is the same lucky-cluster effect seen in sections 22-23 and is not used.
- Other families: recent-performance scaling after the payout (best $16.2k, p25 -$0.6k), number-of-payouts scaling (best $14.2k) and ramps (best $14.6k) all lose profit; "start big, step down" ladders lose the most ($8.9-14.6k) and kill more accounts (90%).
- **Result:** the Combine and post-payout sizes stay as they are. The one worth freezing for a forward test is "0.4 after the payout, 1.2 at a $3,500 balance" as a lower-risk version of the same profit. No combined Combine + post-payout run was made (nothing passed in either stage, as declared).
- Looks: 59 here (22 + 37), 185 in total on the sizing questions since section 22; every acceptance rule was declared before the run; the grids and values are mine, the families and triggers the user's.

## 25. Days to payout and accounts paid: baseline against the two forward-test candidates (`run_days_candidates.py`, `results/days_candidates.csv`)
Asked by the user before choosing which setting to freeze for a forward test. Identical starts for all three (Combine starts every 3 sessions; default split, pooled noise-area x3 + gap-VWAP x2; ten 3-year farms for the whole-farm rows). Option 2 = after the first payout start 0.4 and step to 1.2 at a $3,500 balance (pre-payout unchanged); option 3 = before the first payout start 0.8 and step to 2.0 at +$500 profit (post-payout unchanged).
| One account | baseline | option 2 | option 3 |
|---|---|---|---|
| funded accounts paid, 2010-22 starts (noise-area / gap-VWAP) | 61.5% (57% / 66%) | 61.5% (57% / 66%) | 62.1% (52% / 72%) |
| funded accounts paid, 2023+ starts (noise-area / gap-VWAP) | 64.5% (64% / 65%) | 64.5% (64% / 65%) | 56.5% (64% / 40%) |
| median days Combine start -> first payout, 2010-22 (quarter faster / slower than) | 78 (54 / 125) | 78 (54 / 125) | 105 (68 / 171) |
| median days, 2023+ starts | 79 (56 / 100) | 79 (56 / 100) | 99 (62 / 199) |
| activation -> first payout, 2010-22 | 35 | 35 | 56 |
| funded accounts paid within 60 / 90 / 180 days, 2010-22 | 21% / 36% / 55% | 21% / 36% / 55% | 13% / 27% / 48% |
| funded accounts paid within 90 days, 2023+ | 40% | 40% | 26% |
| median days between 1st and 2nd payout, 2010-22 | 44 | 70 | 62 |
| payouts per paid account / paid per paid account, 2010-22 (runs to the end of 2022) | 3.1 / $4.2k | 6.1 / $5.7k | 3.8 / $8.5k |
| Whole farm (ten 3-year farms, one Combine at a time) | baseline | option 2 | option 3 |
|---|---|---|---|
| median days to the farm's first payout (quarter faster / slower) | 152 (100 / 300) | 152 (100 / 300) | 152 (104 / 259) |
| farms paid within 90 / 180 days (of 10) | 2 / 6 | 2 / 6 | 2 / 5 |
| accounts paid, farm level (unfinished accounts count as unpaid) | 53.6% | 53.9% | 42.9% |
| days to the farm's first payout, 2021 / 2023 starts | 179 / 143 | 179 / 143 | 164 / 149 |
- **Option 2 does not touch the first payout** (by construction: the size changes only after it), so its days and its paid share are identical to the baseline. It changes what happens after: the second payout comes later (70 days instead of 44, because it trades smaller after each payout) but the account survives much longer (67% against 84% of funded accounts die in the dev farms) and pays more often (6.1 payouts per paid account against 3.1 when each account runs to the end of 2022).
- **Option 3 is slower to the first payout at account level**: 105 days against 78 (2023+: 99 against 79); only 27% of funded accounts are paid within 90 days against 36% (26% against 40% for 2023+ starts). Its paid share is the same on 2010-22 starts (62%) but lower on 2023+ starts (56.5% against 64.5%: the gap-VWAP accounts fall to 40% paid). The farm's first payout is the same median (152 days) since other accounts pay meanwhile, but with a lower farm-level paid share (43% against 54%).
- Neither candidate moves the days to payout in the user's favour; option 2 is neutral on them, option 3 is worse. The baseline is the right choice if days to payout is the priority; option 2 is the only one that costs nothing on that measure.
