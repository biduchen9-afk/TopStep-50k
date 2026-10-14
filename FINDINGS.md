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
