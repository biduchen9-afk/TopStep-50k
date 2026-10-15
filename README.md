# TopStep-50k

**Start with [FINDINGS.md](FINDINGS.md)** for the verdict, the final strategy ([`final_strategy.py`](final_strategy.py)) and the evidence. The rest of this README is the engine documentation and the study log.

Monte Carlo valuation engine for a farm of Topstep 150K Combine / XFA accounts
(2026 fee paths: `standard`, `no_activation`, `no_activation_dll`).

```bash
pip install -r requirements.txt
python -m pytest -q                 # invariant tests
python run_comparison.py --sweep    # Sharpe 0 / 0.5 / 1.5 comparison + breakeven Sharpe
python run_comparison.py --sharpes 0.5 --policy wait_for_rebill
```

Modules: `pricing_config.py` (YAML -> dataclasses), `combine_simulator.py`
(daily P&L with Brownian-bridge intraday low for the DLL, trailing EOD MLL,
50% consistency rule), `valuation_engine.py` (p, E[tau], mu_attempt,
Cost_funded, EV_attempt, EV_cycle, mu_ext*, breakeven Sharpe).

`sharpe_annual` is the strategy's **gross** Sharpe; friction ($7/day by default)
is subtracted inside the simulation. XFA payout mechanics are not in the spec,
so they are explicit assumptions in the `xfa:` block of `config.yaml`.

## Real-data study (NQ / ES / GC 1-minute Databento bars, 2010-2026)

Raw files are not in git. Download the three `*_databento.txt` assets from the `v1.0.0` release into
`/home/user/data_raw` (or set `TOPSTEP_RAW_DIR`). The clock in the files is US Eastern.

| File | Purpose |
|---|---|
| `market_data.py` | Loads bars; builds 18:00 ET -> 16:00 ET sessions (VWAP anchor = 18:00 ET); flags contract-roll sessions and skips them. Session maths is tz-aware, so a bot in Asia/Phnom_Penh stays correct through US DST (open = 05:00 or 06:00 ICT). |
| `intraday_strategies.py` | No-edge control, VWAP+ATR momentum, opening-range breakout. Signals on bar close, fills next open. |
| `real_engine.py` | Combine and funded-account rules replayed on real daily P&L, with cushion/payout variants. |
| `run_real_study.py` / `run_decoupled_study.py` | Sweeps -> `results/*.csv`. The decoupled study sizes the Combine and funded phases separately. |

Everything is measured from every historical start date; CIs resample whole calendar years.

## Hold-out and final checks (config-driven)

`config.yaml` now drives the real-data engine: `topstep_official` (verified rules with toggles back to the
spec's older assumptions), `friction.slippage_ticks_rt`, fee paths and billing cycle.

| Script | What it does |
|---|---|
| `run_sweep.py` | 324 strategy variants (ES/NQ/GC), IS/OOS, rank stability, walk-forward, cost stress |
| `run_holdout.py` | selects strategy, sizing and funded plan on data <= 2022, runs once on 2023-2026 |
| `run_final_checks.py` | slippage stress, ES/GC, breach policies, spec table for the 3 fee paths, why-profit-is-low, improvements |
| `chain_real.py` | sequential rinse-and-repeat with cancel_and_reopen / immediate_reset / wait_for_rebill billing clock |
| `real_engine.py` | Combine + funded rules; funded plans with repeated payouts (`max_payouts`) |

Results CSVs are in `results/`. Treat the hold-out as used: it has been run more than once.

## The contract as specified (take every eligible payout, trade until the account dies, repeat)

`run_rinse_repeat.py`, `run_noedge_contract.py`, `run_contract_final.py`, `run_contract_robustness.py`.
After the first payout (50% of balance at trigger t1) every later payout is taken the moment it is eligible, at the
maximum allowed; the account is traded until it dies; then a new Combine starts. Selection uses 2010-2022 only, the
hold-out (2023-2026) uses fresh coin-flip seeds. See `results/contract_*.csv`.

## Literature sleeves, portfolio, dynamic sizing and the account farm

Sources (found by web search; SSRN/arXiv): Gao-Han-Li-Zhou "Market Intraday Momentum" (SSRN 2440866); Zarattini-Aziz-Barbon
"Beat the Market" (SSRN 4824172); Zarattini-Barbon-Aziz "A Profitable Day Trading Strategy" (SSRN 4729284); Moskowitz-Ooi-Pedersen
"Time Series Momentum" (SSRN 2089463). Caution: arXiv 2605.04004 argues OHLCV intraday momentum in MNQ has structural limits (not reviewed here).

| Script / module | What it does |
|---|---|
| `academic_strategies.py` | intraday momentum (last half hour), noise-area momentum, VWAP fade, trend / volatility gates |
| `run_sleeve_sweep.py` | 300 sleeve variants x ES/NQ/GC, dev vs hold-out |
| `portfolio.py`, `run_portfolio.py` | micro-basket portfolios, dev-only weights, switching rules |
| `contract_eval.py`, `run_dynamic_contract.py` | fixed vs cushion-proportional (dynamic) sizing, dev grid |
| `run_dynamic_holdout.py` | frozen per-start + sequential + farm hold-out |
| `farm_real.py`, `run_farm_dev.py`, `run_farm_scaling.py`, `run_farm_robustness.py` | up to 20 evals / 5 funded accounts on one market path |

Dynamic sizing: risk per ATR move = a x (balance - MLL floor): a1 before the first payout, a2 after it (floor is $0 then), so a funded account
starts carefully and fades slowly instead of dying outright.

## Audit (project-wide integrity review)

Independent reviews covered the signals, the account/payout engine, placebo/negative controls and the hold-out protocol.

- **No cheating found.** No look-ahead (signals use bar-close data and fill at the next open; perturbing future bars leaves earlier
  signals unchanged), no rule bugs against `reference_impl.py` (an independent re-implementation, see `test_engine_audit.py`), and
  mutation testing now catches all 37 injected bugs. The suite is 38 tests (`pytest`).
- **Fixed during the audit:** costs were too light (micro round turn now $1.22 plus 1 tick slippage; NQ noise-area Sharpe 0.89 -> 0.56,
  hold-out 1.21); renewal fee billed after the data ended; `farm_real` now rejects misaligned session dates; the basket Sharpe no longer
  produces NaN when ATR is zero.
- **Correction:** an earlier note said the coin-flip control loses under dynamic sizing. The placebo audit showed that is false for the
  headline plan: the zero-edge control also earns money in the recent high-volatility era, because the rules cap losses at the fees
  while payouts are uncapped. Judge any strategy by how far it sits above the control.
- **Caveats that remain:** the hold-out was never clean (the a2=0.4 result of about $53k was picked after seeing 2023-26; the
  pre-registered plan gave about $10.5k); profit is concentrated in 2018 and 2020-24; farms started in 2025 paid nothing for NQ;
  payouts are assumed approved instantly; accounts share one market path; 34 distinct funding dates make the statistics look stronger
  than they are. A clean test needs data after the end of this history with a frozen config and kill criteria set in advance.
- **Dashboard:** `build_dashboard_data.py` writes `results/dashboard_data.json` (monthly payouts, fees, reinvestment, accounts open).
- **Topstep default (section 20 of FINDINGS.md):** 3 noise-area accounts + 2 gap-VWAP accounts, balanced sizing (`final_strategy.ACCOUNT_GROUPS`); `multi_firm.py` is group-aware (`FirmPlan.firm`, `Policy.group_aware`), `kelly_sizing.py` holds the Kelly unit stream, and `build_dashboard_v4.py` writes the dashboard datasets (default variant, splits per firm, scaling pipeline month by month, Kelly and days-to-payout tables).

## Regime study (why the edge faded in 2025-26, and what we tried)

Scripts: `diag/regime_diag.py`, `diag/sizing_diag.py`, `vwap_variants.py` + `run_vwap_variants.py`, `switching.py`, `risk_units.py`,
`run_regime_study.py` (dev grid), `run_regime_ext.py` (boundary check), `run_regime_rank.py` (dev-vs-hold-out rank check),
`run_regime_holdout.py`, `run_regime_walkforward.py`, `build_dashboard_v2.py`. Farm rules added to `farm_real.FarmSpec`: `refill` (all / hyst / buffer) and a trailing pass-rate throttle (`pause_below`).

- **Why 2023-24 held and 2025-26 did not:** in dollars per micro the NQ noise-area edge was about the same in 2025 as in 2024, but daily dispersion roughly doubled and 2025 profit came from a handful of days (the 5 best days exceed the year's total). Trailing variance ratio, autocorrelation and range had no predictive power for the next day's return, so the fade could not be forecast.
- **Sizing:** scaling by trailing strategy P&L std instead of ATR did not help on dev data; larger Combine risk (R up to ~9,600 per ATR unit) helped at the farm level because Combines resolve faster.
- **VWAP variants (156 trials logged in `results/vwap_variant_trials.csv`):** gap-and-first-30-minutes filtered VWAP on NQ was the only clear winner (dev 0.89, hold-out 0.87, neighbours 0.74-0.89); band-fade blends and multi-anchor VWAP did not work; ES/GC variants were weak.
- **Switching:** gating or tilting sleeves by trailing Sharpe did not beat static dev weights on dev data (gates were worse).
- **Farm rules:** the pass-rate throttle and the "buffer" refill rule cut fees and losses in weak periods but also give up payouts; the "hyst" rule (no new Combines once full, until 3 slots empty) is about neutral.
- **Selection (dev only, 4,752 configs):** NQ noise + NQ VWAP + NQ gap30 sleeves, static weights, ATR sizing, Combine R=9,600, funded a1=0.4/a2=1.6, hyst refill, no throttle. Rank correlation between dev objective and hold-out net across all configs is only 0.16 (2023+) and 0.30 (2025+): selection adds modest value.
- **Hold-out (not clean):** from 2023-01 the selected farm nets about $19k vs about $2k for the previous headline; from 2025-01 about $5k vs -$16k; a start in mid-2025 is slightly negative for both.
- **Walk-forward (re-pick every year on earlier data, 3-year farms):** positive in 9 of 11 start years; mean $15.9k vs $18.8k for the previous headline across 2016-26, but +$4.7k vs -$6.1k for 2023-26 starts.
- **No-edge control on the same rules** loses over the full history, so the gain is not purely rule structure.

## Can the farm pay monthly? (`run_monthly_payout.py`, `run_monthly_diversified.py`)

Not reliably. Payouts are lumpy because every account trades the same stream on one market path, so they become eligible in the same months. With the dev-selected stream (standard path, t1=3000, a2=1.6) only ~17% of months since 2012 contain a payout (longest gap 50 months). Lowering post-payout aggressiveness to a2=0.4 raises coverage to ~26% with similar dev net; lowering the payout trigger t1 barely helps and costs dev net; the consistency path pays less often (~7% of months). Mixing ES+GC accounts raises coverage to ~39% (longest gap 19 months) but turns dev net negative. Combine concurrency: with p_est=1.0 at most 5 Combines are ever open, and raising max_evals to 20 changes nothing; forcing more concurrent Combines (p_est 0.4 / 0.2) lowers dev mean from ~$22k to ~$9k / ~$0.7k because funded slots (5) are the bottleneck.
