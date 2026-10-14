# TopStep-50k

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
