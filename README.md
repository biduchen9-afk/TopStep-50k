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
