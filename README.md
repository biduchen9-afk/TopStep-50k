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
