"""Audit: the analyst's hold-out start range stops at n-300. What happens for the (excluded) latest starts, and per start-quarter?"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from pricing_config import load_config
from real_engine import rules_from_config
from contract_eval import CombineSizing, contract_metrics
from portfolio import DEV_END, basket
from run_dynamic_contract import PATH, plan_grid, sleeves
from real_engine import FundedPlan

cfg = load_config(); RULES = rules_from_config(cfg, PATH)
S = sleeves(); nq = basket([S[0]], name="nq_noise")
n = len(nq.dates); n_dev = int(np.searchsorted(nq.dates, DEV_END, side="right"))
plan = next(p for p, r in plan_grid() if p.name == "stan t1=3000 dyn a1=0.1 a2=0.05")
sz = CombineSizing("fixed", 6400)
d = pd.DatetimeIndex(nq.dates)
print("n", n, "n_dev", n_dev, "analyst starts end at idx", n - 300, d[n - 301])
for lab, a, b in (("analyst range", n_dev, n - 300), ("tail n-300..n-60", n - 300, n - 60), ("2025-01 on", int(np.searchsorted(nq.dates, np.datetime64("2025-01-01"))), n - 60),
                  ("2023 starts", n_dev, int(np.searchsorted(nq.dates, np.datetime64("2024-01-01")))),
                  ("2024 starts", int(np.searchsorted(nq.dates, np.datetime64("2024-01-01"))), int(np.searchsorted(nq.dates, np.datetime64("2025-01-01"))))):
    st_ = np.arange(a, b)
    grp = (d[st_].year * 4 + (d[st_].month - 1) // 3).to_numpy()
    m = contract_metrics(nq, nq, sz, plan, None, cfg, RULES, st_, grp)
    print(f"{lab:<18} starts {len(st_):>4} p_pass {m['p_pass']:.2f} n_funded {m['n_funded']:>3} p_first {m['p_first']:.2f} payouts/funded {m['payouts_per_funded']:.2f} extr/funded {m['extraction_per_funded']:.0f} cost/funded {m['cost_funded']:.0f} EV/att {m['ev_att']:.0f}")
