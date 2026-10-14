"""Descriptive (2023+ starts, already seen): time to first payout for the baseline, the best feasible and two fast configs from run_time_to_payout.py."""
import numpy as np, pandas as pd
from combine_policy import CombinePolicy
from final_strategy import PATH, build_stream
from pricing_config import load_config
from real_engine import FundedPlan, rules_from_config, run_combine, run_funded_detail
from run_funded_payout_policy import farm_plan
s = build_stream(); cfg = load_config(); rules = rules_from_config(cfg, PATH)
d = pd.DatetimeIndex(s.dates); x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist(); nd = s.desired_micros(9600.0)
i23 = int(np.searchsorted(s.dates, np.datetime64("2023-01-01")))
for lab, t1, a1 in (("baseline t1=3000 a1=0.4", 3000., .4), ("best feasible t1=2000 a1=0.8", 2000., .8), ("fast t1=1500 a1=1.2", 1500., 1.2), ("fastest t1=500 a1=2.0", 500., 2.0)):
    pl = FundedPlan("p", "standard", t1, .5, 1., 0., .5, 1., 99, a1, 1.6)
    days, nf, npaid, n = [], 0, 0, 0
    for i0 in range(i23, len(d) - 150, 3):
        st, tau = run_combine(x, lo, nd, i0, rules); n += 1
        if st != 1: continue
        nf += 1; tm = []; run_funded_detail(x, lo, nd, i0 + tau, rules, pl, atr, tm)
        if tm: npaid += 1; days.append((d[tm[0]] - d[i0]).days)
    net = [round(farm_plan(s, CombinePolicy(R=9600.), pl, int(np.searchsorted(s.dates, np.datetime64(dt)))) ["net"]) for dt in ("2023-01-02", "2024-01-02", "2025-01-02")]
    print(f"{lab:<30} P(paid|funded) {npaid/max(nf,1):.2f}  median days start->payout {np.median(days):.0f}  share of starts paid <=90d {np.mean(np.array(days)<=90)*npaid/n:.2f}  farm net from 2023/2024/2025 starts {net}")
