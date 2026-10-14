"""Boundary check (DEV only): the dev-best Combine risk keeps landing on the largest R tried. Extend R and report the dev objective, Combine pass rate,
breach rate and days to pass; then (descriptive) the same configurations from the hold-out start dates."""
from __future__ import annotations
import numpy as np, pandas as pd
from combine_policy import CombinePolicy, run_combine_policy
from final_strategy import PATH, build_stream
from portfolio import DEV_END
from pricing_config import load_config
from real_engine import rules_from_config
from run_combine_policy_study import farm
from run_regime_study import windows

s = build_stream(); wins = windows(s); cfg = load_config(); rules = rules_from_config(cfg, PATH)
n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist()
first = int(np.flatnonzero(s.ok)[0]) + 260
rows = []
for R in (6400., 9600., 12800., 16000., 19200., 25600., 32000.):
    pol = CombinePolicy(R=R)
    for a2 in (0.4, 1.6):
        net = np.array([farm(s, pol, a2, int(o), int(o) + 756)["net"] for o in wins])
        st = [run_combine_policy(x, lo, atr, i, rules, pol) for i in range(first, n_dev - 300, 5)]
        hold = [round(farm(s, pol, a2, int(np.searchsorted(s.dates, np.datetime64(d))))["net"]) for d in ("2023-01-02", "2024-01-02", "2025-01-02", "2025-07-01")]
        rows.append(dict(R=R, a2=a2, dev_mean=net.mean(), dev_p10=np.percentile(net, 10), obj=0.5 * (net.mean() + np.percentile(net, 10)), pos=float((net > 0).mean()),
                         pass_rate=np.mean([a == 1 for a, _ in st]), days_to_pass=np.mean([t for a, t in st if a == 1]), hold_starts=str(hold)))
g = pd.DataFrame(rows); g.to_csv("results/combine_risk_boundary.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 60)
print(g.round(2).to_string(index=False))
