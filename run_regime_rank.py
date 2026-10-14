"""Overfitting diagnostic: for all dev-grid configurations, compare the DEV objective (chosen without hold-out) with the net of the same
frozen configuration started 2023-01-02 and 2025-01-02. Rank correlation near zero would mean selection on dev adds nothing."""
from __future__ import annotations
import pickle
import numpy as np, pandas as pd
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from risk_units import with_risk_unit
from run_corrected import cfg, rules, idx_of
from run_farm_dev import plan
from run_dynamic_contract import PATH

g = pd.read_pickle("results/regime_dev_grid.pkl")
streams = pickle.load(open("/home/user/data_raw/regime_streams.pkl", "rb"))
cache, h23, h25 = {}, [], []
for r in g.itertuples():
    key = (r.set, r.rule, r.unit)
    if key not in cache:
        cache[key] = with_risk_unit(streams[(r.set, r.rule)], r.unit, 40)
    s = cache[key]
    sz, pl = CombineSizing("fixed", float(r.R)), plan(float(r.a1), float(r.a2))
    sp = FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill=r.refill, pause_below=float(r.pause))
    h23.append(run_farm([s], sz, None, pl, cfg, rules, PATH, idx_of(s, "2023-01-02"), sp)["net"])
    h25.append(run_farm([s], sz, None, pl, cfg, rules, PATH, idx_of(s, "2025-01-02"), sp)["net"])
g["hold23"], g["hold25"] = h23, h25
g.to_pickle("results/regime_dev_vs_hold.pkl")
print("Spearman(dev objective, hold-out net 2023+) =", round(g.obj.rank().corr(g.hold23.rank()), 3))
print("Spearman(dev objective, hold-out net 2025+) =", round(g.obj.rank().corr(g.hold25.rank()), 3))
top = g.sort_values("obj", ascending=False)
for k in (1, 10, 50, 400):
    print(f"top-{k:<3} by dev: mean hold23 {top.head(k).hold23.mean():9.0f}  mean hold25 {top.head(k).hold25.mean():8.0f}   (all configs: {g.hold23.mean():9.0f} / {g.hold25.mean():8.0f})")
for col in ("rule", "unit", "R", "refill", "pause", "set"):
    print(f"\nhold-out 2023+ / 2025+ mean net by {col}"); print(g.groupby(col)[["hold23", "hold25"]].mean().round(0).to_string())
