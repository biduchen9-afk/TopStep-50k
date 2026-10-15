"""Boundary check for stage 1: the dev-best Combine risk R was the largest value tried (6400). Extend R to 8000 and 9600 for the two
sleeve sets and two switching rules that led the grid, same dev windows and same objective (still DEV only)."""
from __future__ import annotations
import itertools, pickle
import numpy as np, pandas as pd
from contract_eval import CombineSizing
from farm_real import FarmSpec
from risk_units import with_risk_unit
from run_farm_dev import plan
from run_regime_study import evaluate, windows

def main():
    streams = pickle.load(open("/home/user/data_raw/regime_streams.pkl", "rb"))
    rows = []
    for (sname, rule), base in streams.items():
        if not (sname.startswith(("P1", "P2")) and rule in ("static", "tilt")):
            continue
        for ru in ("atr", "pnlstd"):
            s = with_risk_unit(base, ru, 40); wins = windows(s)
            for R in (8000, 9600):
                sz = CombineSizing("fixed", R)
                for a1, a2 in itertools.product((0.2, 0.4), (0.4, 0.8, 1.6)):
                    for rf, pb in [(rf, pb) for rf in ("all", "hyst", "buffer") for pb in (0.0, 0.10, 0.15)]:
                        spec = FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill=rf, pause_below=pb)
                        net = np.array([evaluate(s, o, sz, plan(a1, a2), spec)["net"] for o in wins])
                        rows.append(dict(set=sname, rule=rule, unit=ru, R=R, a1=a1, a2=a2, refill=rf, pause=pb, mean=net.mean(), p10=np.percentile(net, 10),
                                         worst=net.min(), obj=0.5 * (net.mean() + np.percentile(net, 10)), pos=float((net > 0).mean())))
        print(sname, rule, "done", flush=True)
    g = pd.DataFrame(rows); g.to_pickle("results/regime_dev_grid_ext.pkl"); g.to_csv("results/regime_dev_grid_ext.csv", index=False)
    print(len(g), "extra configurations"); print(g.sort_values("obj", ascending=False).head(8).round(2).to_string(index=False))
    print(g.groupby("R")[["obj", "mean", "p10"]].mean().round(0))

if __name__ == "__main__":
    main()
