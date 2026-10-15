"""Stage 1b (DEV only): the funded side. How much of the balance to withdraw (w, max 0.5) and the balance needed before later payouts (t2), together with
post-payout size a2.  Final NQ basket, neutral fixed-risk Combine (R=9600), hyst refill, first payout t1=3000, a1=0.4.  36 combinations; same dev objective."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from combine_policy import CombinePolicy
from final_strategy import PATH, build_stream
from pricing_config import load_config
from real_engine import FundedPlan, rules_from_config
from run_combine_policy_study import farm
from run_regime_study import windows

pd.set_option("display.width", 250, "display.max_columns", 40)


def farm_plan(s, pol, pl, i0, n_end=None):
    from contract_eval import CombineSizing
    from farm_real import FarmSpec, run_farm
    cfg = load_config(); rules = rules_from_config(cfg, PATH)
    spec = FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill="hyst", combine_policy=pol)
    return run_farm([s], CombineSizing("fixed", pol.R), None, pl, cfg, rules, PATH, i0, spec, n_end=n_end)


def main() -> None:
    s = build_stream(); wins = windows(s)
    rows = []
    for R, w, t2, a2 in itertools.product((9600.0,), (0.25, 0.35, 0.5), (0.0, 500.0, 1000.0, 2000.0), (0.4, 1.0, 1.6)):
        pol = CombinePolicy(R=R)
        pl = FundedPlan("p", "standard", 3000.0, w, 1.0, t2, w, 1.0, 99, 0.4, a2)
        res = [farm_plan(s, pol, pl, int(o), int(o) + 756) for o in wins]
        net = np.array([r["net"] for r in res])
        full = farm_plan(s, pol, pl, int(np.flatnonzero(s.ok)[0]))
        rows.append(dict(w=w, t2=t2, a2=a2, dev_mean=net.mean(), dev_p10=np.percentile(net, 10), obj=0.5 * (net.mean() + np.percentile(net, 10)), pos=float((net > 0).mean()),
                         payouts_per_funded=full["n_payouts"] / max(full["n_funded"], 1), paid_per_funded=full["payouts"] / max(full["n_funded"], 1)))
    g = pd.DataFrame(rows); g.to_csv("results/funded_payout_policy_dev_grid.csv", index=False)
    print(g.sort_values("obj", ascending=False).head(10).round(2).to_string(index=False))
    for col in ("w", "t2", "a2"):
        print(f"\nmean by {col}:"); print(g.groupby(col)[["dev_mean", "dev_p10", "obj", "payouts_per_funded", "paid_per_funded"]].mean().round(2).to_string())
    print("\nbaseline (w=0.5, t2=0, a2=1.6 / 0.4):"); print(g[(g.w == 0.5) & (g.t2 == 0.0) & (g.a2.isin([0.4, 1.6]))].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
