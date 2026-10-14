"""Can the farm be made to pay MONTHLY instead of in long droughts? Sweep the funded-account payout trigger (t1), payout path
(standard 5 winning days / consistency 3 days), post-payout aggressiveness (a2) and Combine concurrency, with the dev-selected stream, R=9600, hyst refill.
Selection metrics are DEV (12 windows, 2011-2022). Full-history monthly coverage is DESCRIPTIVE (it includes 2023+, already seen)."""
from __future__ import annotations
import itertools, sys
import numpy as np, pandas as pd
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from real_engine import FundedPlan
from run_corrected import cfg, rules, idx_of
from run_dynamic_contract import PATH
from run_regime_study import windows
from build_dashboard_data import monthly, idx_first
sys.argv = sys.argv[:1]
from build_dashboard_v2 import SEL, top

SZ = CombineSizing("fixed", float(top["R"]))
pd.set_option("display.width", 250, "display.max_columns", 40)


def fplan(path, t1, a1, a2):
    return FundedPlan(f"{path} t1={t1} a2={a2}", path, float(t1), 0.5, 1.0, 0.0, 0.5, 1.0, 99, a1, a2)


def coverage(res, s):
    m = pd.DataFrame(monthly(s, res, "x"))
    m = m[m.m >= "2012-01"]
    pay = m.payouts.to_numpy()
    has = pay > 0
    # longest run of months without a payout
    run = best = 0
    for h in has:
        run = 0 if h else run + 1
        best = max(best, run)
    m23 = m[m.m >= "2023-01"]
    return dict(months_paid=float(has.mean()), longest_drought_m=int(best), mean_month=float(pay.mean()), cv_month=float(pay.std() / pay.mean()) if pay.mean() > 0 else np.nan,
                months_paid_23=float((m23.payouts > 0).mean()), net_all=float(m.payouts.sum() - (m.eval_fees + m.activation_fees).sum()))


def main() -> None:
    s = SEL
    wins = windows(s)
    rows = []
    for path, t1, a2, me in itertools.product(("standard", "consistency"), (500, 1000, 1500, 2000, 3000), (0.4, 1.6), (10, 20)):
        pl = fplan(path, t1, 0.4, a2)
        spec = FarmSpec(max_evals=me, max_funded=5, p_est=1.0, refill="hyst", pause_below=0.0)
        net = np.array([run_farm([s], SZ, None, pl, cfg, rules, PATH, int(o), spec, n_end=int(o) + 756)["net"] for o in wins])
        full = run_farm([s], SZ, None, pl, cfg, rules, PATH, idx_first(s), spec, log=True)
        cov = coverage(full, s)
        rows.append(dict(path=path, t1=t1, a2=a2, max_evals=me, dev_mean=net.mean(), dev_p10=np.percentile(net, 10), dev_pos=float((net > 0).mean()), payouts=full["n_payouts"], funded=full["n_funded"], **cov))
    g = pd.DataFrame(rows)
    g.to_csv("results/monthly_payout_sweep.csv", index=False)
    print(g.sort_values("months_paid", ascending=False).round(2).head(12).to_string(index=False))
    print("\nbaseline (standard, t1=3000, a2=1.6, 10 evals):"); print(g[(g.path == "standard") & (g.t1 == 3000) & (g.a2 == 1.6) & (g.max_evals == 10)].round(2).to_string(index=False))
    print("\nby t1:"); print(g.groupby("t1")[["dev_mean", "dev_p10", "months_paid", "longest_drought_m", "mean_month", "net_all"]].mean().round(2).to_string())
    print("\nby path:"); print(g.groupby("path")[["dev_mean", "dev_p10", "months_paid", "longest_drought_m", "mean_month", "net_all"]].mean().round(2).to_string())
    print("\nby max_evals:"); print(g.groupby("max_evals")[["dev_mean", "dev_p10", "months_paid", "longest_drought_m", "mean_month", "net_all"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
