"""Pre-registration 2, Part A (see docs/PREREGISTRATION_2.md): how short can the Combine-start -> first-payout time be made, and at what cost?"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from combine_policy import CombinePolicy
from final_strategy import PATH, build_stream
from portfolio import DEV_END
from pricing_config import load_config
from real_engine import FundedPlan, rules_from_config, run_combine, run_funded_detail
from run_funded_payout_policy import farm_plan
from run_regime_study import windows

pd.set_option("display.width", 250, "display.max_columns", 40)


def main() -> None:
    s = build_stream(); cfg = load_config(); rules = rules_from_config(cfg, PATH)
    d = pd.DatetimeIndex(s.dates); n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
    x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist(); nd = s.desired_micros(9600.0)
    cut = lambda a: a[:n_dev]
    xd, lod, atrd, ndd = cut(x), cut(lo), cut(atr), cut(nd)
    first = int(np.flatnonzero(s.ok)[0]) + 260
    starts = list(range(first, n_dev - 300, 3))
    comb = [run_combine(xd, lod, ndd, i0, rules) for i0 in starts]
    wins = windows(s)
    pol = CombinePolicy(R=9600.0)
    rows = []
    for path, t1, a1 in itertools.product(("standard", "consistency"), (500.0, 1000.0, 1500.0, 2000.0, 3000.0), (0.4, 0.8, 1.2, 2.0)):
        pl = FundedPlan("p", path, t1, 0.5, 1.0, 0.0, 0.5, 1.0, 99, a1, 1.6)
        n_fund = n_paid = within90 = 0; d_all, d_act = [], []
        for i0, (st, tau) in zip(starts, comb):
            if st != 1:
                continue
            n_fund += 1
            tm = []
            run_funded_detail(xd, lod, ndd, i0 + tau, rules, pl, atrd, tm)
            if tm:
                n_paid += 1
                days = (d[min(tm[0], n_dev - 1)] - d[i0]).days
                d_all.append(days); d_act.append((d[min(tm[0], n_dev - 1)] - d[min(i0 + tau, n_dev - 1)]).days)
                within90 += days <= 90
        net = np.array([farm_plan(s, pol, pl, int(o), int(o) + 756)["net"] for o in wins])
        rows.append(dict(path=path, t1=t1, a1=a1, p_paid_given_funded=n_paid / max(n_fund, 1), median_days_start_to_payout=float(np.median(d_all)) if d_all else np.nan,
                         median_days_activation_to_payout=float(np.median(d_act)) if d_act else np.nan, share_of_starts_paid_within_90d=within90 / len(starts),
                         share_of_paid_within_90d=within90 / max(n_paid, 1), dev_mean=net.mean(), dev_p10=np.percentile(net, 10), obj=0.5 * (net.mean() + np.percentile(net, 10))))
    g = pd.DataFrame(rows); g.to_csv("results/time_to_payout_dev_grid.csv", index=False)
    feas = g[g.obj >= 5000].sort_values("median_days_start_to_payout")
    print(f"combine pass rate on these dev starts: {np.mean([c[0] == 1 for c in comb]):.2f}; median days Combine start -> pass: {np.median([c[1] for c in comb if c[0] == 1]):.0f} sessions")
    print("\nbaseline (standard, t1=3000, a1=0.4):"); print(g[(g.path == 'standard') & (g.t1 == 3000) & (g.a1 == 0.4)].round(2).to_string(index=False))
    print("\nFEASIBLE (dev objective >= 5000), fastest first:"); print(feas.head(12).round(2).to_string(index=False))
    print("\nFastest overall regardless of cost:"); print(g.sort_values("median_days_start_to_payout").head(6).round(2).to_string(index=False))
    for col in ("path", "t1", "a1"):
        print(f"\nmean by {col}:"); print(g.groupby(col)[["p_paid_given_funded", "median_days_start_to_payout", "share_of_starts_paid_within_90d", "obj"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
