"""Pre-registration 5, Part A: speed-vs-value frontier for the first payout (Topstep). See docs/PREREGISTRATION_5.md."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from combine_policy import CombinePolicy
from final_strategy import PATH, build_stream
from portfolio import DEV_END
from pricing_config import load_config
from real_engine import FundedPlan, rules_from_config, run_combine, run_funded_detail
from run_funded_payout_policy import farm_plan

pd.set_option("display.width", 250, "display.max_columns", 40)
YEARS = list(range(2011, 2021))


def main(a1s=(0.4, 0.8, 1.2, 1.6, 2.4), t1s=(500., 1000., 1500., 2000., 3000., 4000.), w1s=(0.25, 0.5), a2s=(0.8,), tag="") -> None:
    s = build_stream(); rules = rules_from_config(load_config(), PATH); d = pd.DatetimeIndex(s.dates)
    n_dev = int(np.searchsorted(s.dates, DEV_END, side="right")); n = len(d)
    idx = lambda dt: int(np.searchsorted(s.dates, np.datetime64(dt)))
    nd = s.desired_micros(9600.0)
    x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist()
    first = int(np.flatnonzero(s.ok)[0]) + 260
    starts_dev = list(range(first, n_dev - 300, 3)); starts_all = list(range(first, n - 300, 3))
    comb_dev = [run_combine(x[:n_dev], lo[:n_dev], nd[:n_dev], i, rules) for i in starts_dev]
    comb_all = [run_combine(x, lo, nd, i, rules) for i in starts_all]
    pol = CombinePolicy(R=9600.0)

    def metrics(pl, starts, comb, ne):
        res = []
        for i0, (st, tau) in zip(starts, comb):
            if st != 1:
                res.append((i0, False, np.nan, np.nan)); continue
            tm = []; pays, *_ = run_funded_detail(x[:ne], lo[:ne], nd[:ne], i0 + tau, rules, pl, atr[:ne], tm)
            res.append((i0, True, (d[min(tm[0], ne - 1)] - d[i0]).days if tm else np.nan, pays[0] if pays else np.nan))
        return res

    def summarize(res):
        nstart = len(res); fund = [r for r in res if r[1]]; paid = [r for r in res if r[1] and not np.isnan(r[2])]
        days = np.array([r[2] for r in paid])
        return dict(p_paid=len(paid) / max(len(fund), 1), med_days=float(np.median(days)) if len(days) else np.nan, first_payout=float(np.mean([r[3] for r in paid])) if paid else np.nan,
                    in90=float(np.sum(days <= 90)) / nstart, in120=float(np.sum(days <= 120)) / nstart, in180=float(np.sum(days <= 180)) / nstart)

    rows = []
    for a1, t1, w1, a2 in itertools.product(a1s, t1s, w1s, a2s):
        pl = FundedPlan("p", "standard", t1, w1, 1.0, 0.0, 0.5, 1.0, 99, a1, a2)
        m = summarize(metrics(pl, starts_dev, comb_dev, n_dev))
        nets = np.array([farm_plan(s, pol, pl, idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["net"] for y in YEARS])
        mh = summarize(metrics(pl, [i for i in starts_all if i >= idx("2023-01-01")], [c for i, c in zip(starts_all, comb_all) if i >= idx("2023-01-01")], n))
        rows.append(dict(a1=a1, t1=t1, w1=w1, a2=a2, **m, dev_mean=nets.mean(), dev_p25=np.percentile(nets, 25), green=int((nets > 0).sum()), hold_med_days=mh["med_days"], hold_p_paid=mh["p_paid"], hold_in120=mh["in120"]))
    g = pd.DataFrame(rows); g.to_csv(f"results/payout_speed_grid{tag}.csv", index=False)
    print("Unconstrained optimum (highest dev p25 with mean >= 5k):")
    feas = g[g.dev_mean >= 5000].sort_values("dev_p25", ascending=False)
    cols = ["a1", "t1", "w1", "a2", "med_days", "p_paid", "first_payout", "in90", "in120", "in180", "dev_mean", "dev_p25", "green", "hold_med_days", "hold_p_paid"]
    print(feas.head(3)[cols].round(2).to_string(index=False))
    print("\nFRONTIER (best dev p25 under a median-days budget, dev mean >= 5k):")
    fr = []
    for B in (90, 120, 150, 210):
        sub = feas[feas.med_days <= B]
        if len(sub):
            r = sub.iloc[0].copy(); r["budget"] = B; fr.append(r)
        else:
            print(f"  budget {B}: no configuration meets the budget with dev mean >= 5k")
    if fr:
        F = pd.DataFrame(fr); print(F[["budget"] + cols].round(2).to_string(index=False)); F.to_csv(f"results/payout_speed_frontier{tag}.csv", index=False)
    print("\nFastest overall (any value):"); print(g.sort_values("med_days").head(5)[cols].round(2).to_string(index=False))
    print("\nmean by a1:"); print(g.groupby("a1")[["med_days", "p_paid", "dev_mean", "dev_p25"]].mean().round(2).to_string())
    print("\nmean by t1:"); print(g.groupby("t1")[["med_days", "p_paid", "first_payout", "dev_mean", "dev_p25"]].mean().round(2).to_string())
    print("\nmean by w1:"); print(g.groupby("w1")[["med_days", "p_paid", "dev_mean", "dev_p25"]].mean().round(2).to_string())
    # by start-year view for the optimum and the B=120 point
    pts = {"optimum": feas.iloc[0]}
    if fr and any(r["budget"] == 120 for r in fr):
        pts["B=120"] = [r for r in fr if r["budget"] == 120][0]
    yr = pd.DatetimeIndex(s.dates[starts_all]).year
    for k, r in pts.items():
        pl = FundedPlan("p", "standard", r.t1, r.w1, 1.0, 0.0, 0.5, 1.0, 99, r.a1, r.a2)
        res = metrics(pl, starts_all, comb_all, n); df = pd.DataFrame(res, columns=["i0", "funded", "days", "pay"]); df["y"] = yr
        t = df.groupby("y").apply(lambda q: pd.Series(dict(funded=int(q.funded.sum()), paid=int(q.days.notna().sum()), med_days=q.days.median())), include_groups=False)
        print(f"\nby start year ({k}: a1={r.a1}, t1={r.t1}, w1={r.w1}):"); print(t.round(0).T.to_string())
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ext":
        main((2.4, 3.2, 4.0, 6.0), (4000., 5000., 6000., 8000.), (0.5,), (0.8, 1.6), "_ext")
    else:
        main()
