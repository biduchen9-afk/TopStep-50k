"""Aggressive Combine, separately-sized funded account. Ranks by P(first payout) where the first
payout covers the average cost of getting funded. Reports pre-2020 vs 2020+ start dates too."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from real_engine import FundedVariant, combine_outcomes, run_funded, scale, year_bootstrap
from pricing_config import load_config
from run_real_study import MIN_REMAINING, build_streams

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="no_activation_dll")
    ap.add_argument("--sigma-combine", type=float, nargs="+", default=[1000, 1500, 2200])
    ap.add_argument("--sigma-funded", type=float, nargs="+", default=[150, 300, 600])
    a = ap.parse_args()
    cfg = load_config(); pth = cfg.paths[a.path]; S = build_streams(); rows = []
    plans = []
    for t1 in (1500, 2000, 3000):
        plans.append(FundedVariant(f"t{t1} close", t1=t1, w1=0.5))
        plans.append(FundedVariant(f"t{t1} +2nd cons half", t1=t1, w1=0.5, size2=0.5, t2=t1, w2=0.5))
        plans.append(FundedVariant(f"t{t1} +2nd cons ALL", t1=t1, w1=0.5, size2=0.5, t2=t1, w2=1.0))
    for sn, s in S.items():
        n = len(s.dates); starts = np.arange(int(np.flatnonzero(s.ok)[0]), n - MIN_REMAINING)
        yrs = pd.DatetimeIndex(s.dates[starts]).year.to_numpy()
        for sc in a.sigma_combine:
            xc, mc = scale(s, s.risk_unit_for_sigma(sc))
            c = combine_outcomes(xc, mc, cfg, a.path, starts); passed = c["status"] == 1
            for sf in a.sigma_funded:
                xf, mf = scale(s, s.risk_unit_for_sigma(sf))
                for v in plans:
                    p1 = np.zeros(len(starts)); p2 = np.zeros(len(starts))
                    for k in np.flatnonzero(passed):
                        i_f = int(starts[k] + c["tau"][k])
                        if i_f < n - 5:
                            p1[k], p2[k], _, _ = run_funded(xf, mf, i_f, pth.intraday_dll, v, cfg.combine_rules.trailing_mll, cfg.profit_split)
                    net = p1 + p2 - c["fees"]
                    def era(mask):
                        pm = passed & mask
                        return (((p1 > 0) & pm).sum() / max(pm.sum(), 1), net[mask].mean() / max(passed[mask].mean(), 1e-9))
                    pre, post = era(yrs < 2020), era(yrs >= 2020)
                    paid1 = (p1 > 0) & passed; cost = c["fees"].sum() / max(passed.sum(), 1)
                    ev = year_bootstrap(net, yrs, np.mean)
                    rows.append(dict(stream=sn, sigma_combine=sc, sigma_funded=sf, plan=v.name, p_pass=passed.mean(),
                        p_first=paid1.sum() / max(passed.sum(), 1), first_net=p1[paid1].mean() if paid1.any() else 0,
                        cost_funded=cost, covers=bool(paid1.any() and p1[paid1].mean() >= cost),
                        p_second=((p2 > 0) & passed).sum() / max(paid1.sum(), 1), ev_funded=ev[0] / max(passed.mean(), 1e-9),
                        ev_lo=ev[1], ev_hi=ev[2], p_first_pre2020=pre[0], ev_funded_pre2020=pre[1],
                        p_first_2020plus=post[0], ev_funded_2020plus=post[1]))
    df = pd.DataFrame(rows); Path("results").mkdir(exist_ok=True)
    df.to_csv(f"results/decoupled_{a.path}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    cols = ["stream","sigma_combine","sigma_funded","plan","p_pass","p_first","first_net","cost_funded","p_second","ev_funded","ev_lo","ev_hi","p_first_pre2020","p_first_2020plus"]
    print(f"{len(df)} configs. Top 12 by P(first payout) where first payout >= avg cost to get funded:")
    print(df[df.covers].sort_values("p_first", ascending=False).head(12)[cols].round(2).to_string(index=False))
    print("\nSame ranking, NO-EDGE controls only:")
    print(df[df.covers & df.stream.str.startswith("noedge")].sort_values("p_first", ascending=False).head(6)[cols].round(2).to_string(index=False))
    print("\nConfigs whose per-attempt EV 90% CI is fully above zero:", int((df.ev_lo > 0).sum()), "of", len(df))
if __name__ == "__main__":
    main()
