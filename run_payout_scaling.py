"""Pre-registration 3 (docs/PREREGISTRATION_3.md): first-payout balance and scaling, robust 'green in all years' objective. DEV farms only."""
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
T1 = (1500., 2000., 3000., 4000., 5000., 6000., 8000.); A1 = (0.2, 0.4, 0.8); A2 = (0.4, 0.8, 1.6, 2.4)
YEARS = list(range(2011, 2021))


def setup():
    s = build_stream(); d = pd.DatetimeIndex(s.dates)
    n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
    starts = {y: int(np.searchsorted(s.dates, np.datetime64(f"{y}-01-01"))) for y in range(2011, 2026)}
    return s, d, n_dev, starts


def dev_net(s, pl, i0, n_end):
    return farm_plan(s, CombinePolicy(R=9600.), pl, i0, n_end)["net"]


def sequential(s, d, n_dev, rules, pl, comb, starts_idx):
    x, lo, atr, nd = (a[:n_dev] for a in (s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist(), s.desired_micros(9600.0)))
    n_f = n_p = n_2 = 0; first_sz, tot, days = [], [], []
    for i0, (st, tau) in zip(starts_idx, comb):
        if st != 1:
            continue
        n_f += 1; tm = []
        pays, *_ = run_funded_detail(x, lo, nd, i0 + tau, rules, pl, atr, tm)
        tot.append(sum(pays))
        if pays:
            n_p += 1; n_2 += len(pays) > 1; first_sz.append(pays[0]); days.append((d[min(tm[0], n_dev - 1)] - d[i0]).days)
    return dict(p_paid=n_p / max(n_f, 1), first_payout=float(np.mean(first_sz)) if first_sz else np.nan, paid_per_funded=float(np.mean(tot)), second_payout_share=n_2 / max(n_p, 1),
                median_days=float(np.median(days)) if days else np.nan)


def main() -> None:
    s, d, n_dev, st0 = setup(); rules = rules_from_config(load_config(), PATH)
    x, lo, nd = (a[:n_dev] for a in (s.pnl.tolist(), s.low.tolist(), s.desired_micros(9600.0)))
    first = int(np.flatnonzero(s.ok)[0]) + 260
    starts_idx = list(range(first, n_dev - 300, 3))
    comb = [run_combine(x, lo, nd, i0, rules) for i0 in starts_idx]
    rows = []
    for t1, a1, a2 in itertools.product(T1, A1, A2):
        pl = FundedPlan("p", "standard", t1, 0.5, 1.0, 0.0, 0.5, 1.0, 99, a1, a2)
        nets = np.array([dev_net(s, pl, st0[y], min(st0[y] + 756, n_dev)) for y in YEARS])
        row = dict(t1=t1, a1=a1, a2=a2, mean=nets.mean(), p25=np.percentile(nets, 25), worst=nets.min(), green=int((nets > 0).sum()), nets=[round(v) for v in nets])
        row.update(sequential(s, d, n_dev, rules, pl, comb, starts_idx))
        rows.append(row)
    g = pd.DataFrame(rows); g.to_pickle("results/payout_scaling_dev_grid.pkl")
    g.drop(columns="nets").to_csv("results/payout_scaling_dev_grid.csv", index=False)
    feas = g[(g["mean"] >= 10000) & (g.green >= 6)].sort_values(["p25", "mean"], ascending=False)
    print(f"{len(g)} configurations; {len(feas)} feasible (mean >= $10k and >= 6 of 10 start years green).")
    cols = ["t1", "a1", "a2", "mean", "p25", "worst", "green", "p_paid", "first_payout", "paid_per_funded", "second_payout_share", "median_days"]
    print("\nTop feasible by p25:"); print(feas.head(10)[cols].round(2).to_string(index=False))
    print("\nTop by p25 regardless of feasibility:"); print(g.sort_values("p25", ascending=False).head(6)[cols].round(2).to_string(index=False))
    print("\nBaseline (t1=3000, a1=0.4, a2=1.6):"); print(g[(g.t1 == 3000) & (g.a1 == 0.4) & (g.a2 == 1.6)][cols].round(2).to_string(index=False))
    print("\nEffect of waiting (first-payout balance), averaged over a1/a2:")
    print(g.groupby("t1")[["mean", "p25", "green", "p_paid", "first_payout", "paid_per_funded", "second_payout_share", "median_days"]].mean().round(2).to_string())
    for col in ("a1", "a2"):
        print(f"\nby {col}:"); print(g.groupby(col)[["mean", "p25", "green", "p_paid", "paid_per_funded"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
