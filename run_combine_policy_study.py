"""Stage 1 (DEV only, 2010-2022): can state-dependent Combine sizing (early-phase multiplier, de-risk near the target, cushion cap) raise the farm's
net without new signals? Final NQ basket, hyst refill, funded t1=3000, a1=0.4, a2 in {0.4, 1.6}. 108 policy x a2 combinations (the neutral
policy R=9600 is one of them).  Objective as before: 0.5*(mean + p10) of net over 12 three-year dev windows.  Hold-out is reported afterwards for the
dev winner and the neutral policy only; nothing is chosen on it."""
from __future__ import annotations
import itertools, sys
import numpy as np, pandas as pd
from combine_policy import CombinePolicy, run_combine_policy
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from final_strategy import FINAL, PATH, build_stream
from portfolio import DEV_END
from pricing_config import load_config
from real_engine import FundedPlan, rules_from_config
from run_regime_study import windows

cfg = load_config(); rules = rules_from_config(cfg, PATH)
pd.set_option("display.width", 250, "display.max_columns", 40)


def plan(a2):
    return FundedPlan("p", "standard", 3000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.4, a2)


def farm(s, pol, a2, i0, n_end=None):
    spec = FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill="hyst", combine_policy=pol)
    return run_farm([s], CombineSizing("fixed", pol.R), None, plan(a2), cfg, rules, PATH, i0, spec, n_end=n_end)


def main() -> None:
    s = build_stream()
    wins = windows(s)
    n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
    x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist()
    first = int(np.flatnonzero(s.ok)[0]) + 260
    starts = range(first, n_dev - 300, 5)
    rows = []
    for R, s0, lk, cc, a2 in itertools.product((6400.0, 9600.0, 12800.0), (1.0, 1.5), (None, (0.7, 0.5), (0.8, 0.25)), (None, 2.0, 3.0), (0.4, 1.6)):
        pol = CombinePolicy(R=R, s0=s0, lock_at=None if lk is None else lk[0], g=1.0 if lk is None else lk[1], cap_c=cc)
        res = [farm(s, pol, a2, int(o), int(o) + 756) for o in wins]
        net = np.array([r["net"] for r in res])
        st = [run_combine_policy(x, lo, atr, i, rules, pol) for i in starts]
        pr = float(np.mean([a == 1 for a, _ in st])); br = float(np.mean([a == 0 for a, _ in st])); taup = float(np.mean([t for a, t in st if a == 1])) if any(a == 1 for a, _ in st) else np.nan
        rows.append(dict(R=R, s0=s0, lock=str(lk), cap_c=cc, a2=a2, dev_mean=net.mean(), dev_p10=np.percentile(net, 10), obj=0.5 * (net.mean() + np.percentile(net, 10)),
                         pos=float((net > 0).mean()), pass_rate=pr, breach=br, unresolved=1 - pr - br, days_to_pass=taup, fees=np.mean([r["fees"] for r in res])))
    g = pd.DataFrame(rows); g.to_csv("results/combine_policy_dev_grid.csv", index=False)
    print(f"{len(g)} policy x a2 combinations on 12 DEV windows.  Top 10 by objective:")
    print(g.sort_values("obj", ascending=False).head(10).round(2).to_string(index=False))
    neutral = g[(g.R == 9600) & (g.s0 == 1.0) & (g.lock == "None") & (g.cap_c.isna())]
    print("\nneutral (R=9600):"); print(neutral.round(2).to_string(index=False))
    for col in ("R", "s0", "lock", "cap_c", "a2"):
        print(f"\nmean by {col}:"); print(g.fillna({"cap_c": -1}).groupby(col)[["dev_mean", "dev_p10", "obj", "pass_rate", "breach", "unresolved", "days_to_pass", "fees"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
