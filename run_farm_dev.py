"""Farm-level search on DEV windows (3-year windows within 2010-2022): post-payout aggressiveness (a2), pre-payout (a1),
and how eagerly new Combines are started (p_est, max_evals). Then the frozen farm on the 2023-2026 hold-out."""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from pricing_config import load_config
from real_engine import FundedPlan, Stream, rules_from_config
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from portfolio import DEV_END
from run_dynamic_contract import streams as dev_streams, PATH

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
cfg = load_config()
RULES = rules_from_config(cfg, PATH)
SZ = CombineSizing("fixed", 6400)


def plan(a1, a2, t1=3000):
    return FundedPlan(f"stan t1={t1} a1={a1} a2={a2}", "standard", t1, 0.5, 1.0, 0.0, 0.5, 1.0, 99, a1, a2)


def cut(s, n):
    return Stream(s.name, s.dates[:n], s.pnl[:n], s.low[:n], s.atr_usd[:n], s.ok[:n])


def main() -> None:
    S, w = dev_streams()
    cands = {"NQ noise-area": [S["nq_noise"]], "portfolio": [S["portfolio"]], "mixed (NQ, portfolio)": [S["nq_noise"], S["portfolio"]]}
    n_dev = int(np.searchsorted(S["nq_noise"].dates, DEV_END, side="right"))
    first = int(np.flatnonzero(S["nq_noise"].ok)[0])
    wins = np.linspace(first, n_dev - 770, 8).astype(int)
    rows = []
    for cname, streams in cands.items():
        dev_streams_cut = [cut(s, n_dev) for s in streams]
        for a1, a2 in itertools.product((0.2, 0.4), (0.1, 0.2, 0.4, 0.8, 1.6)):
            pl = plan(a1, a2)
            for me, pe in ((5, 0.5), (10, 0.5), (20, 0.5), (10, 1.0), (20, 1.0)):
                res = [run_farm(dev_streams_cut, SZ, None, pl, cfg, RULES, PATH, int(o), FarmSpec(max_evals=me, max_funded=5, p_est=pe), n_end=int(o) + 756) for o in wins]
                r = pd.DataFrame(res)
                rows.append(dict(streams=cname, a1=a1, a2=a2, max_evals=me, p_est=pe, dev_net_3y=r.net.mean(), dev_p10=np.percentile(r.net, 10), dev_min=r.net.min(),
                                 fees=r.fees.mean(), payouts=r.payouts.mean(), funded=r.n_funded.mean(), wasted=r.n_wasted.mean(), avg_funded=r.avg_funded_active.mean()))
        print(cname, "done", flush=True)
    dg = pd.DataFrame(rows); dg.to_csv("results/farm_dev_grid.csv", index=False)
    print("=== DEV farm grid: top 10 by mean 3-year net (max 5 funded) ===")
    print(dg.sort_values("dev_net_3y", ascending=False).head(10).round(1).to_string(index=False))
    print("\nBy streams (best dev config each):")
    best = dg.sort_values("dev_net_3y", ascending=False).groupby("streams").head(1); print(best.round(1).to_string(index=False))
    print("\nAverage dev 3y net by a2 (post-payout aggressiveness):"); print(dg.groupby("a2")[["dev_net_3y", "dev_p10"]].mean().round(0).to_string())
    print("\nAverage dev 3y net by (p_est, max_evals):"); print(dg.groupby(["p_est", "max_evals"])[["dev_net_3y", "wasted", "fees"]].mean().round(0).to_string())

    # ---- frozen hold-out
    print("\n=== HOLD-OUT 2023-01 -> 2026-07 for the dev-best farm config of each stream choice ===")
    out = []
    for _, b in best.iterrows():
        streams = cands[b.streams]
        n = len(streams[0].dates); n_h = int(np.searchsorted(streams[0].dates, DEV_END, side="right"))
        yrs = (streams[0].dates[-1] - streams[0].dates[n_h]).astype(float) / 365.25
        pl = plan(b.a1, b.a2)
        res = pd.DataFrame([run_farm(streams, SZ, None, pl, cfg, RULES, PATH, o, FarmSpec(max_evals=int(b.max_evals), max_funded=5, p_est=b.p_est)) for o in range(n_h, n_h + 60, 5)])
        out.append(dict(streams=b.streams, config=f"a1={b.a1} a2={b.a2} evals<={int(b.max_evals)} p_est={b.p_est}", net=res.net.mean(), per_year=res.net.mean() / yrs,
                        worst=res.net.min(), best=res.net.max(), evals=res.n_evals.mean(), funded=res.n_funded.mean(), wasted=res.n_wasted.mean(), payouts=res.payouts.mean(),
                        fees=res.fees.mean(), avg_funded_active=res.avg_funded_active.mean(), max_funded_seen=res.max_funded_seen.max(), max_evals_seen=res.max_evals_seen.max()))
    ho = pd.DataFrame(out); ho.to_csv("results/farm_holdout.csv", index=False)
    print(ho.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
