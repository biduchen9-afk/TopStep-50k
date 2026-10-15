"""Hold-out for the dynamic-sizing contract (configs frozen from run_dynamic_contract.py dev grid), plus the 5-funded / 20-eval farm."""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd

import intraday_strategies as st
from pricing_config import load_config
from real_engine import FundedPlan, Stream, rules_from_config
from contract_eval import CombineSizing, contract_metrics
from chain_real import replay_detail
from farm_real import FarmSpec, run_farm
from portfolio import DEV_END, basket, make_sleeve
from run_dynamic_contract import sleeves, streams as dev_streams, plan_grid, PATH

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
cfg = load_config()
RULES = rules_from_config(cfg, PATH)


def frozen(stream_name):
    df = pd.read_csv("results/dynamic_dev_grid.csv")
    d = df[(df.stream == stream_name) & df.covers & (df.ev_funded > 0)].sort_values(["p_first", "ev_funded"], ascending=False).iloc[0]
    cb = d["combine"]
    sz = CombineSizing("fixed", float(cb[2:])) if cb.startswith("Rc") else CombineSizing("dyn", float(cb[4:]))
    plan, Rf = next((p, r) for p, r in plan_grid() if p.name == d["plan"])
    return sz, plan, Rf, d


def main() -> None:
    S, w = dev_streams()
    first_dev = {}
    out, rep_rows, farm_rows = [], [], []
    for name, s in S.items():
        sz, plan, Rf, d = frozen(name)
        n_dev = int(np.searchsorted(s.dates, DEV_END, side="right")); n = len(s.dates)
        yrs = (s.dates[-1] - s.dates[n_dev]).astype(float) / 365.25
        starts = np.arange(n_dev, n - 300)
        q = pd.DatetimeIndex(s.dates[starts]); grp = (q.year * 4 + (q.month - 1) // 3).to_numpy()
        m = contract_metrics(s, s, sz, plan, Rf, cfg, RULES, starts, grp)
        out.append(dict(stream=name, combine=sz.label(), plan=plan.name, dev_ev=d.ev_funded, **m))
        # sequential one-account replays
        offs = range(n_dev, n_dev + 60, 2)
        reps = pd.DataFrame([replay_detail(s, sz.value if sz.kind == "fixed" else 3000.0, s, Rf or 0.0, plan, cfg, RULES, PATH, o,
                                           dyn_c=sz.value if sz.kind == "dyn" else None) for o in offs])
        rep_rows.append(dict(stream=name, total=reps.net.mean(), per_year=reps.net.mean() / yrs, worst=reps.net.min(), best=reps.net.max(),
                             funded=reps.funded.mean(), payouts=reps.n_payouts.mean(), fees=reps.fees.mean(), payouts_total=reps.payouts.mean(),
                             pct_time_funded=100 * reps.days_funded.mean() / reps.sessions.mean()))
        # farm scaling
        for (me, mf) in ((1, 1), (4, 2), (8, 3), (12, 5), (20, 5)):
            res = [run_farm([s], sz, Rf, plan, cfg, RULES, PATH, o, FarmSpec(max_evals=me, max_funded=mf, p_est=1.0)) for o in range(n_dev, n_dev + 60, 5)]
            r = pd.DataFrame(res)
            farm_rows.append(dict(stream=name, max_evals=me, max_funded=mf, net=r.net.mean(), per_year=r.net.mean() / yrs, worst=r.net.min(), best=r.net.max(),
                                  evals=r.n_evals.mean(), funded=r.n_funded.mean(), wasted_passes=r.n_wasted.mean(), payouts=r.payouts.mean(), fees=r.fees.mean(),
                                  avg_funded_active=r.avg_funded_active.mean(), max_funded_seen=r.max_funded_seen.max()))
        print(name, "done", flush=True)
    res = pd.DataFrame(out); res.to_csv("results/dynamic_holdout_metrics.csv", index=False)
    print("=== HOLD-OUT 2023-01 -> 2026-07: per-start contract metrics (frozen from dev) ===")
    print(res[["stream", "combine", "plan", "p_pass", "tries", "e_tau", "cost_funded", "p_first", "first_net", "covers", "payouts_per_funded",
               "extraction_per_funded", "p_die_no_payout", "days_alive", "ev_funded", "ev_lo", "ev_hi", "dev_ev"]].round(2).to_string(index=False))
    rp = pd.DataFrame(rep_rows); rp.to_csv("results/dynamic_holdout_replay.csv", index=False)
    print("\n=== One account at a time, sequential rinse-and-repeat (30 start offsets) ==="); print(rp.round(0).to_string(index=False))
    fm = pd.DataFrame(farm_rows); fm.to_csv("results/dynamic_holdout_farm.csv", index=False)
    print("\n=== Farm scaling (12 start offsets; all accounts share one market path) ==="); print(fm.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
