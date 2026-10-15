"""Real-data study under Topstep's verified rules: Combine -> funded -> payout plans.

Usage: python run_real_study.py [--path no_activation_dll] [--risk 400 800 1600 2400] [--compare-rules]
Writes results/real_study_<path>.csv and prints the ranking (first-payout probability where the first
payout covers the average cost of getting funded).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import build_sessions
from pricing_config import load_config
from real_engine import (FundedPlan, RuleSet, Stream, combine_outcomes, rules_from_config,
                         run_funded, year_bootstrap, _lists)

MIN_REMAINING = 400      # sessions that must remain after a start date
ERA_SPLIT = 2020


def build_streams() -> dict:
    """Entries only 09:30-15:55 ET (the VWAP is still anchored at the 18:00 ET open); always flat by 15:55 ET."""
    es, nq = build_sessions("es"), build_sessions("nq")
    mk = Stream.from_series
    return {
        "noedge_es": mk("noedge_es", st.run(es, "random", seed=0)),
        "noedge_nq": mk("noedge_nq", st.run(nq, "random", seed=0)),
        "nq_orb": mk("nq_orb", st.run(nq, "orb", buf=0.05, s=0.15)),
        "es_orb": mk("es_orb", st.run(es, "orb", buf=0.05, s=0.15)),
        "nq_vwap_atr": mk("nq_vwap_atr", st.run(nq, "vwap_atr", k=0.2, s=0.15, start_min=st.CASH_OPEN)),
    }


def build_wf_streams() -> dict:
    from run_sweep import GRIDS
    from wf import walk_forward_series
    nq = build_sessions("nq")
    mk = Stream.from_series
    vw, ch_v = walk_forward_series(nq, "vwap_atr", GRIDS["vwap_atr"])
    orb, ch_o = walk_forward_series(nq, "orb", GRIDS["orb"])
    fixed = st.run(nq, "vwap_atr", k=0.2, s=0.2, start_min=st.CASH_OPEN)
    ctl = st.run(nq, "random", seed=0)
    first = int(np.flatnonzero(pd.DatetimeIndex(nq.dates).year >= 2015)[0])
    cut = lambda d: st.DailySeries(d.dates[first:], d.pnl[first:], d.minpnl[first:], d.trades[first:], d.traded_ok[first:], d.atr[first:], d.mult)
    print("walk-forward choices (NQ VWAP+ATR):", {y: f"k={p['k']} s={p['s']}" for y, p in ch_v.items()})
    return {"nq_vwap_WF": mk("nq_vwap_WF", vw), "nq_orb_WF": mk("nq_orb_WF", orb),
            "nq_vwap_fixed(k.2 s.2)": mk("nq_vwap_fixed", cut(fixed)), "noedge_nq": mk("noedge_nq", cut(ctl))}


def plans() -> list:
    out = []
    for path in ("standard", "consistency"):
        for t1 in (2000, 3000):
            for r1 in (1.0, 0.5):
                out.append(FundedPlan(f"{path[:4]} t{t1} r1x{r1} close", path, t1, 0.5, r1))
                for r2, tag in ((0.5, "cons"), (1.0, "same"), (2.0, "aggr")):
                    out.append(FundedPlan(f"{path[:4]} t{t1} r1x{r1} +2nd {tag}", path, t1, 0.5, r1, t1, 0.5, r2))
    return out


def study(path_key: str, risks, rules: RuleSet, plan_list=None, streams=None) -> pd.DataFrame:
    cfg = load_config()
    streams = streams or build_streams()
    plan_list = plan_list or plans()
    rows = []
    for sname, s in streams.items():
        n = len(s.dates)
        starts = np.arange(int(np.flatnonzero(s.ok)[0]), n - MIN_REMAINING)
        yrs = pd.DatetimeIndex(s.dates[starts]).year.to_numpy()
        x, lo = _lists(s)
        for R in risks:
            nd = s.desired_micros(R)
            c = combine_outcomes(x, lo, nd, cfg, path_key, rules, starts)
            passed = c["status"] == 1
            idx = np.flatnonzero(passed)
            sigma = float(np.std(s.pnl[s.ok] * np.array(nd)[s.ok]))
            cost = c["fees"].sum() / max(passed.sum(), 1)
            for v in plan_list:
                p1 = np.zeros(len(starts)); p2 = np.zeros(len(starts)); fdays = np.zeros(len(starts))
                for k in idx:
                    i_f = int(starts[k] + c["tau"][k])
                    if i_f < n - 5:
                        p1[k], p2[k], fdays[k], _ = run_funded(x, lo, nd, i_f, rules, v)
                net = p1 + p2 - c["fees"]
                paid1 = (p1 > 0) & passed
                paid2 = (p2 > 0) & passed
                pre, post = yrs < ERA_SPLIT, yrs >= ERA_SPLIT
                ev = year_bootstrap(net, yrs, np.mean)
                rows.append(dict(
                    stream=sname, path=path_key, risk_unit=R, daily_sigma=sigma, plan=v.name,
                    p_pass=passed.mean(), p_first=paid1.sum() / max(passed.sum(), 1),
                    first_net=p1[paid1].mean() if paid1.any() else 0.0, cost_funded=cost,
                    covers=bool(paid1.any() and p1[paid1].mean() >= cost),
                    p_second=paid2.sum() / max(paid1.sum(), 1),
                    second_net=p2[paid2].mean() if paid2.any() else 0.0,
                    ev_funded=ev[0] / max(passed.mean(), 1e-9), ev_lo=ev[1], ev_hi=ev[2],
                    p_first_pre=((p1 > 0) & passed & pre).sum() / max((passed & pre).sum(), 1),
                    p_first_post=((p1 > 0) & passed & post).sum() / max((passed & post).sum(), 1),
                    n_funded=int(passed.sum()), attempt_days=float(c['tau'].mean()),
                    funded_days=float(fdays[passed].mean()) if passed.any() else 0.0))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="no_activation_dll")
    ap.add_argument("--risk", type=float, nargs="+", default=[400, 800, 1600, 2400])
    ap.add_argument("--compare-rules", action="store_true")
    ap.add_argument("--wf", action="store_true", help="use walk-forward NQ streams (2015+)")
    a = ap.parse_args()
    cfg = load_config()
    rules = rules_from_config(cfg, a.path)
    df = study(a.path, a.risk, rules, streams=(build_wf_streams() if a.wf else None))
    Path("results").mkdir(exist_ok=True)
    df.to_csv(f"results/{'wf_' if a.wf else ''}real_study_{a.path}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    cols = ["stream", "risk_unit", "daily_sigma", "plan", "p_pass", "p_first", "first_net", "cost_funded",
            "p_second", "ev_funded", "ev_lo", "ev_hi", "p_first_pre", "p_first_post"]
    print(f"{len(df)} configurations | path={a.path} | rules: MLL real-time, locks at $0, post-payout MLL=$0, consistency 55%")
    print("\nTop 12 by P(first payout | funded) where the first payout >= average cost of getting funded:")
    print(df[df.covers].sort_values("p_first", ascending=False).head(12)[cols].round(2).to_string(index=False))
    print("\nNo-edge controls, same ranking:")
    print(df[df.covers & df.stream.str.startswith("noedge")].sort_values("p_first", ascending=False).head(5)[cols].round(2).to_string(index=False))
    print(f"\nConfigs with per-attempt EV 90% CI fully above zero: {int((df.ev_lo > 0).sum())} of {len(df)}")
    if a.compare_rules:
        print("\n=== How much each rule correction matters (nq_orb, risk 1600, plan 'stan t2000 r1x1.0 +2nd cons') ===")
        one = [p for p in plans() if p.name == "stan t2000 r1x1.0 +2nd cons"]
        st_ = {"nq_orb": build_streams()["nq_orb"]}
        variants = {
            "official rules (default)": rules,
            "old: consistency 50%": RuleSet(**{**rules.__dict__, "consistency_ratio": 0.50}),
            "old: Combine MLL never locks": RuleSet(**{**rules.__dict__, "combine_mll_locks": False}),
            "old: breach only at the close": RuleSet(**{**rules.__dict__, "intraday_mll": False}),
        }
        out = []
        for name, rs in variants.items():
            d = study(a.path, [1600], rs, one, st_).iloc[0]
            out.append(dict(rules=name, p_pass=d.p_pass, p_first=d.p_first, p_second=d.p_second, cost_funded=d.cost_funded, ev_funded=d.ev_funded))
        print(pd.DataFrame(out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
