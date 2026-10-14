"""Dev-data search (starts 2010-2022 only): fixed vs cushion-proportional sizing, for the NQ strategy, the portfolio, and a coin-flip control.
Goals: (1) first payout covers the cost of getting funded, (2) high chance a funded account reaches payout, (3) positive EV.
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import build_sessions
from pricing_config import load_config
from real_engine import FundedPlan, Stream, rules_from_config
from contract_eval import CombineSizing, contract_metrics
from portfolio import DEV_END, basket, dev_weights, make_sleeve

PATH = "no_activation_dll"
cfg = load_config()
RULES = rules_from_config(cfg, PATH)


def sleeves():
    return [make_sleeve("nq", "noise_area", {}, name="nq_noise"), make_sleeve("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN), name="nq_vwap"),
            make_sleeve("es", "noise_area", {}, name="es_noise"), make_sleeve("es", "vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN), dict(trend=250), name="es_vwap_t250"),
            make_sleeve("gc", "noise_area", {}, dict(trend=60), name="gc_noise_t60")]


def streams():
    S = sleeves()
    w = dev_weights(S, shrink=0.5)
    out = {"portfolio": basket(S, weights=w, name="portfolio"), "nq_noise": basket([S[0]], name="nq_noise")}
    nq = build_sessions("nq")
    out["coin"] = Stream.from_series("coin", st.run(nq, "coin", seed=0, s=None))
    return out, w


def plan_grid(path="standard"):
    plans = []
    for t1 in (1500, 3000):
        for a1 in (0.05, 0.1, 0.2, 0.4):
            for a2 in (0.05, 0.1, 0.2, 0.4):
                plans.append((FundedPlan(f"{path[:4]} t1={t1} dyn a1={a1} a2={a2}", path, t1, 0.5, 1.0, 0.0, 0.5, 1.0, 99, a1, a2), None))
        for Rf in (1200, 3600):
            plans.append((FundedPlan(f"{path[:4]} t1={t1} FIXED Rf={Rf}", path, t1, 0.5, 1.0, 0.0, 0.5, 1.0, 99), Rf))
    return plans


def main() -> None:
    S, w = streams()
    print("portfolio weights (micros per sleeve, from dev data only):", {k: round(v, 2) for k, v in w.items()})
    sizings = [CombineSizing("fixed", 3200), CombineSizing("fixed", 6400), CombineSizing("dyn", 0.75), CombineSizing("dyn", 1.5)]
    rows = []
    for name, s in S.items():
        n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
        cut = Stream(s.name, s.dates[:n_dev], s.pnl[:n_dev], s.low[:n_dev], s.atr_usd[:n_dev], s.ok[:n_dev])
        first = int(np.flatnonzero(cut.ok)[0])
        starts = np.arange(first, n_dev - 400)
        yrs = pd.DatetimeIndex(cut.dates[starts]).year.to_numpy()
        for sz in sizings:
            for pl, Rf in plan_grid():
                m = contract_metrics(cut, cut, sz, pl, Rf, cfg, RULES, starts, yrs)
                rows.append(dict(stream=name, combine=sz.label(), plan=pl.name, **m))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/dynamic_dev_grid.csv", index=False)
    pickle.dump(w, open("/home/user/data_raw/portfolio_weights.pkl", "wb"))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 44)
    df["sizing_kind"] = np.where(df.plan.str.contains("dyn"), "dynamic", "fixed")
    print("\n=== DEV: average over all configs, dynamic vs fixed funded sizing ===")
    print(df.groupby(["stream", "sizing_kind"])[["p_first", "first_net", "cost_funded", "payouts_per_funded", "extraction_per_funded", "p_die_no_payout", "days_alive", "ev_funded"]].mean().round(2).to_string())
    print("\n=== DEV: best by P(first payout) among configs where the first payout covers the premium and EV>0 ===")
    cols = ["stream", "combine", "plan", "p_pass", "cost_funded", "p_first", "first_net", "payouts_per_funded", "extraction_per_funded", "ev_funded", "ev_lo", "ev_hi"]
    for name in S:
        d = df[(df.stream == name) & df.covers & (df.ev_funded > 0)].sort_values(["p_first", "ev_funded"], ascending=False)
        print(f"-- {name}: {len(d)} configs qualify"); print(d.head(4)[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
