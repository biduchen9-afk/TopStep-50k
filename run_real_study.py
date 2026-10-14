"""Real-data study: Combine -> funded -> payout variants, ranked by first-payout probability.

Usage: python run_real_study.py [--path no_activation_dll] [--sigmas 600 1000 1500 2200]
Writes results/real_study.csv and prints the ranking.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import build_sessions
from pricing_config import load_config
from real_engine import (FundedVariant, Stream, combine_outcomes, run_funded, scale,
                         year_bootstrap)

MIN_REMAINING = 400   # need a year and a half of data left after a start date


def build_streams() -> dict:
    es, nq = build_sessions("es"), build_sessions("nq")
    return {
        "noedge_es": Stream.from_series("noedge_es", st.run(es, "random", seed=0)),
        "noedge_nq": Stream.from_series("noedge_nq", st.run(nq, "random", seed=0)),
        "nq_orb": Stream.from_series("nq_orb", st.run(nq, "orb", buf=0.05, s=0.15)),
        "es_orb": Stream.from_series("es_orb", st.run(es, "orb", buf=0.05, s=0.15)),
        "nq_vwap_atr": Stream.from_series("nq_vwap_atr", st.run(nq, "vwap_atr", k=0.2, s=0.15)),
    }


def variants() -> list:
    out = []
    for t1 in (2000, 3000, 4000):
        out.append(FundedVariant(f"close@p1 t{t1}", t1=t1, w1=0.5))
    for t1 in (2000, 3000):
        for size2 in (0.5, 1.0, 2.0):
            for w2 in (0.5, 1.0):
                tag = "cons" if size2 < 1 else ("same" if size2 == 1 else "aggr")
                wtag = "half" if w2 == 0.5 else "ALL"
                out.append(FundedVariant(f"cushion t{t1} 2nd:{tag} take:{wtag}", t1=t1, w1=0.5,
                                         size2=size2, t2=t1, w2=w2))
    return out


def study(path_key: str, sigmas, years_split: bool = False):
    cfg = load_config()
    pth = cfg.paths[path_key]
    streams = build_streams()
    rows = []
    for sname, s in streams.items():
        n = len(s.dates)
        first_ok = int(np.flatnonzero(s.ok)[0])
        starts = np.arange(first_ok, n - MIN_REMAINING)
        yrs = pd.DatetimeIndex(s.dates[starts]).year.to_numpy()
        for sigma in sigmas:
            R = s.risk_unit_for_sigma(sigma)
            x, m = scale(s, R)
            c = combine_outcomes(x, m, cfg, path_key, starts)
            passed = c["status"] == 1
            idx_pass = np.flatnonzero(passed)
            for v in variants():
                p1 = np.zeros(len(starts)); p2 = np.zeros(len(starts)); state = np.zeros(len(starts))
                for k in idx_pass:
                    i_f = int(starts[k] + c["tau"][k])
                    if i_f >= n - 5:
                        continue
                    a, b, d, e = run_funded(x, m, i_f, pth.intraday_dll, v,
                                            cfg.combine_rules.trailing_mll, cfg.profit_split)
                    p1[k], p2[k], state[k] = a, b, e
                net = p1 + p2 - c["fees"]
                n_pass = passed.sum()
                paid1 = (p1 > 0) & passed
                paid2 = (p2 > 0) & passed
                cost_funded = c["fees"].sum() / max(n_pass, 1)
                ev_att = year_bootstrap(net, yrs, np.mean)
                rows.append(dict(
                    stream=sname, path=path_key, sigma_d=sigma, variant=v.name,
                    p_pass=passed.mean(),
                    p_first=paid1.sum() / max(n_pass, 1),
                    first_net=p1[paid1].mean() if paid1.any() else 0.0,
                    p_second_given_first=paid2.sum() / max(paid1.sum(), 1),
                    second_net=p2[paid2].mean() if paid2.any() else 0.0,
                    cost_funded=cost_funded,
                    first_covers_cost=bool(paid1.any() and p1[paid1].mean() >= cost_funded),
                    ev_attempt=ev_att[0], ev_attempt_lo=ev_att[1], ev_attempt_hi=ev_att[2],
                    ev_funded=ev_att[0] / max(passed.mean(), 1e-9),
                    n_starts=len(starts), n_funded=int(n_pass),
                ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="no_activation_dll")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[600, 1000, 1500, 2200])
    args = ap.parse_args()
    df = study(args.path, args.sigmas)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(f"results/real_study_{args.path}.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"{len(df)} configurations; path={args.path}")
    ok = df[df.first_covers_cost].sort_values("p_first", ascending=False)
    print("\nTop 15 by P(first payout | funded), only where the first payout covers the cost of getting funded:")
    print(ok.head(15)[["stream", "sigma_d", "variant", "p_pass", "p_first", "first_net", "cost_funded",
                       "p_second_given_first", "second_net", "ev_funded"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
