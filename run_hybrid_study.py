"""Cost to reach a funded account, and a hybrid: aggressive NO-EDGE Combine -> real strategy once funded.

Part 1: Combine economics by strategy and sizing (pass rate, tries, days, fees, cost per funded account).
Part 2: hybrid (Combine = coin-flip lottery, funded = walk-forward NQ VWAP+ATR) vs the strategy in both phases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import build_sessions
from pricing_config import load_config
from real_engine import (FundedPlan, Stream, combine_outcomes, rules_from_config, run_funded,
                         year_bootstrap, _lists)
from run_real_study import MIN_REMAINING, build_wf_streams

PATH = "no_activation_dll"
FIRST = 2015


def coin_stream(nq, s, seed=0) -> Stream:
    d = st.run(nq, "coin", seed=seed, s=s)
    first = int(np.flatnonzero(pd.DatetimeIndex(nq.dates).year >= FIRST)[0])
    d = st.DailySeries(d.dates[first:], d.pnl[first:], d.minpnl[first:], d.trades[first:], d.traded_ok[first:], d.atr[first:], d.mult)
    return Stream.from_series(f"coin stop={s}", d)


def combine_table(streams: dict, risks, cfg, rules) -> pd.DataFrame:
    rows = []
    for name, s in streams.items():
        n = len(s.dates)
        starts = np.arange(int(np.flatnonzero(s.ok)[0]), n - MIN_REMAINING)
        yrs = pd.DatetimeIndex(s.dates[starts]).year.to_numpy()
        x, lo = _lists(s)
        for R in risks:
            nd = s.desired_micros(R)
            c = combine_outcomes(x, lo, nd, cfg, PATH, rules, starts)
            ps = c["status"] == 1
            p = ps.mean()
            fee_att = c["fees"].mean()
            rows.append(dict(stream=name, risk=R, p_pass=p, tries_per_funded=1 / p if p else np.inf,
                             p_funded_in_4_tries=1 - (1 - p) ** 4, avg_days=c["tau"].mean(),
                             bills_per_attempt=1 + ((np.maximum(c["tau"], 1) - 1) // cfg.billing_cycle_days).mean(),
                             fees_per_attempt=fee_att, cost_per_funded=fee_att / p if p else np.inf,
                             p_pre2020=ps[yrs < 2020].mean(), p_2020plus=ps[yrs >= 2020].mean(),
                             breach_rate=(c["status"] == 0).mean()))
    return pd.DataFrame(rows)


def hybrid(comb: Stream, R_c, fund: Stream, R_f, plan: FundedPlan, cfg, rules):
    """Combine traded with `comb` at R_c, funded account traded with `fund` at R_f (date-aligned)."""
    n = len(comb.dates)
    starts = np.arange(int(np.flatnonzero(comb.ok & fund.ok)[0]), n - MIN_REMAINING)
    yrs = pd.DatetimeIndex(comb.dates[starts]).year.to_numpy()
    xc, lc = _lists(comb); xf, lf = _lists(fund)
    c = combine_outcomes(xc, lc, comb.desired_micros(R_c), cfg, PATH, rules, starts)
    ndf = fund.desired_micros(R_f)
    ps = c["status"] == 1
    p1 = np.zeros(len(starts)); p2 = np.zeros(len(starts))
    for k in np.flatnonzero(ps):
        i_f = int(starts[k] + c["tau"][k])
        if i_f < n - 5:
            p1[k], p2[k], _, _ = run_funded(xf, lf, ndf, i_f, rules, plan)
    net = p1 + p2 - c["fees"]
    paid1 = (p1 > 0) & ps
    ev = year_bootstrap(net, yrs, np.mean)
    cost = c["fees"].sum() / max(ps.sum(), 1)
    return dict(p_pass=ps.mean(), cost_funded=cost, p_first=paid1.sum() / max(ps.sum(), 1),
                first_net=p1[paid1].mean() if paid1.any() else 0.0, p_second=((p2 > 0) & ps).sum() / max(paid1.sum(), 1),
                ev_funded=ev[0] / max(ps.mean(), 1e-9), ev_lo=ev[1], ev_hi=ev[2],
                p_first_pre=((p1 > 0) & ps & (yrs < 2020)).sum() / max((ps & (yrs < 2020)).sum(), 1),
                p_first_post=((p1 > 0) & ps & (yrs >= 2020)).sum() / max((ps & (yrs >= 2020)).sum(), 1))


def main() -> None:
    cfg = load_config(); rules = rules_from_config(cfg, PATH)
    nq = build_sessions("nq")
    wf = build_wf_streams()
    coins = {f"coin stop={s}": coin_stream(nq, s) for s in (None, 0.5, 0.25)}
    pd.set_option("display.width", 250, "display.max_columns", 30)
    risks = [1600, 2400, 3200, 4800, 6400, 9600]
    print("=== PART 1: cost to reach funded (no_activation_dll: $199/month, no activation fee), 2015+ start dates ===")
    streams = {"NQ VWAP+ATR (walk-fwd)": wf["nq_vwap_WF"], "NQ VWAP+ATR fixed k.2 s.2": wf["nq_vwap_fixed(k.2 s.2)"], **coins}
    t1 = combine_table(streams, risks, cfg, rules)
    t1.to_csv("results/combine_cost_table.csv", index=False)
    print(t1.round(2).to_string(index=False))
    print("\nCheapest way to a funded account per Combine strategy:")
    print(t1.sort_values("cost_per_funded").groupby("stream").head(1)[["stream", "risk", "p_pass", "tries_per_funded", "p_funded_in_4_tries", "avg_days", "cost_per_funded", "p_pre2020", "p_2020plus"]].round(2).to_string(index=False))

    print("\n=== PART 2: hybrid. Combine = lottery, funded = NQ VWAP+ATR (walk-forward), plan 'cons t3000 r1x0.5 +2nd cons', funded R=2400 ===")
    plan = FundedPlan("cons t3000 r1x0.5 +2nd cons", "consistency", 3000, 0.5, 0.5, 3000, 0.5, 0.5)
    rows = []
    best_coin = t1[t1.stream.str.startswith("coin")].sort_values("cost_per_funded")
    picks = [("strategy in both phases (R2400)", wf["nq_vwap_WF"], 2400)]
    for _, r in best_coin.head(3).iterrows():
        picks.append((f"{r.stream} R{int(r.risk)}", coins[r.stream], r.risk))
    for label, comb, R in picks:
        out = hybrid(comb, R, wf["nq_vwap_WF"], 2400, plan, cfg, rules)
        rows.append(dict(combine_with=label, **out))
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
