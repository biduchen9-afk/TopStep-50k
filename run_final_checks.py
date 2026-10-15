"""Final checks on the hold-out (2023-01 -> 2026-07) with everything frozen from the pre-2023 selection.

1 slippage / commission stress    2 ES and GC                    3 breach policies (billing clock)
4 spec table, 3 fee paths         5 why is the profit low?       6 improvements chosen on dev data only
"""
from __future__ import annotations

import ast
import math
import sys

import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import build_sessions
from pricing_config import load_config
from real_engine import (FundedPlan, Stream, combine_outcomes, rules_from_config, run_funded, year_bootstrap,
                         costs_from_config, _lists)
from chain_real import POLICIES, policy_stats, replay_detail
from run_holdout import DEV_END, PATH, metrics
from run_real_study import plans as all_plans
from run_sweep import r_series, sharpe
from valuation_engine import calc_ev_per_attempt, calc_ev_per_cycle

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
cfg = load_config()
RULES = rules_from_config(cfg, PATH)
PLANS = {p.name: p for p in all_plans() if "r1x1.0" in p.name}


def build(market, family, params, costs=None):
    sess = build_sessions(market)
    d = st.run(sess, family, costs=costs or costs_from_config(cfg), **params)
    n_dev = int(np.searchsorted(sess.dates, DEV_END, side="right"))
    return Stream.from_series(f"{market}_{family}", d), d, n_dev


def frozen_configs():
    dg = pd.read_csv("results/holdout_dev_grid.csv")
    cov = dg[dg.covers & (dg.ev_funded > 0)]
    A = cov.sort_values(["p_first", "ev_funded"], ascending=False).iloc[0]
    B = dg.sort_values("ev_funded", ascending=False).iloc[0]
    return {"A": A, "B": B}




def hold_eval(stream, n_dev, row, rules=RULES, path=PATH, comb=None):
    n = len(stream.dates)
    starts = np.arange(n_dev, n - 300)
    dts = pd.DatetimeIndex(stream.dates[starts])
    grp = (dts.year * 4 + (dts.month - 1) // 3).to_numpy()
    return metrics(comb or stream, float(row.Rc), stream, float(row.Rf), PLANS[row.plan], cfg, rules, starts, grp)


def replays(stream, n_dev, row, comb=None, offsets=None, **kw):
    offs = offsets or range(n_dev, n_dev + 60, 2)
    res = [replay_detail(comb or stream, float(row.Rc), stream, float(row.Rf), PLANS[row.plan], cfg, RULES, PATH, o, **kw) for o in offs]
    return pd.DataFrame(res)


def main() -> None:
    cfgs = frozen_configs()
    print("Frozen configs (selected on data <= 2022):")
    for k, r in cfgs.items():
        print(f"  {k}: Combine R{int(r.Rc)}, funded R{int(r.Rf)}, plan '{r.plan}'")
    W = dict(market="nq", family="vwap_atr", params={"k": 0.3, "s": 0.2, "start_min": st.CASH_OPEN})

    # ------------------------------------------------------------------ 1
    print("\n=== 1) SLIPPAGE / COMMISSION STRESS (hold-out, strategy in both phases, frozen sizing/plan) ===")
    rows = []
    for slip, comm in ((0, 2.5), (1, 2.5), (2, 2.5), (3, 2.5), (1, 5.0), (2, 5.0)):
        s, d, n_dev = build(**W, costs=st.Costs(commission_rt=comm, slippage_ticks_rt=float(slip)))
        r = r_series(d); ok = d.traded_ok
        hold = ok & (d.dates > DEV_END)
        for k, row in cfgs.items():
            m = hold_eval(s, n_dev, row)
            rp = replays(s, n_dev, row)
            rows.append(dict(slip_ticks=slip, commission=comm, config=k, hold_sharpe=sharpe(r[hold]), p_pass=m["p_pass"],
                             cost_funded=m["cost_funded"], mu_ext=m["mu_ext"], p_first=m["p_first"], ev_funded=m["ev_funded"],
                             ev_lo=m["ev_lo"], ev_hi=m["ev_hi"], replay_total=rp.net.mean(), replay_min=rp.net.min()))
    t1 = pd.DataFrame(rows); t1.to_csv("results/final_slippage.csv", index=False)
    print(t1.round(2).to_string(index=False))

    # ------------------------------------------------------------------ 2
    print("\n=== 2) ES and GC: dev-selected best variant per market, hold-out, config A sizing ===")
    v = pd.read_csv("results/holdout_variants.csv")
    rows = []
    for mkt in ("es", "gc"):
        b = v[v.market == mkt].sort_values("dev_sharpe", ascending=False).iloc[0]
        params = ast.literal_eval(b.params)
        s, d, n_dev = build(mkt, b.family, params)
        m = hold_eval(s, n_dev, cfgs["A"]); rp = replays(s, n_dev, cfgs["A"])
        rows.append(dict(market=mkt, variant=f"{b.family} {params}", dev_sharpe=b.dev_sharpe, hold_sharpe=b.hold_sharpe, hold_t=b.hold_t,
                         p_pass=m["p_pass"], cost_funded=m["cost_funded"], p_first=m["p_first"], ev_funded=m["ev_funded"],
                         ev_lo=m["ev_lo"], ev_hi=m["ev_hi"], replay_total=rp.net.mean()))
    t2 = pd.DataFrame(rows); t2.to_csv("results/final_es_gc.csv", index=False)
    print(t2.round(2).to_string(index=False))

    stream, dser, n_dev = build(**W)
    n = len(stream.dates)
    starts = np.arange(n_dev, n - 300)

    # ------------------------------------------------------------------ 3
    print("\n=== 3) BREACH POLICIES on the hold-out (config A Combine sizing; fees and time to reach a funded account) ===")
    rows = [policy_stats(stream, float(cfgs["A"].Rc), cfg, PATH, RULES, starts, p) for p in POLICIES]
    t3 = pd.DataFrame(rows); t3.to_csv("results/final_policies.csv", index=False)
    print(t3.round(1).to_string(index=False))
    print("(reset fee == subscription fee on all three paths, so 'immediate_reset' falls back to cancel-and-reopen as the spec says;"
          " 'immediate_reset_keep' forces the reset to show what keeping the old clock costs)")

    # ------------------------------------------------------------------ 4
    print("\n=== 4) SPEC TABLE across the three fee paths (hold-out, config A; DLL only on no_activation_dll) ===")
    rows = []
    row = cfgs["A"]
    xs, lo = _lists(stream)
    for pk in ("standard", "no_activation", "no_activation_dll"):
        rules = rules_from_config(cfg, pk)
        c = combine_outcomes(xs, lo, stream.desired_micros(float(row.Rc)), cfg, pk, rules, starts)
        ps = c["status"] == 1; p = ps.mean(); act = cfg.paths[pk].activation_fee
        ndf = stream.desired_micros(float(row.Rf)); pay = np.zeros(len(starts))
        for k in np.flatnonzero(ps):
            i_f = int(starts[k] + c["tau"][k])
            if i_f < n - 5:
                a, b, _, _ = run_funded(xs, lo, ndf, i_f, rules, PLANS[row.plan]); pay[k] = a + b
        mu_att = float((c["fees"] - act * ps).mean()); mu_ext = float(pay[ps].mean())
        ev_att = calc_ev_per_attempt(p, mu_ext, mu_att, act)
        emp = float((pay - c["fees"]).mean())
        rows.append(dict(path=pk, p=p, E_tau=float(c["tau"].mean()), mu_attempt=mu_att, cost_funded=mu_att / p + act, mu_ext=mu_ext,
                         mu_ext_star=mu_att / p + act, EV_attempt=ev_att, EV_cycle=calc_ev_per_cycle(p, mu_ext, mu_att, act),
                         empirical_EV_attempt=emp))
    t4 = pd.DataFrame(rows); t4.to_csv("results/final_spec_table.csv", index=False)
    print(t4.round(2).to_string(index=False))
    assert np.allclose(t4.EV_attempt, t4.empirical_EV_attempt), "EV identity broken"
    print("(EV_attempt from the spec formula equals the empirical mean per attempt, as required)")

    # ------------------------------------------------------------------ 5
    print("\n=== 5) WHY IS THE NET PROFIT LOW? hold-out replay, averages over 30 start offsets (one account at a time) ===")
    yrs = (stream.dates[-1] - stream.dates[n_dev]).astype(float) / 365.25
    rows = []
    for k, r in cfgs.items():
        rp = replays(stream, n_dev, r)
        rows.append(dict(config=k, years=round(yrs, 1), net=rp.net.mean(), payouts=rp.payouts.mean(), fees=rp.fees.mean(),
                         fees_pct_of_payouts=100 * rp.fees.mean() / rp.payouts.mean(), attempts=rp.attempts.mean(), funded=rp.funded.mean(),
                         n_payouts=rp.n_payouts.mean(), avg_payout=rp.payouts.sum() / max(rp.n_payouts.sum(), 1),
                         pct_time_in_combine=100 * rp.days_combine.mean() / rp.sessions.mean(), pct_time_funded=100 * rp.days_funded.mean() / rp.sessions.mean(),
                         funded_closed_by_plan=rp.closed.mean(), funded_died=rp.died.mean(), balance_left_when_closed=rp.left_on_table.mean()))
    t5 = pd.DataFrame(rows); t5.to_csv("results/final_why_low.csv", index=False)
    print(t5.round(1).T.to_string())

    # ------------------------------------------------------------------ 6
    print("\n=== 6) IMPROVEMENTS: choose on DEV windows only (3-year sequential windows, 2010-2022), then run once on the hold-out ===")
    dev = Stream(stream.name, stream.dates[:n_dev], stream.pnl[:n_dev], stream.low[:n_dev], stream.atr_usd[:n_dev], stream.ok[:n_dev])
    first = int(np.flatnonzero(dev.ok)[0])
    wins = np.linspace(first, n_dev - 760, 30).astype(int)
    cand = []
    for Rc in (2400, 3200, 4800):
        for Rf in (1200, 2400, 3600):
            for path in ("standard", "consistency"):
                for t1 in (2000, 3000):
                    for r2 in (0.5, 1.0):
                        for mp in (2, 4, 99):
                            cand.append((Rc, Rf, FundedPlan(f"{path[:4]} t{t1} r2x{r2} max{mp}", path, t1, 0.5, 1.0, t1, 0.5, r2, mp)))
    rows = []
    for Rc, Rf, pl in cand:
        res = [replay_detail(dev, Rc, dev, Rf, pl, cfg, RULES, PATH, int(o), max_sessions=756)["net"] for o in wins]
        rows.append(dict(Rc=Rc, Rf=Rf, plan=pl.name, dev_net_3y=np.mean(res), dev_pct_pos=np.mean(np.array(res) > 0), _pl=pl))
    dv = pd.DataFrame(rows).sort_values("dev_net_3y", ascending=False)
    print("Top 8 on dev windows (mean net profit per 3 years, one account at a time):")
    print(dv.drop(columns="_pl").head(8).round(1).to_string(index=False))
    best = dv.iloc[0]
    PLANS[best.plan] = best["_pl"]
    # compare with A/B and with the same plan capped at 2 payouts, on the hold-out
    rb = pd.Series(dict(Rc=best.Rc, Rf=best.Rf, plan=best.plan))
    out = []
    for label, row in (("config A (2 payouts)", cfgs["A"]), ("config B (2 payouts)", cfgs["B"]), ("dev-chosen improvement", rb)):
        rp = replays(stream, n_dev, row)
        # 5 staggered accounts on the same market path (correlated; each chain has at most one funded account)
        five = []
        for o in range(n_dev, n_dev + 40, 8):
            five.append(sum(replay_detail(stream, float(row.Rc), stream, float(row.Rf), PLANS[row.plan], cfg, RULES, PATH, o + 40 * j)["net"] for j in range(5)))
        out.append(dict(setup=label, plan=row.plan, one_account_total=rp.net.mean(), per_year=rp.net.mean() / yrs, min=rp.net.min(), max=rp.net.max(),
                        payouts=rp.n_payouts.mean(), five_staggered_accounts_total=float(np.mean(five)), five_per_year=float(np.mean(five)) / yrs))
    t6 = pd.DataFrame(out); t6.to_csv("results/final_improvements.csv", index=False)
    print("\nHold-out (2023-2026):")
    print(t6.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
