"""True hold-out. EVERYTHING is selected on data through 2022-12-31, then frozen and run once on 2023-01 -> 2026-07.

Selection (dev only): strategy variant (3 markets x 3 families x grids), Combine sizing, funded sizing and payout plan.
Evaluation (hold-out): per-start outcomes, spec quantities (mu_ext vs mu_ext_star), and a sequential
"rinse and repeat" replay (Combine -> funded -> payout -> close -> new Combine) to the end of the data.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import build_sessions
from pricing_config import load_config
from real_engine import (FundedPlan, Stream, combine_outcomes, rules_from_config, run_combine, run_funded,
                         year_bootstrap, _lists)
from run_real_study import plans as all_plans
from run_sweep import GRIDS, r_series, sharpe

PATH = "no_activation_dll"
DEV_END = np.datetime64("2022-12-31")
RISKS_C = (2400, 3200, 4800, 6400)
RISKS_F = (1200, 2400)


def select_variant():
    best = {}
    allv = []
    for m in ("es", "nq", "gc"):
        sess = build_sessions(m)
        dev = sess.dates <= DEV_END
        for fam, grid in GRIDS.items():
            for p in grid:
                d = st.run(sess, fam, **p)
                r = r_series(d)
                dv, hd = d.traded_ok & dev, d.traded_ok & ~dev
                allv.append(dict(market=m, family=fam, params=p, dev_sharpe=sharpe(r[dv]), hold_sharpe=sharpe(r[hd]),
                                 hold_t=sharpe(r[hd]) * math.sqrt(hd.sum() / 252)))
    df = pd.DataFrame(allv)
    return df


def stream_of(market, family, params, name):
    sess = build_sessions(market)
    return Stream.from_series(name, st.run(sess, family, **params)), sess


def metrics(comb: Stream, Rc, fund: Stream, Rf, plan, cfg, rules, starts, grp):
    xc, lc = _lists(comb); xf, lf = _lists(fund)
    n = len(xc)
    c = combine_outcomes(xc, lc, comb.desired_micros(Rc), cfg, PATH, rules, starts)
    ps = c["status"] == 1
    ndf = fund.desired_micros(Rf)
    p1 = np.zeros(len(starts)); p2 = np.zeros(len(starts))
    for k in np.flatnonzero(ps):
        i_f = int(starts[k] + c["tau"][k])
        if i_f < n - 5:
            p1[k], p2[k], _, _ = run_funded(xf, lf, ndf, i_f, rules, plan)
    net = p1 + p2 - c["fees"]
    paid1 = (p1 > 0) & ps
    ev = year_bootstrap(net, grp, np.mean, n_boot=400)
    cost = c["fees"].sum() / max(ps.sum(), 1)
    return dict(p_pass=ps.mean(), tries=1 / ps.mean() if ps.mean() else np.inf, cost_funded=cost,
                mu_ext=(p1 + p2)[ps].mean() if ps.any() else 0.0, mu_ext_star=cost,
                p_first=paid1.sum() / max(ps.sum(), 1), first_net=p1[paid1].mean() if paid1.any() else 0.0,
                covers=bool(paid1.any() and p1[paid1].mean() >= cost),
                p_second=((p2 > 0) & ps).sum() / max(paid1.sum(), 1),
                ev_funded=ev[0] / max(ps.mean(), 1e-9), ev_att=ev[0], ev_lo=ev[1], ev_hi=ev[2], n_funded=int(ps.sum()), n_starts=len(starts))


def replay(comb, Rc, fund, Rf, plan, cfg, rules, i0, policy="cancel_and_reopen"):
    """Sequential rinse-and-repeat from session i0 to the end of the data."""
    xc, lc = _lists(comb); xf, lf = _lists(fund)
    ndc, ndf = comb.desired_micros(Rc), fund.desired_micros(Rf)
    n = len(xc); sub, act = cfg.paths[PATH].monthly_cost, cfg.paths[PATH].activation_fee
    t, fees, pay, att, nf, npay = i0, 0.0, 0.0, 0, 0, 0
    while t < n - 5:
        status, tau = run_combine(xc, lc, ndc, t, rules)
        fees += (1 + (max(tau, 1) - 1) // cfg.billing_cycle_days) * sub
        att += 1
        if status == 1:
            fees += act; nf += 1
            i_f = t + tau
            if i_f >= n - 5:
                break
            p1, p2, days, state = run_funded(xf, lf, ndf, i_f, rules, plan)
            pay += p1 + p2; npay += (p1 > 0) + (p2 > 0)
            t = i_f + days
            if state == 0:
                break
        else:
            t += tau
    return pay - fees, fees, pay, att, nf, npay


def main() -> None:
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 50)
    cfg = load_config(); rules = rules_from_config(cfg, PATH)
    print("=== STEP 1 (dev data <= 2022): which strategy would we have picked? ===")
    v = select_variant()
    top = v.sort_values("dev_sharpe", ascending=False)
    print(top.head(8).round(2).to_string(index=False))
    win = top.iloc[0]
    print(f"\nPicked on dev: {win.market} {win.family} {win.params}  (dev Sharpe {win.dev_sharpe:.2f}) -> hold-out Sharpe {win.hold_sharpe:.2f} (t={win.hold_t:.1f})")
    print("\nHold-out Sharpe of the dev-top-10 variants (ATR-normalised):", top.head(10).hold_sharpe.round(2).tolist())
    print("Median hold-out Sharpe of ALL 324 variants:", round(v.hold_sharpe.median(), 2), "| NQ variants:", round(v[v.market == 'nq'].hold_sharpe.median(), 2))
    v.to_csv("results/holdout_variants.csv", index=False)

    strat, sess = stream_of(win.market, win.family, win.params, "dev_pick")
    n = len(strat.dates); n_dev = int(np.searchsorted(strat.dates, DEV_END, side="right"))
    first = int(np.flatnonzero(strat.ok)[0])
    # ---------------- dev grid ----------------
    plans_ = [p for p in all_plans() if "r1x1.0" in p.name]
    dev_starts = np.arange(first, n_dev - 400)
    dev_yrs = pd.DatetimeIndex(strat.dates[dev_starts]).year.to_numpy()
    cut = lambda s: Stream(s.name, s.dates[:n_dev], s.pnl[:n_dev], s.low[:n_dev], s.atr_usd[:n_dev], s.ok[:n_dev])
    dev = cut(strat)
    rows = []
    for Rc in RISKS_C:
        for Rf in RISKS_F:
            for pl in plans_:
                m = metrics(dev, Rc, dev, Rf, pl, cfg, rules, dev_starts, dev_yrs)
                rows.append(dict(Rc=Rc, Rf=Rf, plan=pl.name, **m))
    dg = pd.DataFrame(rows)
    dg.to_csv("results/holdout_dev_grid.csv", index=False)
    cov = dg[dg.covers & (dg.ev_funded > 0)]
    A = (cov if len(cov) else dg).sort_values(["p_first", "ev_funded"], ascending=False).iloc[0]
    B = dg.sort_values("ev_funded", ascending=False).iloc[0]
    cols = ["Rc", "Rf", "plan", "p_pass", "cost_funded", "p_first", "first_net", "p_second", "ev_funded", "ev_lo", "ev_hi"]
    print("\n=== STEP 2 (dev data only): sizing + funded plan selection; %d configs, %d cover cost with EV>0 ===" % (len(dg), len(cov)))
    print("Criterion A (yours: max P(first payout) where payout >= cost to get funded and EV>0):"); print(A[cols].to_string())
    print("Criterion B (max expected profit per funded account):"); print(B[cols].to_string())

    # ---------------- hold-out ----------------
    hold_starts = np.arange(n_dev, n - 300)
    q = pd.DatetimeIndex(strat.dates[hold_starts]); grp = (q.year * 4 + (q.month - 1) // 3).to_numpy()
    plan_by = {p.name: p for p in plans_}
    out = []
    print("\n=== STEP 3: FROZEN configs, run ONCE on 2023-01 -> 2026-07 (%d start dates) ===" % len(hold_starts))
    controls = [st.run(sess, "coin", seed=sd, s=None) for sd in range(5)]
    coin_streams = []
    for d in controls:
        coin_streams.append(Stream.from_series("coin", d))
    for label, cfgrow in (("A: P(first) rule", A), ("B: max EV rule", B)):
        Rc, Rf, pl = float(cfgrow.Rc), float(cfgrow.Rf), plan_by[cfgrow.plan]
        h = metrics(strat, Rc, strat, Rf, pl, cfg, rules, hold_starts, grp)
        out.append(dict(setup=f"{label} | strategy both phases", **h))
        hy = [metrics(cs, Rc, strat, Rf, pl, cfg, rules, hold_starts, grp) for cs in coin_streams]
        out.append(dict(setup=f"{label} | coin-flip Combine -> strategy funded (mean of 5 seeds)", **{k: float(np.mean([r[k] for r in hy])) if isinstance(hy[0][k], (int, float, np.floating)) and not isinstance(hy[0][k], bool) else hy[0][k] for k in hy[0]}))
        nn = [metrics(cs, Rc, cs, Rf, pl, cfg, rules, hold_starts, grp) for cs in coin_streams]
        out.append(dict(setup=f"{label} | NO EDGE both phases (mean of 5 seeds)", **{k: float(np.mean([r[k] for r in nn])) if isinstance(nn[0][k], (int, float, np.floating)) and not isinstance(nn[0][k], bool) else nn[0][k] for k in nn[0]}))
    res = pd.DataFrame(out)
    res.to_csv("results/holdout_results.csv", index=False)
    show = ["setup", "p_pass", "tries", "cost_funded", "mu_ext", "mu_ext_star", "p_first", "first_net", "p_second", "ev_funded", "ev_lo", "ev_hi", "n_funded"]
    print(res[show].round(2).to_string(index=False))

    print("\n=== STEP 4: sequential 'rinse and repeat' replay from 2023 to the end of the data (one account at a time) ===")
    years = (strat.dates[-1] - strat.dates[n_dev]).astype(float) / 365.25
    rep = []
    for label, cfgrow in (("A", A), ("B", B)):
        Rc, Rf, pl = float(cfgrow.Rc), float(cfgrow.Rf), plan_by[cfgrow.plan]
        offs = range(n_dev, n_dev + 60, 2)
        for kind, comb, fund in (("strategy both", strat, strat), ("coin Combine -> strategy", coin_streams[0], strat), ("no edge both", coin_streams[0], coin_streams[0])):
            r = [replay(comb, Rc, fund, Rf, pl, cfg, rules, o) for o in offs]
            tot = np.array([x[0] for x in r])
            rep.append(dict(config=label, setup=kind, years=round(years, 1), mean_total=tot.mean(), min=tot.min(), max=tot.max(),
                            frac_positive=(tot > 0).mean(), attempts=np.mean([x[3] for x in r]), funded=np.mean([x[4] for x in r]),
                            payouts=np.mean([x[5] for x in r]), fees=np.mean([x[1] for x in r]), per_year=tot.mean() / years))
    print(pd.DataFrame(rep).round(1).to_string(index=False))
    pd.DataFrame(rep).to_csv("results/holdout_replay.csv", index=False)


if __name__ == "__main__":
    main()
