"""Rinse-and-repeat as specified by the user:
  Combine -> pass -> funded. First payout at a chosen trigger (t1, half the balance). EVERY later payout is taken
  the moment the account is eligible (t2=0 -> any balance that yields >= $125), at the maximum allowed (50% of
  balance, capped). Trade until the account dies, then open a new Combine and repeat.
Settings are chosen on 3-year sequential windows from 2010-2022 only, then run once on 2023-2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import build_sessions
from pricing_config import load_config
from real_engine import FundedPlan, Stream, costs_from_config, rules_from_config
from chain_real import replay_detail
from run_holdout import DEV_END, PATH
from run_final_checks import build, cfg, RULES

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)


def mk_plan(path, t1, r2, anytime=True):
    t2 = 0.0 if anytime else t1
    return FundedPlan(f"{path[:4]} t1={t1} later:{'anytime' if anytime else 'at t1'} r2x{r2}", path, t1, 0.5, 1.0, t2, 0.5, r2, 99)


def summarize(reps: pd.DataFrame, years: float) -> dict:
    return dict(total=reps.net.mean(), per_year=reps.net.mean() / years, worst=reps.net.min(), best=reps.net.max(),
                pct_positive=float((reps.net > 0).mean()), attempts=reps.attempts.mean(), funded=reps.funded.mean(),
                payouts=reps.n_payouts.mean(), avg_payout=reps.payouts.sum() / max(reps.n_payouts.sum(), 1),
                payouts_per_funded=reps.n_payouts.sum() / max(reps.funded.sum(), 1),
                fees=reps.fees.mean(), fees_pct=100 * reps.fees.mean() / max(reps.payouts.mean(), 1),
                days_alive=reps.days_funded.sum() / max(reps.funded.sum(), 1), pct_funded_time=100 * reps.days_funded.mean() / reps.sessions.mean(),
                died=reps.died.mean(), closed_by_plan=reps.closed.mean())


def main() -> None:
    W = dict(market="nq", family="vwap_atr", params={"k": 0.3, "s": 0.2, "start_min": st.CASH_OPEN})
    stream, d, n_dev = build(**W)
    n = len(stream.dates)
    yrs = (stream.dates[-1] - stream.dates[n_dev]).astype(float) / 365.25
    dev = Stream(stream.name, stream.dates[:n_dev], stream.pnl[:n_dev], stream.low[:n_dev], stream.atr_usd[:n_dev], stream.ok[:n_dev])
    first = int(np.flatnonzero(dev.ok)[0])
    wins = np.linspace(first, n_dev - 760, 30).astype(int)

    cand = []
    for anytime in (True, False):
        for Rc in (2400, 3200, 4800, 6400):
            for Rf in (1200, 2400, 3600):
                for path in ("standard", "consistency"):
                    for t1 in (1000, 2000, 3000):
                        for r2 in (0.5, 1.0, 2.0):
                            cand.append((Rc, Rf, mk_plan(path, t1, r2, anytime)))
    rows = []
    for Rc, Rf, pl in cand:
        nets = np.array([replay_detail(dev, Rc, dev, Rf, pl, cfg, RULES, PATH, int(o), max_sessions=756)["net"] for o in wins])
        rows.append(dict(Rc=Rc, Rf=Rf, plan=pl.name, dev_net_3y=nets.mean(), dev_p10=np.percentile(nets, 10), dev_pct_pos=(nets > 0).mean(), _pl=pl))
    dv = pd.DataFrame(rows).sort_values("dev_net_3y", ascending=False)
    pd.DataFrame(rows).drop(columns="_pl").to_csv("results/rinse_repeat_dev_grid.csv", index=False)
    print(f"=== DEV (2010-2022, 30 sequential 3-year windows): {len(dv)} configs ===")
    print("Top 8 overall:"); print(dv.drop(columns="_pl").head(8).round(1).to_string(index=False))
    for lab, anyt in (("later payouts whenever eligible (your rule)", "anytime"), ("later payouts only at balance >= t1", "at t1")):
        sub = dv[dv.plan.str.contains(f"later:{anyt}")]
        print(f"\nBest of '{lab}': mean {sub.dev_net_3y.iloc[0]:.0f} per 3y | median across all such configs {sub.dev_net_3y.median():.0f}")
    best_any = dv[dv.plan.str.contains("anytime")].iloc[0]
    best_t1 = dv[dv.plan.str.contains("at t1")].iloc[0]

    # ------------- hold-out, run once per frozen config -------------
    print("\n=== HOLD-OUT 2023-01 -> 2026-07 (%.1f years), 30 start offsets, one account at a time ===" % yrs)
    from real_engine import xfa_micro_cap  # noqa
    out = []
    offs = range(n_dev, n_dev + 60, 2)
    coin = Stream.from_series("coin", st.run(build_sessions("nq"), "coin", seed=0, s=None))
    coins = [Stream.from_series("coin", st.run(build_sessions("nq"), "coin", seed=sd, s=None)) for sd in range(5)]
    for label, b in (("YOUR RULE (dev-chosen)", best_any), ("old rule: later payouts at t1 (dev-chosen)", best_t1)):
        pl, Rc, Rf = b["_pl"], float(b.Rc), float(b.Rf)
        reps = pd.DataFrame([replay_detail(stream, Rc, stream, Rf, pl, cfg, RULES, PATH, o) for o in offs])
        out.append(dict(setup=label, config=f"Rc{int(Rc)} Rf{int(Rf)} {pl.name}", **summarize(reps, yrs)))
        if "YOUR" in label:
            hy = pd.concat([pd.DataFrame([replay_detail(c, Rc, stream, Rf, pl, cfg, RULES, PATH, o) for o in offs]) for c in coins])
            out.append(dict(setup="  same, coin-flip Combine -> strategy funded (5 seeds)", config="", **summarize(hy, yrs)))
            ne = pd.concat([pd.DataFrame([replay_detail(c, Rc, c, Rf, pl, cfg, RULES, PATH, o) for o in offs]) for c in coins])
            out.append(dict(setup="  same, NO EDGE in both phases (5 seeds)", config="", **summarize(ne, yrs)))
            for slip in (2, 3):
                s2, _, _ = build(**W, costs=st.Costs(slippage_ticks_rt=float(slip)))
                r2_ = pd.DataFrame([replay_detail(s2, Rc, s2, Rf, pl, cfg, RULES, PATH, o) for o in offs])
                out.append(dict(setup=f"  same, {slip}-tick slippage", config="", **summarize(r2_, yrs)))
    res = pd.DataFrame(out)
    res.to_csv("results/rinse_repeat_holdout.csv", index=False)
    show = ["setup", "total", "per_year", "worst", "best", "pct_positive", "attempts", "funded", "payouts", "avg_payout", "payouts_per_funded",
            "fees", "fees_pct", "days_alive", "pct_funded_time", "died", "closed_by_plan"]
    print(res[show].round(1).to_string(index=False))
    print("\nchosen (your rule):", res.config.iloc[0])


if __name__ == "__main__":
    main()
