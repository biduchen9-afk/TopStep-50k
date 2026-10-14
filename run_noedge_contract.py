"""Is the 'contract' positive with NO edge when later payouts are taken whenever eligible and the account is
traded until it dies? 20 independent coin-flip sequences per setting, dev windows and hold-out, vs the strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import build_sessions
from real_engine import FundedPlan, Stream
from chain_real import replay_detail
from run_holdout import DEV_END, PATH
from run_final_checks import build, cfg, RULES

pd.set_option("display.width", 250, "display.max_columns", 40)
SEEDS = range(20)


def plan(path, t1):
    return FundedPlan(f"{path[:4]} t1={t1}", path, t1, 0.5, 1.0, 0.0, 0.5, 1.0, 99)


def totals(comb, fund, Rc, Rf, pl, offs, max_sessions=None):
    return np.mean([replay_detail(comb, Rc, fund, Rf, pl, cfg, RULES, PATH, o, max_sessions=max_sessions)["net"] for o in offs])


def main() -> None:
    nq = build_sessions("nq")
    strat, _, n_dev = build("nq", "vwap_atr", {"k": 0.3, "s": 0.2, "start_min": st.CASH_OPEN})
    n = len(strat.dates)
    yrs = (strat.dates[-1] - strat.dates[n_dev]).astype(float) / 365.25
    cut = lambda s: Stream(s.name, s.dates[:n_dev], s.pnl[:n_dev], s.low[:n_dev], s.atr_usd[:n_dev], s.ok[:n_dev])
    coins = [Stream.from_series("coin", st.run(nq, "coin", seed=sd, s=None)) for sd in SEEDS]
    first = int(np.flatnonzero(strat.ok)[0])
    dev_offs = np.linspace(first, n_dev - 760, 12).astype(int)
    hold_offs = range(n_dev, n_dev + 60, 6)
    rows = []
    for path in ("standard", "consistency"):
        for t1 in (1000, 3000):
            for Rc in (3200, 6400, 9600):
                for Rf in (1200, 2400, 3600, 7200):
                    pl = plan(path, t1)
                    ne_dev = np.array([totals(cut(c), cut(c), Rc, Rf, pl, dev_offs, 756) for c in coins])
                    ne_hold = np.array([totals(c, c, Rc, Rf, pl, hold_offs) for c in coins])
                    st_dev = totals(cut(strat), cut(strat), Rc, Rf, pl, dev_offs, 756)
                    st_hold = totals(strat, strat, Rc, Rf, pl, hold_offs)
                    hy_hold = np.mean([totals(c, strat, Rc, Rf, pl, hold_offs) for c in coins[:10]])
                    se = lambda a: a.std(ddof=1) / np.sqrt(len(a))
                    rows.append(dict(path=path, t1=t1, Rc=Rc, Rf=Rf,
                                     noedge_dev_3y=ne_dev.mean(), noedge_dev_se=se(ne_dev), noedge_dev_pct_seeds_pos=(ne_dev > 0).mean(),
                                     noedge_hold=ne_hold.mean(), noedge_hold_se=se(ne_hold), noedge_hold_pct_seeds_pos=(ne_hold > 0).mean(),
                                     noedge_hold_p10=np.percentile(ne_hold, 10), noedge_hold_p90=np.percentile(ne_hold, 90),
                                     strat_dev_3y=st_dev, strat_hold=st_hold, coin_comb_strat_funded_hold=hy_hold))
    df = pd.DataFrame(rows)
    df.to_csv("results/noedge_contract.csv", index=False)
    print(f"{len(df)} settings x 20 coin-flip seeds. Hold-out = {yrs:.1f} years, dev = 3-year windows from 2010-2022.\n")
    show = ["path", "t1", "Rc", "Rf", "noedge_dev_3y", "noedge_dev_se", "noedge_hold", "noedge_hold_se", "noedge_hold_pct_seeds_pos",
            "noedge_hold_p10", "noedge_hold_p90", "strat_dev_3y", "strat_hold", "coin_comb_strat_funded_hold"]
    print(df[show].round(0).to_string(index=False))
    print("\nSettings where NO-EDGE mean is positive:  dev %d / %d,  hold-out %d / %d" % ((df.noedge_dev_3y > 0).sum(), len(df), (df.noedge_hold > 0).sum(), len(df)))
    print("Settings where no-edge mean exceeds 2 standard errors above zero: dev %d, hold-out %d" % ((df.noedge_dev_3y > 2 * df.noedge_dev_se).sum(), (df.noedge_hold > 2 * df.noedge_hold_se).sum()))
    g = df.groupby("Rf")[["noedge_dev_3y", "noedge_hold", "strat_dev_3y", "strat_hold"]].mean().round(0)
    print("\nAverage by funded risk Rf:"); print(g.to_string())
    g = df.groupby("Rc")[["noedge_dev_3y", "noedge_hold", "strat_dev_3y", "strat_hold"]].mean().round(0)
    print("\nAverage by Combine risk Rc:"); print(g.to_string())


if __name__ == "__main__":
    main()
