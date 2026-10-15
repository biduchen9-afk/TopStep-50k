"""The 'contract' as the user specified it, selected on 2010-2022 only, evaluated on 2023-2026 with FRESH seeds.

Pipeline: Combine -> pass -> funded -> first payout at trigger t1 (50% of balance) -> every later payout taken the
moment it is eligible (max allowed) -> trade until the account dies -> new Combine -> repeat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import build_sessions
from real_engine import FundedPlan, Stream
from chain_real import replay_detail
from run_holdout import PATH
from run_final_checks import build, cfg, RULES

pd.set_option("display.width", 250, "display.max_columns", 40)
DEV_SEEDS, HOLD_SEEDS = range(0, 10), range(100, 140)


def plan(t1, r2):
    return FundedPlan(f"standard t1={t1} r2x{r2}", "standard", t1, 0.5, 1.0, 0.0, 0.5, r2, 99)


def agg(reps: list, yrs: float) -> dict:
    r = pd.DataFrame(reps)
    nets = r.net.values
    funded = max(r.funded.sum(), 1)
    return dict(mean=nets.mean(), per_year=nets.mean() / yrs, se=nets.std(ddof=1) / np.sqrt(len(nets)) if len(nets) > 1 else np.nan,
                p10=np.percentile(nets, 10), median=np.median(nets), p90=np.percentile(nets, 90), pct_loss=float((nets < 0).mean()),
                fees=r.fees.mean(), payouts=r.payouts.mean(), funded=r.funded.mean(), n_payouts=r.n_payouts.mean(),
                per_funded_extraction=r.payouts.sum() / funded, cost_per_funded=r.fees.sum() / funded,
                p_die_without_payout=r.no_payout_deaths.sum() / funded, days_alive=r.days_funded.sum() / funded)


def main() -> None:
    nq = build_sessions("nq")
    strat, _, n_dev = build("nq", "vwap_atr", {"k": 0.3, "s": 0.2, "start_min": st.CASH_OPEN})
    yrs = (strat.dates[-1] - strat.dates[n_dev]).astype(float) / 365.25
    cut = lambda s: Stream(s.name, s.dates[:n_dev], s.pnl[:n_dev], s.low[:n_dev], s.atr_usd[:n_dev], s.ok[:n_dev])
    first = int(np.flatnonzero(strat.ok)[0])
    dev_offs = np.linspace(first, n_dev - 760, 12).astype(int)
    hold_offs = list(range(n_dev, n_dev + 60, 6))
    dev_coins = [cut(Stream.from_series("c", st.run(nq, "coin", seed=s, s=None))) for s in DEV_SEEDS]
    hold_coins = [Stream.from_series("c", st.run(nq, "coin", seed=s, s=None)) for s in HOLD_SEEDS]

    rows = []
    for Rc in (3200, 6400, 9600):
        for Rf in (2400, 3600, 7200, 10800, 14400):
            for t1 in (1000, 3000, 5000, 8000):
                for r2 in (0.5, 1.0, 2.0):
                    pl = plan(t1, r2)
                    ne = [replay_detail(c, Rc, c, Rf, pl, cfg, RULES, PATH, int(o), max_sessions=756)["net"] for c in dev_coins for o in dev_offs]
                    sd = [replay_detail(cut(strat), Rc, cut(strat), Rf, pl, cfg, RULES, PATH, int(o), max_sessions=756)["net"] for o in dev_offs]
                    rows.append(dict(Rc=Rc, Rf=Rf, t1=t1, r2=r2, dev_noedge_3y=np.mean(ne), dev_noedge_p10=np.percentile(ne, 10), dev_strategy_3y=np.mean(sd)))
    dg = pd.DataFrame(rows)
    dg.to_csv("results/contract_dev_grid.csv", index=False)
    print(f"=== DEV (2010-2022): {len(dg)} configs x ({len(DEV_SEEDS)} coin seeds x {len(dev_offs)} 3-year windows) ===")
    print("Top 8 by NO-EDGE dev profit (per 3 years):"); print(dg.sort_values("dev_noedge_3y", ascending=False).head(8).round(0).to_string(index=False))
    print("\nAverage no-edge dev profit by trigger t1:"); print(dg.groupby("t1")[["dev_noedge_3y", "dev_strategy_3y"]].mean().round(0).to_string())
    print("\nAverage no-edge dev profit by funded risk Rf:"); print(dg.groupby("Rf")[["dev_noedge_3y", "dev_strategy_3y"]].mean().round(0).to_string())
    print("\nAverage no-edge dev profit by second-stage multiplier r2:"); print(dg.groupby("r2")[["dev_noedge_3y", "dev_strategy_3y"]].mean().round(0).to_string())

    bn = dg.sort_values("dev_noedge_3y", ascending=False).iloc[0]
    bs = dg.sort_values("dev_strategy_3y", ascending=False).iloc[0]
    print(f"\nFrozen NO-EDGE config : Rc{int(bn.Rc)} Rf{int(bn.Rf)} t1={int(bn.t1)} r2x{bn.r2}")
    print(f"Frozen STRATEGY config: Rc{int(bs.Rc)} Rf{int(bs.Rf)} t1={int(bs.t1)} r2x{bs.r2}")

    out = []
    for label, c in (("NO EDGE, config picked on dev (40 fresh seeds)", bn), ("NO EDGE, strategy's dev config (40 seeds)", bs)):
        pl = plan(int(c.t1), c.r2)
        reps = [replay_detail(s, c.Rc, s, c.Rf, pl, cfg, RULES, PATH, o) for s in hold_coins for o in hold_offs]
        per_seed = [np.mean([x["net"] for x in reps[i * len(hold_offs):(i + 1) * len(hold_offs)]]) for i in range(len(hold_coins))]
        a = agg([{**x} for x in reps], yrs); a["mean"] = np.mean(per_seed); a["se"] = np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))
        a["p10"], a["median"], a["p90"] = np.percentile(per_seed, 10), np.median(per_seed), np.percentile(per_seed, 90)
        a["pct_loss"] = float((np.array(per_seed) < 0).mean()); a["per_year"] = a["mean"] / yrs
        out.append(dict(setup=label, **a))
        hy = [replay_detail(s, c.Rc, strat, c.Rf, pl, cfg, RULES, PATH, o) for s in hold_coins[:20] for o in hold_offs]
        ps = [np.mean([x["net"] for x in hy[i * len(hold_offs):(i + 1) * len(hold_offs)]]) for i in range(20)]
        a2 = agg(hy, yrs); a2["mean"] = np.mean(ps); a2["se"] = np.std(ps, ddof=1) / np.sqrt(20); a2["p10"], a2["median"], a2["p90"] = np.percentile(ps, 10), np.median(ps), np.percentile(ps, 90)
        a2["pct_loss"] = float((np.array(ps) < 0).mean()); a2["per_year"] = a2["mean"] / yrs
        out.append(dict(setup=label.split(",")[0] + ", coin-flip Combine -> strategy funded (20 seeds)", **a2))
    for label, c in (("STRATEGY both phases, strategy's dev config", bs), ("STRATEGY both phases, no-edge dev config", bn)):
        pl = plan(int(c.t1), c.r2)
        reps = [replay_detail(strat, c.Rc, strat, c.Rf, pl, cfg, RULES, PATH, o) for o in range(n_dev, n_dev + 60, 2)]
        a = agg(reps, yrs); out.append(dict(setup=label, **a))
    res = pd.DataFrame(out); res.to_csv("results/contract_holdout.csv", index=False)
    cols = ["setup", "mean", "per_year", "se", "p10", "median", "p90", "pct_loss", "fees", "payouts", "funded", "n_payouts", "per_funded_extraction", "cost_per_funded", "p_die_without_payout", "days_alive"]
    print(f"\n=== HOLD-OUT 2023-01 -> 2026-07 ({yrs:.1f} years), one account at a time ===")
    print(res[cols].round(1).to_string(index=False))
    print("\nTheory check (zero-edge fair game): expected extraction per funded account ~ 0.9 x $4,500 x P(die before first payout)")
    for _, r in res.iterrows():
        print(f"  {r.setup[:62]:<62} predicted {0.9 * 4500 * r.p_die_without_payout:7.0f} | simulated {r.per_funded_extraction:7.0f}")


if __name__ == "__main__":
    main()
