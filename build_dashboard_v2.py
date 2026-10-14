"""Dashboard data for the regime study: previous headline vs the dev-selected config (with and without the pass-rate throttle) vs a
no-edge control on the same rules, month by month over the full history. Writes JSON files for the dashboard datasets."""
from __future__ import annotations
import json, pickle, sys
import numpy as np, pandas as pd
import intraday_strategies as st
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from market_data import build_sessions
from real_engine import Stream, costs_from_config
from risk_units import with_risk_unit
from run_corrected import NEW, cfg, rules, idx_of, streams as old_streams
from run_farm_dev import plan
from run_dynamic_contract import PATH
from build_dashboard_data import monthly, finish, idx_first

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
G = pd.concat([pd.read_pickle("results/regime_dev_grid.pkl"), pd.read_pickle("results/regime_dev_grid_ext.pkl")], ignore_index=True)
STREAMS = pickle.load(open("/home/user/data_raw/regime_streams.pkl", "rb"))
top = G.sort_values("obj", ascending=False).iloc[0].to_dict()
SEL = with_risk_unit(STREAMS[(top["set"], top["rule"])], top["unit"], 40)
NQ = old_streams(NEW)["NQ noise-area"]
A = dict(R=6400, a1=0.4, a2=1.6, refill="all", pause=0.0)


def spec(row, **kw):
    r = dict(row, **kw)
    return CombineSizing("fixed", float(r["R"])), plan(float(r["a1"]), float(r["a2"])), FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill=r["refill"], pause_below=float(r["pause"]))


def go(s, row, i0=None, log=False, **kw):
    sz, pl, sp = spec(row, **kw)
    return run_farm([s], sz, None, pl, cfg, rules, PATH, idx_first(s) if i0 is None else i0, sp, log=log)


def main() -> None:
    V = [("selected", "Selected on 2010-22 data (NQ noise + VWAP + gap-VWAP)", "Settings picked on 2010-22 only, then frozen. 2023+ was already seen by us.", SEL, top, {}),
         ("selected_throttle", "Selected + pause when recent Combines mostly fail", "Same as above, new Combines pause when under 10% of recent ones pass.", SEL, top, dict(pause=0.10)),
         ("previous", "Previous headline (NQ noise-area, a2=1.6)", "The configuration shown earlier, for comparison.", NQ, A, {})]
    variants = []
    for vid, label, note, s, row, kw in V:
        r = go(s, row, log=True, **kw)
        variants.append(dict(id=vid, label=label, note=note, months=finish(monthly(s, r, label)), totals=dict(net=r["net"], payouts=r["payouts"], fees=r["fees"], funded=r["n_funded"], n_payouts=r["n_payouts"])))
    nq_sess = build_sessions("nq"); acc = None; tot = []
    for sd in range(10):
        c = with_risk_unit(Stream.from_series("coin", st.run(nq_sess, "coin", seed=sd, s=None, costs=costs_from_config(cfg))), top["unit"], 40)
        r = go(c, top, log=True)
        df = pd.DataFrame(monthly(c, r, "coin")).set_index("m")
        acc = df if acc is None else acc.add(df, fill_value=0.0); tot.append(r["net"])
    rows = (acc / 10).reset_index().to_dict("records")
    for r in rows:
        r["payout_count"] = float(r["payout_count"])
    variants.append(dict(id="control", label="No-edge control (coin-flip) on the selected rules, mean of 10", note="Shows what the payout rules alone give with zero edge.",
                         months=finish(rows), totals=dict(net=float(np.mean(tot)))))
    # ---- flatten to dataset rows
    mrows, vrows, yrows = [], [], []
    for v in variants:
        ms = pd.DataFrame(v["months"])
        ms["month"] = ms.m + "-01"
        ms["sample"] = np.where(ms.m <= "2022-12", "in-sample", "out-of-sample")
        for r in ms.to_dict("records"):
            mrows.append(dict(variant=v["id"], month=r["month"], sample=r["sample"], payouts=round(r["payouts"], 2), payout_count=round(float(r["payout_count"]), 2), fees=round(r["fees"], 2),
                              net=round(r["net"], 2), cum_net=round(r["cum_net"], 2), cum_payouts=round(r["cum_payouts"], 2), cum_fees=round(r["cum_fees"], 2),
                              evals_passed=round(float(r["evals_passed"]), 2), evals_failed=round(float(r["evals_failed"]), 2), activations=round(float(r["activations"]), 2),
                              funded_died=round(float(r["funded_died"]), 2), funded_closed=round(float(r["funded_closed"]), 2), avg_evals=round(r["avg_evals"], 2), avg_funded=round(r["avg_funded"], 2)))
        pay, fee = float(ms.payouts.sum()), float(ms.fees.sum())
        vrows.append(dict(id=v["id"], label=v["label"], note=v["note"], net_total=round(pay - fee, 2), net_in_sample=round(float(ms[ms["sample"] == "in-sample"].net.sum()), 2),
                          net_out_of_sample=round(float(ms[ms["sample"] == "out-of-sample"].net.sum()), 2), payouts_total=round(pay, 2), fees_total=round(fee, 2),
                          reinvest_pct=round(fee / pay, 4) if pay > 0 else 0.0, peak_outlay=round(float(-min(0.0, ms.cum_net.min())), 2),
                          funded_accounts=int(round(float(ms.activations.sum()))), payout_count=int(round(float(ms.payout_count.sum()))), months=int(len(ms))))
        ms["year"] = ms.m.str[:4]
        for y, g in ms.groupby("year"):
            yrows.append(dict(variant=v["id"], year=y, payouts=round(float(g.payouts.sum()), 2), fees=round(float(g.fees.sum()), 2), net=round(float(g.net.sum()), 2),
                              payout_count=round(float(g.payout_count.sum()), 2), activations=round(float(g.activations.sum()), 2), funded_died=round(float(g.funded_died.sum()), 2),
                              avg_funded=round(float(g.avg_funded.mean()), 2), sample="in-sample" if y <= "2022" else "out-of-sample"))
    srows = []
    starts = ["2012-01-03", "2015-01-02", "2018-01-02", "2019-01-02", "2021-01-04", "2023-01-02", "2024-01-02", "2025-01-02", "2025-07-01"]
    for vid, label, _, s, row, kw in V:
        for d in starts:
            r = go(s, row, i0=idx_of(s, d), **kw)
            srows.append(dict(config=label, start=d, net=round(r["net"], 2), payouts=round(r["payouts"], 2), fees=round(r["fees"], 2), n_payouts=int(r["n_payouts"]), funded=int(r["n_funded"])))
    import os
    os.makedirs(OUT, exist_ok=True)
    for name, data in (("monthly", mrows), ("variants", vrows), ("yearly", yrows), ("start_dates", srows)):
        json.dump(data, open(f"{OUT}/{name}.json", "w"))
    print(pd.DataFrame(vrows)[["id", "net_total", "net_in_sample", "net_out_of_sample", "payouts_total", "fees_total", "reinvest_pct", "peak_outlay", "funded_accounts"]].to_string(index=False))
    print(pd.DataFrame(srows).pivot_table(index="start", columns="config", values="net").round(0).to_string())


if __name__ == "__main__":
    main()
