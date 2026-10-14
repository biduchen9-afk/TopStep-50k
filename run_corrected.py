"""Re-run the frozen configs with realistic MICRO fees ($1.22 all-in per micro round turn + 1 tick) and by start date."""
from __future__ import annotations
import json, pickle
import numpy as np
import pandas as pd
import intraday_strategies as st
from market_data import build_sessions
from pricing_config import load_config
from real_engine import FundedPlan, Stream, costs_from_config, rules_from_config
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from portfolio import DEV_END, basket, dev_weights, make_sleeve, r_series, sharpe
from run_farm_dev import plan, SZ
from run_dynamic_contract import PATH

pd.set_option("display.width", 250, "display.max_columns", 40)
cfg = load_config(); rules = rules_from_config(cfg, PATH)
W = pickle.load(open("/home/user/data_raw/portfolio_weights.pkl", "rb"))
OLD = st.Costs(2.5, 1.0)
NEW = costs_from_config(cfg)


def sleeves(costs):
    return [make_sleeve("nq", "noise_area", {}, costs=costs, name="nq_noise"), make_sleeve("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN), costs=costs, name="nq_vwap"),
            make_sleeve("es", "noise_area", {}, costs=costs, name="es_noise"), make_sleeve("es", "vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN), dict(trend=250), costs=costs, name="es_vwap_t250"),
            make_sleeve("gc", "noise_area", {}, dict(trend=60), costs=costs, name="gc_noise_t60")]


def streams(costs):
    S = sleeves(costs)
    return {"NQ noise-area": basket([S[0]], name="nq"), "Portfolio": basket(S, weights=W, name="pf")}


def sh(s):
    r = r_series(s); ok = s.ok; dev = s.dates <= DEV_END
    return sharpe(r[ok]), sharpe(r[ok & dev]), sharpe(r[ok & ~dev])


def farm(s, pl, i0, n_end=None, log=False, me=10):
    return run_farm([s], SZ, None, pl, cfg, rules, PATH, i0, FarmSpec(max_evals=me, max_funded=5, p_est=1.0), n_end=n_end, log=log)


def idx_of(s, date):
    return int(np.searchsorted(s.dates, np.datetime64(date)))


def main() -> None:
    old, new = streams(OLD), streams(NEW)
    print("=== Sharpe (ATR-normalised): old costs ($2.50/mini-eq + 1 tick) vs realistic micro fees ($1.22/micro + 1 tick) ===")
    for k in old:
        print(f"{k:<15} old full/dev/hold {tuple(round(x,2) for x in sh(old[k]))} | new {tuple(round(x,2) for x in sh(new[k]))}")
    configs = [("NQ noise-area", "dev-frozen a1=.4 a2=1.6", plan(0.4, 1.6)), ("NQ noise-area", "post-hoc a1=.4 a2=.4", plan(0.4, 0.4)), ("Portfolio", "dev-frozen a1=.4 a2=.4", plan(0.4, 0.4))]
    rows = []
    for stream, lab, pl in configs:
        for cname, S in (("old costs", old), ("realistic micro fees", new)):
            s = S[stream]
            n_h = idx_of(s, "2023-01-01")
            for start in ("2023-01-02", "2024-01-02", "2025-01-02", "2025-07-01"):
                r = farm(s, pl, idx_of(s, start))
                rows.append(dict(stream=stream, config=lab, costs=cname, start=start, net=r["net"], payouts=r["payouts"], fees=r["fees"], funded=r["n_funded"], n_payouts=r["n_payouts"]))
    df = pd.DataFrame(rows)
    df.to_csv("results/corrected_costs_farm_by_start.csv", index=False)
    print("\n=== 5-funded farm, net $ from each start date to the end of data (2026-07) ===")
    print(df.pivot_table(index=["stream", "config", "costs"], columns="start", values="net").round(0).to_string())
    print("\n(payout counts)"); print(df.pivot_table(index=["stream", "config", "costs"], columns="start", values="n_payouts").round(0).to_string())


if __name__ == "__main__":
    main()
