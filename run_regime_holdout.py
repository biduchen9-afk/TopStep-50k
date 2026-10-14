"""Stage 2: the DEV-selected configuration, frozen, on 2023-01 -> 2026-07 (already seen by us, so labelled 'not clean'), by start date,
against (a) the previous headline configuration and (b) a no-edge coin-flip control on identical farm rules.  Nothing here is tuned."""
from __future__ import annotations
import pickle, sys
import numpy as np, pandas as pd
import intraday_strategies as st
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from market_data import build_sessions
from real_engine import Stream
from risk_units import with_risk_unit
from run_corrected import cfg, rules, idx_of
from run_farm_dev import plan
from run_dynamic_contract import PATH

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
STARTS = ("2023-01-02", "2024-01-02", "2025-01-02", "2025-07-01")


def farm_cfg(row):
    return CombineSizing("fixed", float(row["R"])), plan(float(row["a1"]), float(row["a2"])), FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill=row["refill"], pause_below=float(row["pause"]))


def coin_streams(unit, n=8):
    nq = build_sessions("nq")
    return [with_risk_unit(Stream.from_series(f"coin{k}", st.run(nq, "coin", seed=k, s=None)), unit, 40) for k in range(n)]


def run_all(s, row, label):
    sz, pl, spec = farm_cfg(row)
    out = {}
    for d in STARTS:
        r = run_farm([s], sz, None, pl, cfg, rules, PATH, idx_of(s, d), spec)
        out[d] = (r["net"], r["payouts"], r["fees"], r["n_funded"])
    r = run_farm([s], sz, None, pl, cfg, rules, PATH, idx_of(s, "2023-01-02"), spec)
    return out


def main() -> None:
    g = pd.read_pickle("results/regime_dev_grid.pkl")
    streams = pickle.load(open("/home/user/data_raw/regime_streams.pkl", "rb"))
    top = g.sort_values("obj", ascending=False).head(1).iloc[0]
    base_row = dict(R=6400, a1=0.4, a2=1.6, refill="all", pause=0.0, unit="atr")
    cands = {"previous headline (NQ noise, ATR, R=6400, a2=1.6, no throttle)": (streams[("P2 NQ only (noise, vwap, gap30)", "static")], None), }
    print("DEV-selected config:", top[["set", "rule", "unit", "R", "a1", "a2", "refill", "pause", "mean", "p10", "obj"]].to_dict())
    rows = []
    # previous headline: NQ noise-area stream only -> rebuild
    from run_corrected import NEW, streams as old_streams
    nq = old_streams(NEW)["NQ noise-area"]
    for lab, s, row in (("previous headline", nq, base_row), ("DEV-selected", with_risk_unit(streams[(top.set, top.rule)], top.unit, 40), top.to_dict())):
        res = run_all(s, row, lab)
        for d, (net, pay, fee, nf) in res.items():
            rows.append(dict(config=lab, start=d, net=net, payouts=pay, fees=fee, funded=nf))
        # no-edge control, same farm rules
        cs = coin_streams(row["unit"])
        nets = {d: [] for d in STARTS}
        for c in cs:
            r2 = run_all(c, row, lab)
            for d in STARTS:
                nets[d].append(r2[d][0])
        for d in STARTS:
            rows.append(dict(config=lab + " | coin-flip control (mean of 8)", start=d, net=float(np.mean(nets[d])), payouts=np.nan, fees=np.nan, funded=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv("results/regime_holdout.csv", index=False)
    print(df.pivot_table(index="config", columns="start", values="net").round(0).to_string())
    print("\npayouts / fees / funded:"); print(df.dropna().round(0).to_string(index=False))


if __name__ == "__main__":
    main()
