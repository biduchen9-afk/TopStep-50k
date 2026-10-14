"""Walk-forward check of the whole selection procedure (stage 3). For each test year Y (2016..2026): pick the best configuration of the full
declared grid by the dev objective computed ONLY on 3-year windows that end before Y, then run that configuration from the first session of
Y for 3 years (or to the end of the data; fresh farm). Compare with the previous headline (A) and with the single config chosen once on 2010-22 (B)."""
from __future__ import annotations
import pickle, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from risk_units import with_risk_unit
from run_corrected import NEW, cfg, rules, streams as old_streams
from run_farm_dev import plan
from run_dynamic_contract import PATH

G = pd.concat([pd.read_pickle("results/regime_dev_grid.pkl"), pd.read_pickle("results/regime_dev_grid_ext.pkl")], ignore_index=True)
STREAMS = pickle.load(open("/home/user/data_raw/regime_streams.pkl", "rb"))
_CACHE = {}


def stream(set_, rule, unit):
    k = (set_, rule, unit)
    if k not in _CACHE:
        _CACHE[k] = with_risk_unit(STREAMS[k[:2]], unit, 40)
    return _CACHE[k]


def spec_of(r):
    return CombineSizing("fixed", float(r["R"])), plan(float(r["a1"]), float(r["a2"])), FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill=r["refill"], pause_below=float(r["pause"]))


def year_job(Y: int):
    base = STREAMS[("P2 NQ only (noise, vwap, gap30)", "static")]
    d = pd.DatetimeIndex(base.dates)
    first = int(np.flatnonzero(base.ok)[0]) + 260
    y0 = int(np.searchsorted(d, pd.Timestamp(f"{Y}-01-01")))
    y1 = min(y0 + 756, len(d))                 # test span: 3 years (or to the end of the data)
    last_train = y0 - 1
    if last_train - 756 <= first:
        return Y, None
    wins = np.linspace(first, last_train - 756, 6).astype(int)
    best, best_obj = None, -1e18
    for _, r in G.iterrows():
        s = stream(r["set"], r["rule"], r["unit"]); sz, pl, sp = spec_of(r)
        net = np.array([run_farm([s], sz, None, pl, cfg, rules, PATH, int(o), sp, n_end=int(o) + 756)["net"] for o in wins])
        obj = 0.5 * (net.mean() + np.percentile(net, 10))
        if obj > best_obj:
            best, best_obj = r, obj
    s = stream(best["set"], best["rule"], best["unit"]); sz, pl, sp = spec_of(best)
    res = run_farm([s], sz, None, pl, cfg, rules, PATH, y0, sp, n_end=y1)
    return Y, dict(chosen=f'{best["set"][:2]} {best["rule"]} {best["unit"]} R{int(best["R"])} a{best["a1"]}/{best["a2"]} {best["refill"]} p{best["pause"]}', wf_net=res["net"], wf_pay=res["payouts"], wf_fees=res["fees"])


def fixed_net(s, row, Y):
    d = pd.DatetimeIndex(s.dates)
    y0 = int(np.searchsorted(d, pd.Timestamp(f"{Y}-01-01"))); y1 = min(y0 + 756, len(d))
    sz, pl, sp = spec_of(row)
    return run_farm([s], sz, None, pl, cfg, rules, PATH, y0, sp, n_end=y1)["net"]


if __name__ == "__main__":
    years = list(range(2016, 2027))
    with Pool(4) as p:
        out = dict(p.map(year_job, years, chunksize=1))
    top = G.sort_values("obj", ascending=False).iloc[0]
    sB = stream(top["set"], top["rule"], top["unit"])
    nq = old_streams(NEW)["NQ noise-area"]
    A = dict(R=6400, a1=0.4, a2=1.6, refill="all", pause=0.0)
    rows = []
    for Y in years:
        o = out[Y]
        if o is None:
            continue
        rows.append(dict(year=Y, walk_forward=o["wf_net"], wf_payouts=o["wf_pay"], wf_fees=o["wf_fees"], B_fixed_dev_choice=fixed_net(sB, top, Y), A_previous=fixed_net(nq, A, Y), chosen=o["chosen"]))
    df = pd.DataFrame(rows); df.to_csv("results/regime_walkforward.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(df.round(0).to_string(index=False))
    cols = ["walk_forward", "B_fixed_dev_choice", "A_previous"]
    print("\nmean over start years 2016-2026 (windows overlap; compare methods, do not add):", df[cols].mean().round(0).to_dict(), "| positive:", (df[cols] > 0).sum().to_dict())
    print("start years 2023-2026:", df[df.year >= 2023][cols].mean().round(0).to_dict())
