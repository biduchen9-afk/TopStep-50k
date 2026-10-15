"""Descriptive view of the hold-out (2023-01 to 2026-07-02) next to the development years, same evaluator, nothing selected.

Account level: Combine started every 3 sessions, baseline cells (a1 1.2, first payout at $2,000, a2 0.8, R 9,600), funded accounts followed to the END OF THE DATA (2026-07-02),
by calendar year of the start. Farm level: the 3 noise + 2 gap baseline farm, started every 10 sessions, for windows of 378 and 504 sessions (a 756-session window does not fit after 2023),
as modelled (the dev cost of $1.72 per micro, which is about today's real cost-to-ATR in the hold-out years) and on the exact re-based cost basis.
Run with TOPSTEP_RAW_DIR=/home/user/data_raw_live (retest.py binds it by default).
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import numpy as np
import pandas as pd

import retest as R
import retest_cost as RC
from multi_firm import Policy, run_multifirm

HOLD_START = np.datetime64("2023-01-02")
BASE = R.BASE
OUT = "results/holdout_view"


def account_level(D: dict) -> pd.DataFrame:
    rows = []
    for g in ("noise", "gap"):
        s = D["ST"][g]; n = D["n"]
        ev = R.AcctEval(s, n, BASE["R"], starts=None)           # n_dev = full length: funded life runs to the end of the data
        df = ev.run(BASE["a1"], BASE["t1"], BASE["a2"])
        df["year"] = df.start.dt.year; df["group"] = g
        rows.append(df)
    return pd.concat(rows)


def year_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (g, y), x in df.groupby(["group", "year"]):
        m = R.summ(x)
        out.append(dict(group=g, year=y, starts=m["starts"], pass_rate=m["pass_rate"], funded=m["funded"], paid_share=m["paid_share"], rate=m["rate"], ev=m["ev"]))
    return pd.DataFrame(out)


def farm_nets(ST: dict, ref, windows=(378, 504)) -> pd.DataFrame:
    fps = R.farm_plans(ST, [("noise", 3, BASE), ("gap", 2, BASE)])
    pol = lambda: Policy(reserve_cycles=None, global_one_eval=True)
    first = int(np.flatnonzero(ref.ok)[0]); n = len(ref.dates); d = pd.DatetimeIndex(ref.dates)
    rows = []
    for w in windows:
        for i0 in range(first, n - w + 1, R.STEP):
            r = run_multifirm(fps, pol(), i0, i0 + w)
            rows.append(dict(window=w, start=d[i0], year=d[i0].year, net=r["net"], hold=bool(d[i0] >= pd.Timestamp(HOLD_START))))
    return pd.DataFrame(rows)


def main() -> None:
    D = R.load(); ref = D["ref"]
    print("sessions", D["n"], "dev end index", D["n_dev"], "last date", pd.Timestamp(ref.dates[-1]).date(), flush=True)
    acct = account_level(D); yt = year_table(acct); yt.to_csv(OUT + "_account_by_year.csv", index=False)
    print(yt.round(3).to_string(), flush=True)
    res = {}
    for label, ST in (("as modelled", D["ST"]), ("exact re-based 1 tick", RC.rebase(D["ST"], 1.0, "exact"))):
        fn = farm_nets(ST, ref); fn["basis"] = label
        res[label] = fn
        fn.to_csv(OUT + f"_farms_{'modelled' if label == 'as modelled' else 'exact'}.csv", index=False)
        for w in (378, 504):
            x = fn[fn.window == w]
            for nm, s in (("dev starts", x[~x.hold]), ("hold-out starts", x[x.hold])):
                if len(s) == 0:
                    continue
                q = s.net.quantile([.25, .5]).values
                print(f"[{label}] window {w} {nm}: n={len(s)} mean {s.net.mean():.0f} median {q[1]:.0f} p25 {q[0]:.0f} positive {(s.net > 0).mean():.2f} worst {s.net.min():.0f}", flush=True)


if __name__ == "__main__":
    main()
