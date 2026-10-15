"""Diagnostic for the weak farm starts since mid-2024 (descriptive; nothing here selects or changes the baseline).
Farm = one Combine at a time, 250-session windows (about one year after the start), farms started every 10 sessions.
Periods: 'hold-out good' = starts 2023-01-02..2024-06-30, 'since mid-2024' = starts 2024-07-01..(end of data - 250 sessions), 'dev (re-based, exact)' = starts up to 2022 on the re-based streams.
Variants: splits of the 5 slots (incl. a VWAP group) and one-at-a-time changes of the baseline cells R / a1 / t1 / a2. As-modelled costs are the real cost environment in the hold-out years."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import numpy as np
import pandas as pd

import retest as R
import retest_cost as RC
from multi_firm import Policy, run_multifirm

W = 250
B = R.BASE


def cells(**kw):
    c = dict(B); c.update(kw); return c


SPLITS = {"3 noise + 2 gap (baseline)": [("noise", 3, B), ("gap", 2, B)], "5 noise": [("noise", 5, B)], "4 noise + 1 gap": [("noise", 4, B), ("gap", 1, B)], "2 noise + 3 gap": [("noise", 2, B), ("gap", 3, B)],
          "1 noise + 4 gap": [("noise", 1, B), ("gap", 4, B)], "5 gap": [("gap", 5, B)], "5 vwap": [("vwap", 5, B)], "2 noise + 2 gap + 1 vwap": [("noise", 2, B), ("gap", 2, B), ("vwap", 1, B)],
          "1 noise + 2 gap + 2 vwap": [("noise", 1, B), ("gap", 2, B), ("vwap", 2, B)]}
SETTINGS = {"baseline R 9,600 / a1 1.2 / t1 $2,000 / a2 0.8": cells()}
for v in (4800, 6400, 12800):
    SETTINGS[f"Combine R {v:,}"] = [("noise", 3, cells(R=v)), ("gap", 2, cells(R=v))]
for v in (0.6, 0.8, 1.6):
    SETTINGS[f"funded a1 {v}"] = [("noise", 3, cells(a1=v)), ("gap", 2, cells(a1=v))]
for v in (1500.0, 3000.0, 4000.0):
    SETTINGS[f"first payout at ${v:,.0f}"] = [("noise", 3, cells(t1=v)), ("gap", 2, cells(t1=v))]
for v in (0.4, 1.2):
    SETTINGS[f"funded a2 {v}"] = [("noise", 3, cells(a2=v)), ("gap", 2, cells(a2=v))]
SETTINGS["baseline R 9,600 / a1 1.2 / t1 $2,000 / a2 0.8"] = [("noise", 3, B), ("gap", 2, B)]


def run_set(ST, ref, groups_by_name, starts):
    out = {}
    for name, groups in groups_by_name.items():
        fps = R.farm_plans(ST, groups)
        out[name] = np.array([run_multifirm(fps, Policy(reserve_cycles=None, global_one_eval=True), i0, i0 + W)["net"] for i0 in starts])
    return out


def summarize(label, nets):
    return dict(set=label, n=len(nets), mean=nets.mean(), median=np.median(nets), p25=np.percentile(nets, 25), pos=(nets > 0).mean(), worst=nets.min())


def main() -> None:
    D = R.load(); ref = D["ref"]; n = D["n"]; d = pd.DatetimeIndex(ref.dates)
    gs = R.K.with_risk_unit(R.K.group_stream(["nq_vwap"]), "atr", 40)
    assert np.array_equal(gs.dates, ref.dates), "vwap stream dates differ"
    ST = dict(D["ST"]); ST["vwap"] = gs
    ix = lambda s: int(np.searchsorted(ref.dates, np.datetime64(s)))
    first = int(np.flatnonzero(ref.ok)[0])
    periods = {"hold-out good (starts 2023-01..2024-06)": list(range(ix("2023-01-02"), ix("2024-07-01"), 10)), "since mid-2024 (starts 2024-07..)": list(range(ix("2024-07-01"), n - W - 5, 10))}
    print({k: (len(v), str(d[v[0]].date()), str(d[v[-1]].date())) for k, v in periods.items()}, flush=True)
    pd.set_option("display.width", 250)
    for title, table in (("SPLITS (diversity)", SPLITS), ("SETTINGS, one at a time", SETTINGS)):
        if "--dev-only" in sys.argv:
            break
        rows = []
        for per, st in periods.items():
            res = run_set(ST, ref, table, st)
            rows += [dict(period=per, **summarize(k, v)) for k, v in res.items()]
        df = pd.DataFrame(rows); df.to_csv("results/diag_farm_" + ("splits" if "SPLIT" in title else "settings") + ".csv", index=False)
        print("\n=====", title, "(250-session windows, real cost era)"); print(df.round(0).set_index(["period", "set"]).to_string(), flush=True)
    # same variants on the re-based DEV farms (no hold-out data involved), to see whether dev points the same way
    STx = RC.rebase({k: ST[k] for k in ("noise", "gap")}, 1.0, "exact")           # the cost cache holds the two farm sleeves only
    SPLITS_DEV = {k: v for k, v in SPLITS.items() if "vwap" not in k}
    dev_starts = list(range(first, R.farm_starts(ref, D["n_dev"])[-1] + 1, 10))
    dev_starts = [i for i in dev_starts if i + W < D["n_dev"]]
    rows = []
    for title, table in (("splits", SPLITS_DEV), ("settings", SETTINGS)):
        res = run_set(STx, ref, table, dev_starts)
        rows += [dict(kind=title, **summarize(k, v)) for k, v in res.items()]
    df = pd.DataFrame(rows); df.to_csv("results/diag_farm_dev_exact.csv", index=False)
    print("\n===== SAME VARIANTS on the re-based DEV farms (250-session windows, starts up to 2022; exact basis, 1 tick)"); print(df.round(0).set_index(["kind", "set"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
