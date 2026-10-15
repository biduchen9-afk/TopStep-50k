"""PR16 extras: the user-target frontier (from the S1 grid) and the stress tests. python run_retest_extra.py frontier | stress <variant>
stress variants: base | slip100 | slip250 | ladder30 | price229   (slip variants set TOPSTEP_DLL_SLIP before the engine is imported)"""
from __future__ import annotations

import json
import os
import sys

STAGE = sys.argv[1]; VARIANT = sys.argv[2] if len(sys.argv) > 2 else "base"
if VARIANT.startswith("slip"):
    os.environ["TOPSTEP_DLL_SLIP"] = VARIANT[4:]
from dataclasses import replace

import numpy as np
import pandas as pd

import retest as T

OUT = "results"
BASE = (T.BASE["a1"], T.BASE["t1"], T.BASE["a2"], T.BASE["R"])
pd.set_option("display.width", 250, "display.max_columns", 40)


def frontier() -> None:
    R = pd.read_csv(f"{OUT}/retest_s1_real.csv"); bf = json.load(open(f"{OUT}/retest_base_farm.json"))
    ws, wf = bf["w_start"], bf["w_fund"]
    N = R[R.group == "noise"].reset_index(drop=True); G = R[R.group == "gap"].reset_index(drop=True)
    rows = []
    n = len(N); g = len(G)
    # vectors over all (noise cell, gap cell) pairs
    tau_n, tau_g = N.mean_tau.values[:, None], G.mean_tau.values[None, :]
    ts_n = ws["noise"] * tau_n / (ws["noise"] * tau_n + ws["gap"] * tau_g); ts_g = 1 - ts_n
    rate = ts_n * N.rate.values[:, None] + ts_g * G.rate.values[None, :]
    paid = wf["noise"] * N.paid_share.values[:, None] + wf["gap"] * G.paid_share.values[None, :]
    w90 = wf["noise"] * N.within_90.values[:, None] + wf["gap"] * G.within_90.values[None, :]
    paid_in90 = w90 / paid                                  # share of PAID accounts whose first payout came within 90 days of the Combine start; median <= 90 days iff this >= 0.5
    base_i = N[(N.a1 == BASE[0]) & (N.t1 == BASE[1]) & (N.a2 == BASE[2]) & (N.R == BASE[3])].index[0]; base_j = G[(G.a1 == BASE[0]) & (G.t1 == BASE[1]) & (G.a2 == BASE[2]) & (G.R == BASE[3])].index[0]
    print(f"baseline pair: mix rate {rate[base_i, base_j]:.1f}/session (= ${rate[base_i, base_j] * T.WINDOW:,.0f} per mature 3 years), paid at farm mix {paid[base_i, base_j]:.1%}, paid within 90 days {w90[base_i, base_j]:.1%} of funded, median-days<=90: {paid_in90[base_i, base_j] >= 0.5}")
    def best(mask, label):
        m = np.where(mask, rate, -np.inf)
        if not np.isfinite(m).any():
            print(f"{label}: no pair of the 216 x 216 qualifies"); return None
        i, j = np.unravel_index(np.argmax(m), m.shape)
        a, b = N.iloc[i], G.iloc[j]
        d = dict(label=label, rate=float(rate[i, j]), mature_3yr=float(rate[i, j] * T.WINDOW), paid_mix=float(paid[i, j]), within_90=float(w90[i, j]), paid_in90=float(paid_in90[i, j]),
                 noise=f"a1 {a.a1} t1 {a.t1:.0f} a2 {a.a2} R {a.R:.0f}", gap=f"a1 {b.a1} t1 {b.t1:.0f} a2 {b.a2} R {b.R:.0f}")
        print(d); return d
    out = [d for d in (best(paid >= 0.80, "paid >= 80% at the farm mix"), best((paid >= 0.80) & (paid_in90 >= 0.5), "paid >= 80% and median days to first payout <= 90"),
                       best((paid >= 0.70) & (paid_in90 >= 0.5), "paid >= 70% and median days <= 90"), best((paid >= 0.60) & (paid_in90 >= 0.5), "paid >= 60% and median days <= 90"),
                       best(paid_in90 >= 0.5, "median days <= 90 (any paid share)")) if d]
    # descriptive: highest paid share at each level of retained rate
    for keep in (1.0, 0.9, 0.75, 0.5):
        m = np.where(rate >= keep * rate[base_i, base_j], paid, -np.inf); i, j = np.unravel_index(np.argmax(m), m.shape)
        print(f"max paid share keeping >= {keep:.0%} of the baseline rate: {paid[i, j]:.1%} (rate {rate[i, j]:.1f}, within 90 days {w90[i, j]:.1%}) noise [{N.iloc[i].a1}, {N.iloc[i].t1:.0f}, {N.iloc[i].a2}, {N.iloc[i].R:.0f}] gap [{G.iloc[j].a1}, {G.iloc[j].t1:.0f}, {G.iloc[j].a2}, {G.iloc[j].R:.0f}]")
        out.append(dict(label=f"max paid share keeping >= {keep:.0%} of the baseline rate", rate=float(rate[i, j]), paid_mix=float(paid[i, j]), within_90=float(w90[i, j]), noise=f"a1 {N.iloc[i].a1} t1 {N.iloc[i].t1:.0f} a2 {N.iloc[i].a2} R {N.iloc[i].R:.0f}", gap=f"a1 {G.iloc[j].a1} t1 {G.iloc[j].t1:.0f} a2 {G.iloc[j].a2} R {G.iloc[j].R:.0f}"))
    json.dump(out, open(f"{OUT}/retest_frontier.json", "w"))


def stress(variant: str) -> None:
    spec = T.S["spec"]
    if variant == "ladder30":
        T.S["spec"] = replace(spec, funded=replace(spec.funded, ladder=((0.0, 3), (3000.0, 10), (4500.0, 15))))
    elif variant == "price229":
        T.S["spec"] = replace(spec, eval_fee=229.0)
    D = T.load(); ST, ref, n_dev = D["ST"], D["ref"], D["n_dev"]
    cells = {"baseline": {g: dict(a1=BASE[0], t1=BASE[1], a2=BASE[2], R=BASE[3]) for g in ST}}
    sel = f"{OUT}/retest_s1_selected.json"
    if os.path.exists(sel):
        ch = json.load(open(sel))
        if any(ch.values()):
            cells["selected S1"] = {g: (ch.get(g) or cells["baseline"][g]) for g in ST}
    rows = []
    for lab, cp in cells.items():
        part = {g: T.AcctEval(ST[g], n_dev, cp[g]["R"]).run(cp[g]["a1"], cp[g]["t1"], cp[g]["a2"]) for g in ST}
        rate = {g: T.summ(part[g])["rate"] for g in ST}; tau = {g: T.summ(part[g])["mean_tau"] for g in ST}
        bf = json.load(open(f"{OUT}/retest_base_farm.json")); ts = T.time_shares(bf["w_start"], tau)
        fs = T.farm_stats(T.farm_plans(ST, [("noise", 3, cp["noise"]), ("gap", 2, cp["gap"])]), ref, n_dev, seen=False)
        rows.append(dict(variant=variant, config=lab, noise_rate=rate["noise"], gap_rate=rate["gap"], mix_rate=sum(ts[g] * rate[g] for g in ST), farm_mean=fs["mean"], farm_median=fs["median"], farm_p25=fs["p25"], share_pos=fs["share_pos"],
                         paid_share_farm=fs["paid_share_farm"], worst=fs["worst"]))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/retest_stress_{variant}.csv", index=False)


if __name__ == "__main__":
    frontier() if STAGE == "frontier" else stress(VARIANT)
