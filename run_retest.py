"""Pre-registration 16: run the stages. python run_retest.py base | s1 | s2 | s3 | all   (results/retest_*.csv; log to stdout)"""
from __future__ import annotations

import itertools
import json
import os
import sys

STAGES = sys.argv[1:] or ["base"]          # read before retest.py clears sys.argv

import numpy as np
import pandas as pd

import retest as T
from scaling_common import placebo_streams

pd.set_option("display.width", 250, "display.max_columns", 60)
A1S, T1S, A2S, RS = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0), (1500.0, 2000.0, 3000.0, 4000.0), (0.4, 0.8, 1.2), (6400.0, 9600.0, 12800.0)
FILTERS = [("gap", 1.0), ("gap", 1.5), ("gap", 2.0), ("rng", 1.5), ("rng", 2.0), ("rng", 2.5), ("vol", 1.5), ("vol", 2.0), ("vol", 2.5)]
OUT = "results"
BASE_CELL = (T.BASE["a1"], T.BASE["t1"], T.BASE["a2"], T.BASE["R"])
D = T.load(); ST, REF, N, N_DEV = D["ST"], D["ref"], D["n"], D["n_dev"]
F = T.features(REF)
HOLD = "STATE"  # tag for the mix file


def tag_check() -> None:
    assert str(T.md.RAW_DIR).rstrip("/").endswith("data_raw_live"), f"retest must run on the repaired data, got {T.md.RAW_DIR}"
    assert os.environ.get("TOPSTEP_RULES", "v2") == "v2"


def base_stage() -> dict:
    tag_check()
    parts = {g: T.AcctEval(ST[g], N_DEV, BASE_CELL[3]).run(*BASE_CELL[:3]) for g in ST}
    fps = T.farm_plans(ST, [("noise", 3, dict(a1=BASE_CELL[0], t1=BASE_CELL[1], a2=BASE_CELL[2], R=BASE_CELL[3])), ("gap", 2, dict(a1=BASE_CELL[0], t1=BASE_CELL[1], a2=BASE_CELL[2], R=BASE_CELL[3]))])
    fs = T.farm_stats(fps, REF, N_DEV)
    w_start, w_fund = T.farm_mix(fs)
    rows = []
    for g, df in parts.items():
        rows.append(dict(group=g, mature_3yr=T.summ(df)["rate"] * T.WINDOW, **T.summ(df)))
    pool = pd.concat([parts["noise"].assign(w=w_fund["noise"]), parts["gap"].assign(w=w_fund["gap"])])
    mixed = dict(group="farm mix (funded accounts)", pass_rate=np.nan, funded=int(fs["funded_total"]),
                 paid_share=sum(w_fund[g] * T.summ(parts[g])["paid_share"] for g in parts), days_med=np.nan,
                 within_90=sum(w_fund[g] * T.summ(parts[g])["within_90"] for g in parts), payouts_per_funded=sum(w_fund[g] * T.summ(parts[g])["payouts_per_funded"] for g in parts),
                 ev=sum(w_start[g] * T.summ(parts[g])["ev"] for g in parts))
    ts = T.time_shares(w_start, {g: T.summ(parts[g])["mean_tau"] for g in parts})
    mixed["rate"] = sum(ts[g] * T.summ(parts[g])["rate"] for g in parts); mixed["mature_3yr"] = mixed["rate"] * T.WINDOW
    # median days at the farm mix: weight paid accounts by the funded-account mix
    pn, pg = parts["noise"], parts["gap"]
    pn = pn[pn.passed & pn.paid]; pg = pg[pg.passed & pg.paid]
    w = np.r_[np.full(len(pn), w_fund["noise"] * T.summ(parts["noise"])["paid_share"] / max(len(pn), 1)), np.full(len(pg), w_fund["gap"] * T.summ(parts["gap"])["paid_share"] / max(len(pg), 1))]
    v = np.r_[pn.d_total.values, pg.d_total.values]; o = np.argsort(v); cw = np.cumsum(w[o]) / w.sum()
    mixed["days_med"] = float(v[o][np.searchsorted(cw, 0.5)])
    rows.append(mixed)
    A = pd.DataFrame(rows); A.to_csv(f"{OUT}/retest_base_account.csv", index=False)
    farm = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in fs.items() if k not in ("nets",)}
    farm["w_start"], farm["w_fund"], farm["time_share"] = w_start, w_fund, ts
    json.dump(farm, open(f"{OUT}/retest_base_farm.json", "w"), default=float)
    np.save(f"{OUT}/retest_base_farm_nets.npy", fs["nets"])
    print("BASELINE on repaired data, rules v2"); print(A.round(3).to_string(index=False))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in farm.items() if k not in ("evals", "funded", "paid")})
    print("mix (Combine starts / funded accounts):", {k: round(v, 3) for k, v in w_start.items()}, {k: round(v, 3) for k, v in w_fund.items()})
    return farm


def grid(streams: dict, g: str, label: str, cells=None) -> pd.DataFrame:
    """Account-level S1 grid for one group on `streams`: rate = EV per Combine session, SE of the paired difference to the baseline cell (year blocks), halves."""
    rows = []; cache = {}
    for R in RS:
        cache[R] = T.AcctEval(streams[g], N_DEV, R)
    ref_df = cache[BASE_CELL[3]].run(*BASE_CELL[:3]); rb = T.summ(ref_df)["rate"]; hb = T.halves_rate(ref_df)
    for R, a1, t1, a2 in itertools.product(RS, A1S, T1S, A2S):
        df = cache[R].run(a1, t1, a2)
        if len(df) != len(ref_df):
            raise RuntimeError("different start sets")
        s = T.summ(df); h = T.halves_rate(df)
        rows.append(dict(set=label, group=g, a1=a1, t1=t1, a2=a2, R=R, rate=s["rate"], gain=s["rate"] - rb, se=T.boot_rate_se(df, ref_df), gain_h0=h[0] - hb[0], gain_h1=h[1] - hb[1], ev=s["ev"], mean_tau=s["mean_tau"], pass_rate=s["pass_rate"],
                         paid_share=s["paid_share"], days_med=s["days_med"], within_90=s["within_90"], payouts_per_funded=s["payouts_per_funded"], fees_per_funded=s["fees_per_funded"], funded=s["funded"]))
    return pd.DataFrame(rows)


def s1_stage() -> None:
    tag_check()
    real, plc = [], []
    PL = placebo_streams(ST)
    for g in ST:
        print(f"S1 grid {g} (real)", flush=True); real.append(grid(ST, g, "real"))
        print(f"S1 grid {g} (placebo)", flush=True); plc.append(grid(PL, g, "placebo"))
    R = pd.concat(real, ignore_index=True); P = pd.concat(plc, ignore_index=True)
    R.to_csv(f"{OUT}/retest_s1_real.csv", index=False); P.to_csv(f"{OUT}/retest_s1_placebo.csv", index=False)
    for g in ST:
        pg = P[P.group == g]; rg = R[R.group == g]
        print(f"\n{g}: best placebo gain over the baseline cell (selection-noise allowance) ${pg.gain.max():,.0f}; best real gain ${rg.gain.max():,.0f}")
        print(rg.sort_values("gain", ascending=False).head(8).round(3).to_string(index=False))



def load_base() -> dict:
    return json.load(open(f"{OUT}/retest_base_farm.json"))


def pick_s1(R: pd.DataFrame, P: pd.DataFrame, bf: dict) -> dict:
    """Apply the PR16 acceptance to the S1 grid, group by group; farm-level checks for the top 5 account-level survivors."""
    chosen = {}; base_nets = np.load(f"{OUT}/retest_base_farm_nets.npy")
    base_q = np.percentile(base_nets, [25, 50])
    for g in ST:
        rg = R[R.group == g]; allowance = float(P[P.group == g].gain.max())
        b = rg[(rg.a1 == BASE_CELL[0]) & (rg.t1 == BASE_CELL[1]) & (rg.a2 == BASE_CELL[2]) & (rg.R == BASE_CELL[3])].iloc[0]
        ok = rg[(rg.gain > 2 * rg.se) & (rg.gain > allowance) & (rg.gain_h0 > 0) & (rg.gain_h1 > 0) & (rg.paid_share >= b.paid_share - 0.05)].sort_values("gain", ascending=False)
        print(f"\nS1 {g}: allowance ${allowance:,.0f}; account-level survivors {len(ok)}", flush=True)
        passers = []
        for _, c in ok.head(5).iterrows():
            groups = [("noise", 3, dict(a1=BASE_CELL[0], t1=BASE_CELL[1], a2=BASE_CELL[2], R=BASE_CELL[3])), ("gap", 2, dict(a1=BASE_CELL[0], t1=BASE_CELL[1], a2=BASE_CELL[2], R=BASE_CELL[3]))]
            for k, (gg, n, pp) in enumerate(groups):
                if gg == g:
                    groups[k] = (gg, n, dict(a1=c.a1, t1=c.t1, a2=c.a2, R=c.R))
            fs = T.farm_stats(T.farm_plans(ST, groups), REF, N_DEV, seen=False)
            why = []
            if not fs["median"] >= base_q[1]: why.append("lower farm median")
            if not fs["p25"] >= base_q[0]: why.append("lower farm p25")
            if not fs["worst"] >= (bf["worst"] * 1.25 if bf["worst"] < 0 else bf["worst"] * 0.75): why.append("deeper worst farm")
            if not fs["paid_share_farm"] >= bf["paid_share_farm"] - 0.05: why.append("fewer accounts paid")
            print(f"   cell a1 {c.a1} t1 {c.t1:.0f} a2 {c.a2} R {c.R:.0f}: rate gain ${c.gain:.2f}/session (SE {c.se:.2f}); farm mean {fs['mean']:,.0f} median {fs['median']:,.0f} p25 {fs['p25']:,.0f} worst {fs['worst']:,.0f} paid {fs['paid_share_farm']:.0%} -> {why or 'PASS'}", flush=True)
            if not why:
                passers.append((fs["p25"], dict(a1=float(c.a1), t1=float(c.t1), a2=float(c.a2), R=float(c.R), ev_gain=float(c.gain), farm_p25=float(fs["p25"]), farm_median=float(fs["median"]), farm_mean=float(fs["mean"]))))
        chosen[g] = max(passers, key=lambda x: x[0])[1] if passers else None
        print(f"S1 {g} selected:", chosen[g] or "none - baseline cell stays")
    json.dump(chosen, open(f"{OUT}/retest_s1_selected.json", "w"))
    return chosen


def s2_stage(chosen: dict, bf: dict) -> dict:
    tag_check(); PL = placebo_streams(ST); rows = []; sel2 = {}
    base_nets = np.load(f"{OUT}/retest_base_farm_nets.npy"); base_q = np.percentile(base_nets, [25, 50])
    for g in ST:
        cell = chosen.get(g) or dict(a1=BASE_CELL[0], t1=BASE_CELL[1], a2=BASE_CELL[2], R=BASE_CELL[3])
        base_real = T.AcctEval(ST[g], N_DEV, cell["R"]).run(cell["a1"], cell["t1"], cell["a2"])
        base_plc = T.AcctEval(PL[g], N_DEV, cell["R"]).run(cell["a1"], cell["t1"], cell["a2"])
        rb, hb, rbp = T.summ(base_real)["rate"], T.halves_rate(base_real), T.summ(base_plc)["rate"]
        for kind, thr in FILTERS:
            mk = T.skip_mask(F, kind, thr)
            fr = T.AcctEval(T.filtered(ST[g], mk), N_DEV, cell["R"]).run(cell["a1"], cell["t1"], cell["a2"]); fp = T.AcctEval(T.filtered(PL[g], mk), N_DEV, cell["R"]).run(cell["a1"], cell["t1"], cell["a2"])
            s = T.summ(fr); h = T.halves_rate(fr)
            rows.append(dict(group=g, filter=kind, thr=thr, skipped_share=float(mk[:N_DEV].mean()), rate=s["rate"], gain=s["rate"] - rb, se=T.boot_rate_se(fr, base_real), gain_h0=h[0] - hb[0], gain_h1=h[1] - hb[1], placebo_gain=T.summ(fp)["rate"] - rbp,
                             paid_share=s["paid_share"], days_med=s["days_med"], within_90=s["within_90"], pass_rate=s["pass_rate"]))
    S2 = pd.DataFrame(rows); S2.to_csv(f"{OUT}/retest_s2.csv", index=False); print("\nS2 filters"); print(S2.round(3).to_string(index=False))
    for g in ST:
        sg = S2[S2.group == g]; allowance = float(sg.placebo_gain.max())
        cell = chosen.get(g) or dict(a1=BASE_CELL[0], t1=BASE_CELL[1], a2=BASE_CELL[2], R=BASE_CELL[3])
        ok = sg[(sg.gain > 2 * sg.se) & (sg.gain > allowance) & (sg.gain_h0 > 0) & (sg.gain_h1 > 0)].sort_values("gain", ascending=False)
        print(f"\nS2 {g}: filter placebo allowance ${allowance:,.0f}; survivors {len(ok)}")
        sel2[g] = None
        for _, c in ok.head(3).iterrows():
            ST2 = dict(ST); ST2[g] = T.filtered(ST[g], T.skip_mask(F, c["filter"], c.thr))
            groups = [(gg, n, (dict(a1=(chosen.get(gg) or {}).get("a1", BASE_CELL[0]), t1=(chosen.get(gg) or {}).get("t1", BASE_CELL[1]), a2=(chosen.get(gg) or {}).get("a2", BASE_CELL[2]), R=(chosen.get(gg) or {}).get("R", BASE_CELL[3])))) for gg, n in (("noise", 3), ("gap", 2))]
            fs = T.farm_stats(T.farm_plans(ST2, groups), REF, N_DEV, seen=False)
            why = []
            if not fs["median"] >= base_q[1]: why.append("lower farm median")
            if not fs["p25"] >= base_q[0]: why.append("lower farm p25")
            if not fs["worst"] >= (bf["worst"] * 1.25 if bf["worst"] < 0 else bf["worst"] * 0.75): why.append("deeper worst farm")
            if not fs["paid_share_farm"] >= bf["paid_share_farm"] - 0.05: why.append("fewer accounts paid")
            print(f"   {c['filter']} {c.thr}: rate gain ${c.gain:.2f}/session; farm mean {fs['mean']:,.0f} median {fs['median']:,.0f} p25 {fs['p25']:,.0f} worst {fs['worst']:,.0f} paid {fs['paid_share_farm']:.0%} -> {why or 'PASS'}", flush=True)
            if not why and sel2[g] is None:
                sel2[g] = dict(filter=c["filter"], thr=float(c.thr), ev_gain=float(c.gain), farm_p25=float(fs["p25"]), farm_median=float(fs["median"]), farm_mean=float(fs["mean"]))
        print(f"S2 {g} selected:", sel2[g] or "none")
    json.dump(sel2, open(f"{OUT}/retest_s2_selected.json", "w"))
    return sel2


def final_streams(sel2: dict) -> dict:
    ST2 = dict(ST)
    for g, f in (sel2 or {}).items():
        if f:
            ST2[g] = T.filtered(ST[g], T.skip_mask(F, f["filter"], f["thr"]))
    return ST2


def s3_stage(chosen: dict, sel2: dict, bf: dict) -> None:
    tag_check(); ST2 = final_streams(sel2); rows = []
    prm = lambda g: dict(a1=(chosen.get(g) or {}).get("a1", BASE_CELL[0]), t1=(chosen.get(g) or {}).get("t1", BASE_CELL[1]), a2=(chosen.get(g) or {}).get("a2", BASE_CELL[2]), R=(chosen.get(g) or {}).get("R", BASE_CELL[3]))
    for lab, (kn, kg) in {"noise x5": (5, 0), "4 + 1": (4, 1), "3 + 2 (baseline split)": (3, 2), "2 + 3": (2, 3), "gap x5": (0, 5)}.items():
        groups = [(g, k, prm(g)) for g, k in (("noise", kn), ("gap", kg)) if k > 0]
        fps = T.farm_plans(ST2, groups)
        fs = T.farm_stats(fps, REF, N_DEV, seen=True)
        rows.append(dict(split=lab, mean=fs["mean"], median=fs["median"], p10=fs["p10"], p25=fs["p25"], p75=fs["p75"], p90=fs["p90"], share_pos=fs["share_pos"], worst=fs["worst"], paid_share_farm=fs["paid_share_farm"], funded_total=fs["funded_total"], days_first_payout=fs["days_first_payout"], farms_paid_in_window=fs["farms_paid_in_window"], seen=str(fs["seen"])))
        np.save(f"{OUT}/retest_s3_nets_{lab.split()[0].replace('+','p')}_{kn}{kg}.npy", fs["nets"])
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/retest_s3.csv", index=False)


if __name__ == "__main__":
    bf = None
    if "base" in STAGES:
        bf = base_stage()
    if "s1" in STAGES:
        s1_stage()
    if "pick" in STAGES or "s2" in STAGES or "s3" in STAGES:
        bf = bf or load_base()
        R = pd.read_csv(f"{OUT}/retest_s1_real.csv"); P = pd.read_csv(f"{OUT}/retest_s1_placebo.csv")
        chosen = pick_s1(R, P, bf) if ("pick" in STAGES) else json.load(open(f"{OUT}/retest_s1_selected.json"))
        sel2 = s2_stage(chosen, bf) if "s2" in STAGES else (json.load(open(f"{OUT}/retest_s2_selected.json")) if os.path.exists(f"{OUT}/retest_s2_selected.json") else {})
        if "s3" in STAGES:
            s3_stage(chosen, sel2, bf)
