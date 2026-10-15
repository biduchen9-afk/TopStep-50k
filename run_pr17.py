"""Pre-registration 17. python run_pr17.py A | B | C | D | all   (results/pr17_*.csv|json)"""
from __future__ import annotations

import sys
STAGES = sys.argv[1:] or ["all"]
import dataclasses
import json
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pandas as pd

import retest as T
import retest_cost as C
from multi_firm import Policy, run_multifirm
from real_engine import Stream
from scaling_common import placebo_streams

pd.set_option("display.width", 250, "display.max_columns", 60)
OUT = "results"
D = T.load(); ST0, REF, N, N_DEV = D["ST"], D["ref"], D["n"], D["n_dev"]
BASE = T.BASE
POL = dict(reserve_cycles=None, global_one_eval=True)
SPLITS = {"3+2": (3, 2), "2+3": (2, 3), "gap x5": (0, 5)}
WIN = T.WINDOW


@contextmanager
def spec_variant(ticks=None, payout_path=None, eval_fee=None, activation_fee=None, ladder=None, max_payouts=None):
    """Temporarily replace the Topstep spec in the shared specs dict (retest.S is run_kelly.S)."""
    old = dict(T.S)
    sp = T.S["spec"]
    if payout_path is not None:
        from firm_engine import topstep_150k
        sp = topstep_150k("no_activation_dll", payout_path)
    f, ph = sp.funded, sp.eval_phases
    if ticks is not None:
        f = replace(f, dll_slip_ticks=ticks); ph = tuple(replace(p, dll_slip_ticks=ticks) for p in ph)
    if ladder is not None:
        f = replace(f, ladder=ladder)
    sp = replace(sp, funded=f, eval_phases=ph, **({} if eval_fee is None else dict(eval_fee=eval_fee)), **({} if activation_fee is None else dict(activation_fee=activation_fee)))
    T.S["spec"] = sp
    if max_payouts is not None:
        T.S["max_payouts"] = max_payouts
    try:
        yield
    finally:
        T.S.clear(); T.S.update(old)


def farm_runs(ST, split=(3, 2), cells=None, starts=None, n_end_fn=None, cohort=False, ref=None, n_dev=None, fee_per_payout=0.0):
    """Per-farm outcomes for the 3-year window (cohort=False) or cohort accounting (cohort=True)."""
    ref = REF if ref is None else ref; n_dev = N_DEV if n_dev is None else n_dev
    cells = cells or {"noise": BASE, "gap": BASE}
    groups = [(g, k, cells[g]) for g, k in (("noise", split[0]), ("gap", split[1])) if k > 0]
    fps = T.farm_plans(ST, groups)
    if starts is None:
        starts = T.farm_starts(ref, n_dev)
        if cohort:
            starts = [i for i in starts if i + WIN + 252 <= n_dev]
    rows = []
    dts = pd.DatetimeIndex(ref.dates)
    for i0 in starts:
        if cohort:
            r = run_multifirm(fps, Policy(stop_buy_at=i0 + WIN, **POL), i0, n_dev)
        else:
            r = run_multifirm(fps, Policy(**POL), i0, i0 + WIN)
        npay = sum(v["n_payouts"] for v in r["firms"].values())
        fb = [v["first_buy_t"] for v in r["firms"].values() if v["first_buy_t"] is not None]; fp_ = [v["first_payout_t"] for v in r["firms"].values() if v["first_payout_t"] is not None]
        d1 = (dts[min(min(fp_), len(dts) - 1)] - dts[min(fb)]).days if (fb and fp_) else np.nan
        rows.append(dict(start=i0, year=int(dts[i0].year), net=r["net"] - fee_per_payout * npay, npay=npay, min_cash=r["min_cash"], d1=d1,
                         funded=sum(v["n_funded"] for v in r["firms"].values()), paid=sum(v["n_paid_accts"] for v in r["firms"].values())))
    return pd.DataFrame(rows)


def q(x):
    x = np.asarray(x, float)
    return dict(mean=float(np.mean(x)), median=float(np.median(x)), p10=float(np.percentile(x, 10)), p25=float(np.percentile(x, 25)), share_pos=float((x > 0).mean()), worst=float(x.min()))


def full_eval(ST, split=(3, 2), label="", fee_per_payout=0.0):
    ac = {}
    for g in ("noise", "gap"):
        ae = T.AcctEval(ST[g], N_DEV, BASE["R"]); df = ae.run(BASE["a1"], BASE["t1"], BASE["a2"]); ac[g] = T.summ(df)
    bf = json.load(open(f"{OUT}/retest_base_farm.json")); ts = T.time_shares(bf["w_start"], {g: ac[g]["mean_tau"] for g in ac}); wf = bf["w_fund"]
    mix_rate = sum(ts[g] * ac[g]["rate"] for g in ac); mix_paid = sum(wf[g] * ac[g]["paid_share"] for g in ac)
    F = farm_runs(ST, split, fee_per_payout=fee_per_payout); Co = farm_runs(ST, split, cohort=True, fee_per_payout=fee_per_payout)
    r = dict(label=label, noise_rate=ac["noise"]["rate"], gap_rate=ac["gap"]["rate"], mix_rate=mix_rate, mix_paid=mix_paid)
    r.update({f"farm_{k}": v for k, v in q(F.net).items()}); r.update({f"cohort_{k}": v for k, v in q(Co.net).items()})
    r.update(farm_paid=F.paid.sum() / max(F.funded.sum(), 1), days_first_payout=float(np.nanmedian(F.d1)), peak_cash_med=float(-np.median(F.min_cash)), peak_cash_p90=float(-np.percentile(F.min_cash, 10)),
             p_cash_2k=float((F.min_cash <= -2000).mean()), p_cash_5k=float((F.min_cash <= -5000).mean()), p_cash_10k=float((F.min_cash <= -10000).mean()), n_payouts=float(F.npay.mean()))
    return r, F, Co


def stage_A():
    rows = []
    for lab, ST in (("as modelled (dev costs)", ST0), ("re-based, 1 tick", C.rebase(ST0, 1.0)), ("re-based, 2 ticks", C.rebase(ST0, 2.0)), ("re-based, 3 ticks", C.rebase(ST0, 3.0))):
        r, F, Co = full_eval(ST, label=lab); rows.append(r); print(lab, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("mix_rate", "mix_paid", "farm_mean", "farm_median", "farm_p25", "cohort_mean", "cohort_median", "cohort_p25", "days_first_payout", "peak_cash_p90")}, flush=True)
        if lab == "re-based, 1 tick":
            F.to_csv(f"{OUT}/pr17_A_farms_rebased1.csv", index=False); Co.to_csv(f"{OUT}/pr17_A_cohort_rebased1.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{OUT}/pr17_A.csv", index=False)


def stage_B():
    rows = []
    for basis, ST in (("re-based, 1 tick cost", C.rebase(ST0, 1.0)), ("dev costs", ST0)):
        for tk in (0, 1, 2, 3, 4):
            with spec_variant(ticks=float(tk)):
                r, _, _ = full_eval(ST, label=f"{basis} | overshoot {tk} ticks")
            r["basis"], r["overshoot_ticks"] = basis, tk; rows.append(r)
            print(r["label"], round(r["mix_rate"], 1), round(r["farm_mean"]), round(r["farm_median"]), round(r["cohort_median"]), flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/pr17_B.csv", index=False)


def stage_C():
    ST = C.rebase(ST0, 1.0); rows = []
    lad = ((0.0, 3), (3000.0, 10), (4500.0, 15))
    scen = [("reference (re-based, 1 tick)", dict(), 0.0, ST), ("(1) all accounts on the Consistency path", dict(payout_path="consistency"), 0.0, ST), ("(2a) payouts capped at 5 per account", dict(max_payouts=5), 0.0, ST),
            ("(2b) payouts capped at 3 per account", dict(max_payouts=3), 0.0, ST), ("(3) $30 fee on every payout", dict(), 30.0, ST), ("(4) Standard-path price ($199 + $149 per XFA)", dict(activation_fee=149.0), 0.0, ST),
            ("(5) XFA ladder flat at 30 micros to $3,000", dict(ladder=lad), 0.0, ST),
            ("(6) combined adverse: 2 ticks cost, 2 ticks overshoot, $30 per payout, 5 payouts", dict(ticks=2.0, max_payouts=5), 30.0, C.rebase(ST0, 2.0))]
    for lab, kw, fee, S_ in scen:
        with spec_variant(**kw):
            r, _, _ = full_eval(S_, label=lab, fee_per_payout=fee)
        rows.append(r); print(lab, round(r["mix_rate"], 1), round(r["farm_mean"]), round(r["farm_median"]), round(r["farm_p25"]), round(r["cohort_median"]), flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/pr17_C.csv", index=False)


# ------------------------------------------------------------------ Part D
YEARS = list(range(2011, 2023))
DATES = pd.DatetimeIndex(REF.dates)


def synth(ST, draw):
    """Concatenate whole calendar years of the dev streams in the order `draw` (same draw for both sleeves)."""
    idx = np.concatenate([np.flatnonzero(DATES.year == y) for y in draw])
    n = len(idx); dates = pd.bdate_range("2011-01-03", periods=n).values.astype("datetime64[D]")
    out = {}
    for g, s in ST.items():
        out[g] = Stream(s.name + "|synth", dates, s.pnl[idx], s.low[idx], s.atr_usd[idx], s.ok[idx], None if s.fx is None else s.fx[idx])
    return out


def split_metrics(SY, split, step=20):
    ref = SY["noise"]; n = len(ref.dates); first = int(np.flatnonzero(ref.ok)[0])
    starts = list(range(first, n - WIN, step)); coh = [i for i in starts if i + WIN + 252 <= n]
    f = farm_runs(SY, split, starts=starts, ref=ref, n_dev=n); c = farm_runs(SY, split, starts=coh, cohort=True, ref=ref, n_dev=n)
    return dict(farm=q(f.net), cohort=q(c.net))


def one_draw(args):
    seed, null = args
    rng = np.random.default_rng(seed); draw = rng.choice(YEARS, size=len(YEARS), replace=True)
    ST = C.rebase(ST0, 1.0)
    if null:
        ST = {**ST, **placebo_streams(ST)}
    SY = synth(ST, draw)
    return {lab: split_metrics(SY, sp) for lab, sp in SPLITS.items()}


def stage_D():
    # (i) real dev farms, paired by start year
    ST = C.rebase(ST0, 1.0); real = {}; rows = []
    for lab, sp in SPLITS.items():
        F = farm_runs(ST, sp); Co = farm_runs(ST, sp, cohort=True); real[lab] = (F, Co)
        rows.append(dict(split=lab, **{f"farm_{k}": v for k, v in q(F.net).items()}, **{f"cohort_{k}": v for k, v in q(Co.net).items()}, farm_paid=F.paid.sum() / max(F.funded.sum(), 1), days_first_payout=float(np.nanmedian(F.d1))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/pr17_D_real.csv", index=False)
    # paired year-block bootstrap of real differences against 3+2 (cohort and farm)
    rng = np.random.default_rng(17); res = {}
    for lab in ("2+3", "gap x5"):
        for kind, ix in (("farm", 0), ("cohort", 1)):
            a, b = real[lab][ix], real["3+2"][ix]; diff = a.net.values - b.net.values; yrs = a.year.values; ys = np.unique(yrs)
            boots = []
            for _ in range(4000):
                pick = rng.choice(ys, len(ys)); idx = np.concatenate([np.flatnonzero(yrs == y) for y in pick]); x, y_ = a.net.values[idx], b.net.values[idx]
                boots.append((x.mean() - y_.mean(), np.median(x) - np.median(y_), np.percentile(x, 25) - np.percentile(y_, 25)))
            bo = np.array(boots); res[f"{lab}|{kind}"] = dict(mean=float(diff.mean()), median=float(np.median(a.net) - np.median(b.net)), p25=float(np.percentile(a.net, 25) - np.percentile(b.net, 25)),
                                                          ci_mean=[float(np.percentile(bo[:, 0], 5)), float(np.percentile(bo[:, 0], 95))], ci_median=[float(np.percentile(bo[:, 1], 5)), float(np.percentile(bo[:, 1], 95))],
                                                          ci_p25=[float(np.percentile(bo[:, 2], 5)), float(np.percentile(bo[:, 2], 95))])
            # outside 2016-18 stratum (cohort only)
            if kind == "cohort":
                m = ~a.year.isin([2016, 2017, 2018]).values
                res[f"{lab}|cohort_outside_2016_18"] = dict(median_diff=float(np.median(a.net[m]) - np.median(b.net[m])), mean_diff=float(a.net[m].mean() - b.net[m].mean()), n=int(m.sum()))
    # fill-stress worst case on real cohort medians
    worst = {}
    for lab, sp in SPLITS.items():
        meds = []
        for tk in (0, 2, 4):
            with spec_variant(ticks=float(tk)):
                meds.append(float(np.median(farm_runs(ST, sp, cohort=True).net)))
        worst[lab] = dict(by_ticks=meds, worst=min(meds))
    # (ii) and (iii): regime-resampled bootstrap, real and zero-edge
    B = 200
    with ProcessPoolExecutor(max_workers=4) as ex:
        R_real = list(ex.map(one_draw, [(1000 + i, False) for i in range(B)]))
        R_null = list(ex.map(one_draw, [(5000 + i, True) for i in range(B)]))
    boot = {}
    for lab in ("2+3", "gap x5"):
        for tag, R in (("real", R_real), ("null", R_null)):
            d = {}
            for kind in ("farm", "cohort"):
                for k in ("mean", "median", "p25"):
                    diff = np.array([r[lab][kind][k] - r["3+2"][kind][k] for r in R])
                    d[f"{kind}_{k}"] = dict(p_beats=float((diff > 0).mean()), mean_gain=float(diff.mean()), p95=float(np.percentile(diff, 95)))
            boot[f"{lab}|{tag}"] = d
    json.dump(dict(paired=res, fill_worst=worst, boot=boot), open(f"{OUT}/pr17_D.json", "w"), indent=1)
    # decision
    out = {}
    for lab in ("2+3", "gap x5"):
        br, bn = boot[f"{lab}|real"], boot[f"{lab}|null"]
        gain_c = res[f"{lab}|cohort"]["mean"]; null95 = bn["cohort_mean"]["p95"]
        checks = {"P(beats) mean >= 0.975": br["farm_mean"]["p_beats"] >= 0.975, "P(beats) median >= 0.80": br["farm_median"]["p_beats"] >= 0.80, "P(beats) p25 >= 0.80": br["farm_p25"]["p_beats"] >= 0.80,
                  "cohort-mean gain >= $5k": gain_c >= 5000, "outside 2016-18 cohort median diff >= -$1k": res[f"{lab}|cohort_outside_2016_18"]["median_diff"] >= -1000,
                  "gain above null 95th percentile": gain_c > null95, "not worse in worst fill case": worst[lab]["worst"] >= worst["3+2"]["worst"]}
        out[lab] = dict(checks={k: bool(v) for k, v in checks.items()}, switch=bool(all(checks.values())), cohort_gain=gain_c, null95=null95)
        print(lab, out[lab], flush=True)
    json.dump(out, open(f"{OUT}/pr17_D_decision.json", "w"), indent=1)


if __name__ == "__main__":
    st = STAGES
    if "all" in st: st = ["A", "B", "C", "D"]
    for s in st:
        {"A": stage_A, "B": stage_B, "C": stage_C, "D": stage_D}[s]()
