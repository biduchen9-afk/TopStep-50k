"""Pre-registration 4, Part B: cash-gated ladder across firms. See docs/PREREGISTRATION_4.md."""
from __future__ import annotations
import itertools, pickle, sys
import numpy as np, pandas as pd
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import FundedPlan
from risk_units import with_risk_unit
from run_firm_comparison import A1S, A2S, RS, YEARS, farm as solo_farm, mkplan

sys.argv = sys.argv[:1]
from run_monthly_diversified import group_stream

pd.set_option("display.width", 250, "display.max_columns", 40)
NQ3 = ["nq_noise", "nq_vwap", "nq_gap30"]


def select(s, meta, n_dev, idx):
    best = None
    for R, a1, a2, t1 in itertools.product(RS, A1S, A2S, meta["t1_grid"]):
        pl = mkplan(meta, t1, a1, a2)
        nets = np.array([solo_farm(s, meta, R, pl, idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["net"] for y in YEARS])
        row = dict(R=R, a1=a1, a2=a2, t1=t1, mean=nets.mean(), p25=np.percentile(nets, 25))
        if row["mean"] > 0 and (best is None or (row["p25"], row["mean"]) > (best["p25"], best["mean"])):
            best = row
    return best


def build_firm_plans(order, streams_for, cfgs, S):
    fps = []
    for name in order:
        b = cfgs[name]; meta = S[name]
        fps.append(FirmPlan(meta["spec"], [streams_for[name]], float(b["R"]), mkplan(meta, float(b["t1"]), float(b["a1"]), float(b["a2"]))))
    return fps


def summarize(res, d, i0, label, extra=None):
    f0 = res["firms"]
    first_buy = min(v["first_buy_t"] for v in f0.values() if v["first_buy_t"] is not None)
    fp = [v["first_payout_t"] for v in f0.values() if v["first_payout_t"] is not None]
    r = dict(case=label, net=res["net"], payouts=res["payouts"], fees=res["fees"], oop=res["oop_total"], min_cash=res["min_cash"], shortfall_days=res["shortfall_days"],
             end_cash=res["end_cash"], ruined=res["ruined_t"] is not None, days_to_first_payout=(d[min(fp) ] - d[first_buy]).days if fp else np.nan, firms_started=sum(v["first_buy_t"] is not None for v in f0.values()))
    r.update({f"net[{k.split()[0]}]": v["net"] for k, v in f0.items()})
    if extra:
        r.update(extra)
    return r


def main() -> None:
    S = specs(); best_all = pickle.load(open("results/firm_best_configs.pkl", "rb")); comp = pd.read_csv("results/firm_comparison.csv")
    fam = {n: n.split()[0] for n in S}
    comp["family"] = comp.firm.map(fam)
    chosen = comp.sort_values("net_per_fee", ascending=False).groupby("family").head(1).sort_values("net_per_fee", ascending=False)
    order = ["Topstep 150K"] + [f for f in chosen.firm if f != "Topstep 150K"]
    print("ladder order (one variant per firm, by dev net per fee dollar):", order)
    A = with_risk_unit(group_stream(NQ3), "atr", 40)
    groups = {"NQ trend": with_risk_unit(group_stream(["nq_noise", "nq_vwap"]), "atr", 40), "NQ gap-VWAP": with_risk_unit(group_stream(["nq_gap30"]), "atr", 40),
              "ES+GC": with_risk_unit(group_stream(["es_noise", "es_vwap_t250", "gc_noise_t60"]), "atr", 40)}
    d = pd.DatetimeIndex(A.dates); n_dev = int(np.searchsorted(A.dates, DEV_END, side="right"))
    idx = lambda dt: int(np.searchsorted(A.dates, np.datetime64(dt)))
    glist = list(groups)
    streams_B = {name: groups[glist[i % len(glist)]] for i, name in enumerate(order)}
    cfg_A = {n: best_all[n] for n in order}
    cfg_B = {}
    for n in order:
        b = select(streams_B[n], S[n], n_dev, idx)
        cfg_B[n] = b if b is not None else best_all[n]
    pol = Policy(k0=1, reserve_cycles=3.0)
    cases = {"A same basket": (build_firm_plans(order, {n: A for n in order}, cfg_A, S)), "B different sleeves": build_firm_plans(order, streams_B, cfg_B, S)}
    rows = []
    for mix, fps in cases.items():
        for label, fpl in (("ladder (all firms)", fps), ("Topstep only, same cash rule", fps[:1])):
            for y in range(2012, 2021):
                i0 = idx(f"{y}-01-01"); res = run_multifirm(fpl, pol, i0, min(i0 + 756, n_dev))
                rows.append(dict(mix=mix, scope=label, start=y, kind="dev 3y", **summarize(res, d, i0, label)))
            for y in (2021, 2023, 2025):
                i0 = idx(f"{y}-01-01"); res = run_multifirm(fpl, pol, i0)
                rows.append(dict(mix=mix, scope=label, start=y, kind="seen to end", **summarize(res, d, i0, label)))
            i0 = int(np.flatnonzero(A.ok)[0]); res = run_multifirm(fpl, pol, i0)
            rows.append(dict(mix=mix, scope=label, start=2010, kind="full history", **summarize(res, d, i0, label)))
        unc = run_multifirm(fps[:1], Policy(reserve_cycles=None), int(np.flatnonzero(A.ok)[0]))
        rows.append(dict(mix=mix, scope="Topstep only, NO cash rule (old farm)", start=2010, kind="full history", **summarize(unc, d, 0, "old")))
    R = pd.DataFrame(rows); R.to_csv("results/multi_firm_ladder.csv", index=False)
    cols = ["mix", "scope", "start", "kind", "net", "payouts", "fees", "oop", "min_cash", "shortfall_days", "ruined", "days_to_first_payout", "firms_started"]
    for mix in cases:
        sub = R[R.mix == mix]
        print(f"\n===== {mix} =====")
        print(sub[sub.kind == "full history"][cols].round(0).to_string(index=False))
        print(sub[sub.kind == "seen to end"][cols].round(0).to_string(index=False))
        piv = sub[sub.kind == "dev 3y"].pivot_table(index="start", columns="scope", values="net").round(0)
        print("dev 3-year farms, net by start year:"); print(piv.to_string())
        sc = sub[(sub.kind == "dev 3y")].groupby("scope").agg(mean=("net", "mean"), green=("net", lambda v: int((v > 0).sum())), worst=("net", "min"), oop_med=("oop", "median"), oop_max=("oop", "max"), ruined=("ruined", "sum"))
        print(sc.round(0).to_string())
    pickle.dump(dict(order=order, cfg_A=cfg_A, cfg_B=cfg_B), open("results/multi_firm_configs.pkl", "wb"))
    print("\nconfigs B (reselected on each firm's assigned stream):"); print(pd.DataFrame(cfg_B).T.round(2).to_string())


if __name__ == "__main__":
    main()
