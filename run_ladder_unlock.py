"""Exploratory (dev-selected): how much cash must the self-funded farm hold before a second/third firm is added? Same ladder as run_multi_firm_ladder.py."""
from __future__ import annotations
import pickle, sys
import numpy as np, pandas as pd
from firm_specs import specs
from multi_firm import Policy, run_multifirm
from portfolio import DEV_END
from risk_units import with_risk_unit
from run_multi_firm_ladder import NQ3, build_firm_plans
sys.argv = sys.argv[:1]
from run_monthly_diversified import group_stream

pd.set_option("display.width", 250, "display.max_columns", 40)


def main() -> None:
    S = specs(); C = pickle.load(open("results/multi_firm_configs.pkl", "rb")); order = C["order"]
    A = with_risk_unit(group_stream(NQ3), "atr", 40)
    groups = {"NQ trend": with_risk_unit(group_stream(["nq_noise", "nq_vwap"]), "atr", 40), "NQ gap-VWAP": with_risk_unit(group_stream(["nq_gap30"]), "atr", 40),
              "ES+GC": with_risk_unit(group_stream(["es_noise", "es_vwap_t250", "gc_noise_t60"]), "atr", 40)}
    gl = list(groups); streams_B = {n: groups[gl[i % 3]] for i, n in enumerate(order)}
    n_dev = int(np.searchsorted(A.dates, DEV_END, side="right")); d = pd.DatetimeIndex(A.dates)
    idx = lambda dt: int(np.searchsorted(A.dates, np.datetime64(dt)))
    cases = {"A same basket": build_firm_plans(order, {n: A for n in order}, C["cfg_A"], S), "B different sleeves": build_firm_plans(order, streams_B, C["cfg_B"], S)}
    rows = []
    for mix, fps in cases.items():
        for uc in (0.0, 3000.0, 6000.0, 10000.0, 15000.0, None):            # None = Topstep only
            fl = fps[:1] if uc is None else fps; pol = Policy(k0=1, reserve_cycles=3.0, unlock_cash=0.0 if uc is None else uc)
            for kind, runs in (("dev long (to 2022-12)", [(y, n_dev) for y in (2012, 2014, 2016, 2018)]), ("dev 3y", [(y, None) for y in range(2012, 2021)]), ("seen to end", [(y, len(A.dates)) for y in (2021, 2023, 2025)])):
                for y, ne in runs:
                    i0 = idx(f"{y}-01-01"); ne2 = min(i0 + 756, n_dev) if kind == "dev 3y" else ne
                    r = run_multifirm(fl, pol, i0, ne2)
                    rows.append(dict(mix=mix, unlock="Topstep only" if uc is None else int(uc), kind=kind, start=y, net=r["net"], oop=r["oop_total"], ruined=r["ruined_t"] is not None,
                                     firms=sum(v["first_buy_t"] is not None for v in r["firms"].values()), min_cash=r["min_cash"]))
    R = pd.DataFrame(rows); R.to_csv("results/ladder_unlock.csv", index=False)
    for mix in cases:
        print(f"\n=== {mix} ===")
        g = R[R.mix == mix].groupby(["kind", "unlock"]).agg(mean_net=("net", "mean"), min_net=("net", "min"), green=("net", lambda v: int((v > 0).sum())), n=("net", "size"), ruined=("ruined", "sum"),
                                                          oop_max=("oop", "max"), firms_max=("firms", "max"), worst_cash=("min_cash", "min"))
        print(g.round(0).to_string())
        print("seen starts net:"); print(R[(R.mix == mix) & (R.kind == "seen to end")].pivot_table(index="unlock", columns="start", values="net").round(0).to_string())


if __name__ == "__main__":
    main()
