"""Pre-registration 9, Part B: the full scaling pipeline  Topstep (3 noise-area + 2 gap-VWAP accounts) -> FundedNext -> FTMO Growth, self-funded, one Combine open at a time."""
from __future__ import annotations
import pickle, sys
from dataclasses import replace
import numpy as np, pandas as pd
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from risk_units import with_risk_unit
from run_firm_comparison import mkplan
from run_firm_splits import CFG, S, SPLITS3, SPLITS5

sys.argv = sys.argv[:1]
from run_monthly_diversified import group_stream

pd.set_option("display.width", 250, "display.max_columns", 40)
THRESHOLDS = (3000.0, 6000.0, 10000.0)


def build(ST, choice, firms=("Topstep 150K", "FundedNext Flex 150K", "FTMO Growth 150K")):
    fps = []
    for firm in firms:
        meta, cfg = S[firm], CFG[firm]
        if firm == "Topstep 150K":
            groups = [("noise", 3), ("gap", 2)]
        else:
            groups = (SPLITS5 if meta["spec"].max_funded == 5 else SPLITS3)[choice[firm]]
        short = firm.split()[0]
        for g, k in groups:
            fps.append(FirmPlan(replace(meta["spec"], name=f"{short}:{g}", max_funded=k, max_evals=1), [ST[g]], cfg["R"], mkplan(meta, cfg["t1"], cfg["a1"], cfg["a2"]), p_est=1.0, refill="all", firm=short))
    return fps


def main() -> None:
    gs = lambda names: with_risk_unit(group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "vwap": gs(["nq_vwap"]), "gap": gs(["nq_gap30"]), "basket": gs(["nq_noise", "nq_vwap", "nq_gap30"])}
    choice = pickle.load(open("results/firm_split_choice.pkl", "rb")); print("splits used:", {k: v for k, v in choice.items() if k != "Topstep 150K"}, "| Topstep: 3 noise + 2 gap")
    ref = ST["basket"]; d = pd.DatetimeIndex(ref.dates); n = len(d); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt)))
    cases = {"Topstep only": build(ST, choice, ("Topstep 150K",)), "Topstep + FundedNext": build(ST, choice, ("Topstep 150K", "FundedNext Flex 150K")), "Topstep + FundedNext + FTMO": build(ST, choice)}
    rows = []
    for lab, fps in cases.items():
        for th in ((None,) if lab == "Topstep only" else THRESHOLDS):
            pol = Policy(k0=1, reserve_cycles=3.0, global_one_eval=True, group_aware=True, unlock_cash=th or 0.0)
            for kind, runs in (("dev long", [(y, n_dev) for y in (2012, 2014, 2016, 2018)]), ("dev 3y", [(y, None) for y in range(2012, 2021)]), ("seen", [(y, n) for y in (2021, 2023, 2025)]), ("full", [(2010, n)])):
                for y, ne in runs:
                    i0 = idx(f"{y}-01-01") if kind != "full" else int(np.flatnonzero(ref.ok)[0])
                    ne2 = min(i0 + 756, n_dev) if kind == "dev 3y" else ne
                    r = run_multifirm(fps, pol, i0, ne2)
                    fp_t = [v["first_payout_t"] for v in r["firms"].values() if v["first_payout_t"] is not None]; fb = min(v["first_buy_t"] for v in r["firms"].values() if v["first_buy_t"] is not None)
                    fam = {}
                    for k_, v in r["firms"].items():
                        fam[k_.split(":")[0]] = fam.get(k_.split(":")[0], 0.0) + v["net"]
                    rows.append(dict(case=lab, unlock=th, kind=kind, start=y, net=r["net"], oop=r["oop_total"], ruined=r["ruined_t"] is not None, min_cash=r["min_cash"], end_cash=r["end_cash"],
                                     days_first_payout=(d[min(min(fp_t), n - 1)] - d[fb]).days if fp_t else np.nan, nets=str({k: round(v) for k, v in fam.items()}),
                                     firms_started=len({k.split(':')[0] for k, v in r["firms"].items() if v["first_buy_t"] is not None})))
    R = pd.DataFrame(rows); R.to_csv("results/scaling_pipeline.csv", index=False)
    g = R.groupby(["case", "unlock", "kind"], dropna=False).agg(mean_net=("net", "mean"), worst=("net", "min"), green=("net", lambda v: int((v > 0).sum())), runs=("net", "size"), ruined=("ruined", "sum"), oop_max=("oop", "max"), firms_max=("firms_started", "max"))
    print(g.round(0).to_string())
    print("\nSEEN starts (net to end of data) and the full history:"); print(R[R.kind.isin(["seen", "full"])].pivot_table(index=["case", "unlock"], columns=["kind", "start"], values="net", dropna=False).round(0).to_string())
    print("\nfull history detail:"); print(R[R.kind == "full"][["case", "unlock", "net", "oop", "min_cash", "end_cash", "days_first_payout", "ruined", "nets"]].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
