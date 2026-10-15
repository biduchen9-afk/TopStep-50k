"""Pre-registration 9, Part A: best account split (one sleeve per group of accounts) for each firm. One Combine open at a time within the firm."""
from __future__ import annotations
import pickle, sys
from dataclasses import replace
import numpy as np, pandas as pd
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import FundedPlan
from risk_units import with_risk_unit
from run_firm_comparison import mkplan

sys.argv = sys.argv[:1]
from run_monthly_diversified import group_stream

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 70)
S = specs(); best = pickle.load(open("results/firm_best_configs.pkl", "rb"))
CFG = {"Topstep 150K": dict(R=9600.0, a1=1.2, a2=0.8, t1=2000.0), "FundedNext Flex 150K": {k: float(best["FundedNext Flex 150K"][k]) for k in ("R", "a1", "a2", "t1")},
       "FTMO Growth 150K": {k: float(best["FTMO Growth 150K"][k]) for k in ("R", "a1", "a2", "t1")}}
SPLITS5 = {"same basket x5": [("basket", 5)], "3 noise + 2 gap": [("noise", 3), ("gap", 2)], "2 noise + 1 VWAP + 2 gap": [("noise", 2), ("vwap", 1), ("gap", 2)],
           "2 noise + 3 gap": [("noise", 2), ("gap", 3)], "5 gap": [("gap", 5)], "5 noise": [("noise", 5)]}
SPLITS3 = {"same basket x3": [("basket", 3)], "2 noise + 1 gap": [("noise", 2), ("gap", 1)], "1 noise + 2 gap": [("noise", 1), ("gap", 2)], "3 gap": [("gap", 3)], "3 noise": [("noise", 3)],
           "1 noise + 1 VWAP + 1 gap": [("noise", 1), ("vwap", 1), ("gap", 1)]}


def main() -> None:
    gs = lambda names: with_risk_unit(group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "vwap": gs(["nq_vwap"]), "gap": gs(["nq_gap30"]), "basket": gs(["nq_noise", "nq_vwap", "nq_gap30"])}
    ref = ST["basket"]; d = pd.DatetimeIndex(ref.dates); n = len(d); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt)))
    rows = []; chosen = {}
    for firm, cfg in CFG.items():
        meta = S[firm]; slots = meta["spec"].max_funded; splits = SPLITS5 if slots == 5 else SPLITS3
        pol = Policy(reserve_cycles=None, global_one_eval=True)
        for lab, groups in splits.items():
            fps = [FirmPlan(replace(meta["spec"], name=f"{g}", max_funded=k, max_evals=1), [ST[g]], cfg["R"], mkplan(meta, cfg["t1"], cfg["a1"], cfg["a2"]), p_est=1.0, refill="all") for g, k in groups]
            nets = np.array([run_multifirm(fps, pol, idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["net"] for y in range(2011, 2021)])
            seen = [round(run_multifirm(fps, pol, idx(x))["net"]) for x in ("2021-01-04", "2023-01-02", "2025-01-02", "2025-07-01")]
            full = run_multifirm(fps, pol, int(np.flatnonzero(ref.ok)[0]), log=True)
            ev = pd.DataFrame(full["events"], columns=["t", "g", "kind", "amt"]); ev["m"] = d[np.minimum(ev.t.to_numpy(), n - 1)].to_period("M")
            pay_m = ev[ev.kind == "payout"].groupby("m").amt.sum().reindex(pd.period_range("2012-01", "2026-06", freq="M")).fillna(0.0)
            rows.append(dict(firm=firm, split=lab, dev_mean=nets.mean(), dev_p25=np.percentile(nets, 25), dev_worst=nets.min(), green=int((nets > 0).sum()), seen=str(seen), full_net=full["net"],
                             full_fees=full["fees"], months_with_payout=float((pay_m > 0).mean())))
    R = pd.DataFrame(rows); R.to_csv("results/firm_splits.csv", index=False)
    print(R.round(0).to_string(index=False))
    print("\nBEST SPLIT PER FIRM (rule: highest dev p25 among splits with mean > 0):")
    for firm in CFG:
        sub = R[(R.firm == firm) & (R.dev_mean > 0)].sort_values(["dev_p25", "dev_mean"], ascending=False)
        b = sub.iloc[0] if len(sub) else R[R.firm == firm].sort_values("dev_p25", ascending=False).iloc[0]
        print(f"  {firm:<22} {b.split:<28} dev mean {b.dev_mean:8,.0f}  p25 {b.dev_p25:8,.0f}  worst {b.dev_worst:8,.0f}  full {b.full_net:9,.0f}  payout months {b.months_with_payout:.0%}")
        chosen[firm] = b.split
    pickle.dump(chosen, open("results/firm_split_choice.pkl", "wb"))


if __name__ == "__main__":
    main()
