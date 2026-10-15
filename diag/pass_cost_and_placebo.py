"""(1) Combine fees per funded account (the cost-effective pass rate) by sleeve and R, dev starts. (2) Edge-removed placebo for the account split vs the same-basket farm."""
import sys
from dataclasses import replace
import numpy as np, pandas as pd
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import FundedPlan, Stream, rules_from_config, run_combine
from pricing_config import load_config
from risk_units import with_risk_unit
from final_strategy import PATH
sys.argv = sys.argv[:1]
from run_monthly_diversified import group_stream
pd.set_option("display.width", 250, "display.max_columns", 40)
TS = specs()["Topstep 150K"]["spec"]; rules = rules_from_config(load_config(), PATH); fee = TS.eval_fee
BAL = FundedPlan("p", "x", 2000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8)
gs = lambda names: with_risk_unit(group_stream(names), "atr", 40)
S = {"noise-area": gs(["nq_noise"]), "VWAP+ATR": gs(["nq_vwap"]), "gap-VWAP": gs(["nq_gap30"]), "NQ basket": gs(["nq_noise", "nq_vwap", "nq_gap30"])}
ref = S["NQ basket"]; d = pd.DatetimeIndex(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right")); first = int(np.flatnonzero(ref.ok)[0]) + 260
idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt)))
rows = []
for name, s in S.items():
    x, lo = s.pnl.tolist(), s.low.tolist()
    for R in (3200., 4800., 6400., 9600., 12800.):
        nd = s.desired_micros(R); res = [run_combine(x[:n_dev], lo[:n_dev], nd[:n_dev], i, rules) for i in range(first, n_dev - 300, 3)]
        bills = np.mean([1 + (max(t, 1) - 1) // 21 for _, t in res]); pr = np.mean([st == 1 for st, _ in res])
        rows.append(dict(sleeve=name, R=R, pass_rate=pr, bills_per_attempt=bills, fees_per_funded=fee * bills / pr if pr else np.nan))
T = pd.DataFrame(rows); T.to_csv("results/pass_cost_by_sleeve.csv", index=False)
print("COMBINE FEES PER FUNDED ACCOUNT (dev starts; $199 per 21 sessions, no activation fee):"); print(T.pivot_table(index="sleeve", columns="R", values="fees_per_funded").round(0).to_string())
print("\nbest R per sleeve by fees per funded account:"); print(T.sort_values("fees_per_funded").groupby("sleeve").head(1)[["sleeve", "R", "pass_rate", "fees_per_funded"]].round(2).to_string(index=False))
# placebo
def null(s):
    ok = (s.dates <= DEV_END) & s.ok; mu = float(np.mean(s.pnl[ok] / s.atr_usd[ok])); dm = np.where(s.ok, s.pnl - mu * np.nan_to_num(s.atr_usd), s.pnl)
    return Stream(s.name, s.dates, dm, np.minimum(s.low - (s.pnl - dm), 0.0), s.atr_usd, s.ok)
def fp(name, s, slots): return FirmPlan(replace(TS, name=name, max_funded=slots, max_evals=1), [s], 9600.0, BAL, p_est=1.0, refill="all")
def cases(m):
    return {"same basket x5": [fp("b", m["NQ basket"], 5)], "split 3 noise-area / 2 gap-VWAP": [fp("n", m["noise-area"], 3), fp("g", m["gap-VWAP"], 2)],
            "split 2 noise / 1 VWAP / 2 gap": [fp("n", m["noise-area"], 2), fp("v", m["VWAP+ATR"], 1), fp("g", m["gap-VWAP"], 2)]}
print("\nPLACEBO (dev mean of ten 3-year farms; one Combine at a time):")
for tag, m in (("strategy", S), ("edge removed", {k: null(v) for k, v in S.items()})):
    for lab, fps in cases(m).items():
        nets = np.array([run_multifirm(fps, Policy(reserve_cycles=None, global_one_eval=True), idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["net"] for y in range(2011, 2021)])
        print(f"  {tag:<13} {lab:<34} mean {nets.mean():9,.0f}  green {int((nets > 0).sum())}/10  worst {nets.min():9,.0f}")
