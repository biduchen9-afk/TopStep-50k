"""Pre-registration 5, Part B (+ parallelism): days from the first Combine to the first payout and out-of-pocket money, for different lead firms and for k0 parallel Combines before the first payout."""
import pickle
import numpy as np, pandas as pd
from final_strategy import build_stream
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import FundedPlan
from run_firm_comparison import mkplan
s = build_stream(); d = pd.DatetimeIndex(s.dates); n = len(d); n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
idx = lambda dt: int(np.searchsorted(s.dates, np.datetime64(dt)))
S = specs(); best = pickle.load(open("results/firm_best_configs.pkl", "rb"))
runs = [(y, min(idx(f"{y}-01-01") + 756, n_dev)) for y in range(2012, 2021)] + [(y, n) for y in (2021, 2023, 2025)]
def evaluate(name, plan, R, k0, label):
    meta = S[name]; rows = []
    for y, ne in runs:
        i0 = idx(f"{y}-01-01")
        r = run_multifirm([FirmPlan(meta["spec"], [s], R, plan)], Policy(k0=k0, reserve_cycles=3.0, single_pipeline=(k0 == 1)), i0, ne)
        v = r["firms"][name]; days = (d[min(v["first_payout_t"], n - 1)] - d[v["first_buy_t"]]).days if v["first_payout_t"] is not None else np.nan
        rows.append((y, r["oop_total"], days, r["net"]))
    df = pd.DataFrame(rows, columns=["y", "oop", "days", "net"]); dv = df[df.y <= 2020]; sn = df[df.y > 2020]
    return dict(case=label, k0=k0, dev_days_med=dv.days.median(), dev_never=int(dv.days.isna().sum()), dev_in90=float((dv.days <= 90).mean()), dev_in180=float((dv.days <= 180).mean()), dev_oop_med=dv.oop.median(), dev_oop_max=dv.oop.max(),
                dev_net_mean=dv.net.mean(), seen_days=str(sn.days.round(0).tolist()), seen_net=str(sn.net.round(0).tolist()), seen_oop=str(sn.oop.round(0).tolist()))
out = []
base = FundedPlan("p", "x", 4000., .5, 1., 0., .5, 1., 99, .4, .8); fast = FundedPlan("p", "x", 6000., .5, 1., 0., .5, 1., 99, 2.4, .8)
for k0 in (1, 2, 3, 5):
    out.append(evaluate("Topstep 150K", base, 9600., k0, "Topstep, previous config"))
    out.append(evaluate("Topstep 150K", fast, 9600., k0, "Topstep, fast config"))
for name in ("FTMO Growth 150K", "FundedNext Flex 150K", "Take Profit Trader PRO 150K"):
    b = best[name]; out.append(evaluate(name, mkplan(S[name], float(b["t1"]), float(b["a1"]), float(b["a2"])), float(b["R"]), 1, name + " (own best config)"))
R = pd.DataFrame(out); R.to_csv("results/lead_and_parallel.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
print(R[["case", "k0", "dev_days_med", "dev_never", "dev_in90", "dev_in180", "dev_oop_med", "dev_oop_max", "dev_net_mean"]].round(2).to_string(index=False))
print(R[["case", "k0", "seen_days", "seen_net", "seen_oop"]].to_string(index=False))
