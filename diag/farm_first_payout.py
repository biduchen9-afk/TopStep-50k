"""Farm-level time to the FIRST payout: calendar days from the day the farm buys its first Combine to the first payout of ANY account (staggered Combines, up to 5 funded)."""
import numpy as np, pandas as pd
from combine_policy import CombinePolicy
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from final_strategy import PATH, build_stream
from pricing_config import load_config
from real_engine import FundedPlan, rules_from_config
s = build_stream(); cfg = load_config(); rules = rules_from_config(cfg, PATH); d = pd.DatetimeIndex(s.dates)
n_dev = int(np.searchsorted(s.dates, np.datetime64("2022-12-31"), side="right"))
first = int(np.flatnonzero(s.ok)[0]) + 260
def first_payout_days(pl, i0, n_end=None):
    r = run_farm([s], CombineSizing("fixed", 9600.), None, pl, cfg, rules, PATH, i0, FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill="hyst"), n_end=n_end, log=True)
    t = [e[0] for e in r["events"] if e[1] == "payout"]
    return (d[min(min(t), len(d) - 1)] - d[i0]).days if t else np.nan
for lab, t1, a1 in (("baseline t1=3000 a1=0.4", 3000., .4), ("t1=2000 a1=0.8", 2000., .8), ("t1=1500 a1=1.2", 1500., 1.2)):
    pl = FundedPlan("p", "standard", t1, .5, 1., 0., .5, 1., 99, a1, 1.6)
    dev = [first_payout_days(pl, i0, min(i0 + 400, n_dev)) for i0 in range(first, n_dev - 400, 21)]
    hold = [first_payout_days(pl, i0, len(d)) for i0 in range(int(np.searchsorted(s.dates, np.datetime64("2023-01-01"))), len(d) - 150, 21)]
    f = lambda v: f"median {np.nanmedian(v):.0f}d, within 90d: {np.mean(np.array(v) <= 90):.0%}, no payout within 400 sessions/data: {np.mean(np.isnan(v)):.0%}"
    print(f"{lab:<26} dev farms: {f(dev)} | 2023+ farms: {f(hold)}")
