"""Audit: how much of the farm hold-out result comes from which sub-period? (diagnostic only; reuses the analyst's frozen farm config a1=a2=0.4, 5 slots)"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from pricing_config import load_config
from real_engine import rules_from_config
from farm_real import FarmSpec, run_farm
from portfolio import DEV_END
from run_farm_dev import plan, SZ
from run_farm_robustness import build
from run_dynamic_contract import PATH

cfg = load_config(); rules = rules_from_config(cfg, PATH)
st_ = build()["NQ noise-area"]
n = len(st_.dates); n_h = int(np.searchsorted(st_.dates, DEV_END, side="right"))
d = pd.DatetimeIndex(st_.dates)
print("hold-out first/last date", d[n_h], d[-1], "sessions", n - n_h)
for lab, a2 in (("a2=0.4", 0.4), ("a2=1.6 (dev-frozen)", 1.6)):
    print(lab)
    for start in ("2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"):
        i0 = int(np.searchsorted(st_.dates, np.datetime64(start)))
        for end_lab, n_end in (("to data end", None),):
            r = run_farm([st_], SZ, None, plan(0.4, a2), cfg, rules, PATH, i0, FarmSpec(max_evals=20, max_funded=5, p_est=1.0), n_end=n_end)
            yrs = (st_.dates[-1] - st_.dates[i0]).astype(float) / 365.25
            print(f"  start {start}: yrs {yrs:.1f} net {r['net']:>9.0f}  per-yr {r['net']/yrs:>8.0f} payouts {r['payouts']:.0f} fees {r['fees']:.0f} funded {r['n_funded']} died {r['died']}")
    # also a dev-window spot check (single 3.5y windows)
for start in ("2012-01-01", "2014-01-01", "2016-01-01", "2018-01-01", "2019-06-01"):
    i0 = int(np.searchsorted(st_.dates, np.datetime64(start)))
    r = run_farm([st_], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, i0, FarmSpec(max_evals=20, max_funded=5, p_est=1.0), n_end=i0 + 882)
    print(f"  DEV window {start} (3.5y): net {r['net']:.0f}")
