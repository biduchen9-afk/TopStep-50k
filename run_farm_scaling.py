"""Hold-out scaling of the farm: funded slots 1..5 and eval aggressiveness (concurrent evals up to 20)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pricing_config import load_config
from real_engine import rules_from_config
from farm_real import FarmSpec, run_farm
from portfolio import DEV_END
from run_farm_dev import plan, SZ
from run_farm_robustness import build
from run_dynamic_contract import PATH
pd.set_option("display.width", 250, "display.max_columns", 40)
cfg = load_config(); rules = rules_from_config(cfg, PATH)
streams = build()
rows = []
for name in ("NQ noise-area", "portfolio", "NQ-only sleeves (noise+vwap)"):
    s = streams[name]
    n_h = int(np.searchsorted(s.dates, DEV_END, side="right")); yrs = (s.dates[-1] - s.dates[n_h]).astype(float) / 365.25
    for mf, me, pe in [(1, 20, 1.0), (2, 20, 1.0), (3, 20, 1.0), (4, 20, 1.0), (5, 20, 1.0), (5, 20, 0.5), (5, 20, 0.3), (5, 5, 1.0)]:
        r = pd.DataFrame([run_farm([s], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, o, FarmSpec(max_evals=me, max_funded=mf, p_est=pe)) for o in range(n_h, n_h + 60, 10)])
        rows.append(dict(stream=name, funded_slots=mf, max_evals=me, p_est=pe, net=r.net.mean(), per_year=r.net.mean() / yrs, worst=r.net.min(), best=r.net.max(),
                         evals=r.n_evals.mean(), funded=r.n_funded.mean(), wasted_passes=r.n_wasted.mean(), payouts=r.payouts.mean(), fees=r.fees.mean(),
                         avg_funded_active=r.avg_funded_active.mean(), avg_evals_active=r.avg_evals_active.mean(), max_funded_seen=r.max_funded_seen.max(), max_evals_seen=r.max_evals_seen.max()))
df = pd.DataFrame(rows); df.to_csv("results/farm_scaling.csv", index=False)
print(df.round(1).to_string(index=False))
