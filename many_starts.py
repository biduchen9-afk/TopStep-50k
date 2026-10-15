"""Distribution of 3-year farm outcomes over many overlapping start dates (every STEP sessions through the development period) for the baseline, on the raw dir given by TOPSTEP_RAW_DIR.
A single set of ten annual starts is a noisy estimate (audit, October 2026): neighbouring farms share almost all their path, and a handful of accounts decide a farm's result."""
import os, sys
import numpy as np, pandas as pd
import run_kelly as K, run_paid_share as P
from multi_firm import Policy, run_multifirm
from portfolio import DEV_END

STEP = int(os.environ.get("STEP", "10"))
WINDOW = int(os.environ.get("WINDOW", "756"))
gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}; ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
first = int(np.flatnonzero(ref.ok)[0]); starts = list(range(first, n_dev - WINDOW, STEP))
fps = P.plans_g(ST, [("noise", 3), ("gap", 2)], 1.2, 2000.0, 0.8); pol = Policy(reserve_cycles=None, global_one_eval=True)
nets = np.array([run_multifirm(fps, pol, i0, i0 + WINDOW)["net"] for i0 in starts])
d = pd.DatetimeIndex(ref.dates)
tag = os.path.basename(os.environ.get("TOPSTEP_RAW_DIR", "data_raw"))
pd.DataFrame(dict(start=d[starts], net=nets)).to_csv(f"results/many_starts_{tag}_w{WINDOW}.csv", index=False)
q = np.percentile(nets, [10, 25, 50, 75, 90])
print(f"{tag} window {WINDOW}: {len(nets)} farms (every {STEP} sessions) mean {nets.mean():,.0f}  p10 {q[0]:,.0f}  p25 {q[1]:,.0f}  median {q[2]:,.0f}  p75 {q[3]:,.0f}  p90 {q[4]:,.0f}  share > 0: {(nets > 0).mean():.0%}  worst {nets.min():,.0f}")
