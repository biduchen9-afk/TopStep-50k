import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from pricing_config import load_config
from real_engine import rules_from_config
from farm_real import FarmSpec, run_farm
from portfolio import DEV_END, basket, make_sleeve
from run_farm_dev import plan, SZ
PATH = "no_activation_dll"
cfg = load_config(); rules = rules_from_config(cfg, PATH)
s = basket([make_sleeve("nq", "noise_area", {}, name="nq_noise")], name="nq")
n_h = int(np.searchsorted(s.dates, DEV_END, side="right"))
print('n', len(s.dates), 'n_h', n_h, s.dates[n_h])
t = time.time()
rows = []
for o in range(n_h, n_h + 60, 10):
    r = run_farm([s], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, o, FarmSpec(max_evals=20, max_funded=5, p_est=1.0))
    rows.append(r)
print(time.time() - t)
df = pd.DataFrame(rows)
print(df[['net', 'fees', 'payouts', 'n_evals', 'n_funded', 'n_passed', 'sessions']])
print(df.net.mean())
