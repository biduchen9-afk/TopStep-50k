"""PR19 operating characteristics (docs/PREREGISTRATION_19.md): how often the pre-registered R-unit kill line fires and the 36-month confirm line is met, by the size of the true edge.
R = daily P&L per micro / ATR in $ per micro, sleeves re-charged at today's cost-to-ATR (retest_cost.rebase 'exact', same as PR18), development days only (both sleeves tradable), 21-day block bootstrap
of the development days with the edge scaled to lambda x its development mean (as run_forward_power.py), 756 sessions, monthly looks (21 sessions) from session 63.
Run with TOPSTEP_RAW_DIR=/home/user/data_raw_live. Output: results/pr19_operating_characteristics.csv"""
from __future__ import annotations

import math
import os

os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import numpy as np
import pandas as pd

import retest as T
import retest_cost as C
from paper import verdict as V
from portfolio import DEV_END

D = T.load(); ST = C.rebase(D["ST"], 1.0, "exact")
dev = (ST["noise"].dates <= np.datetime64(DEV_END)) & ST["noise"].ok & ST["gap"].ok
Rr = {g: np.where(dev, ST[g].pnl / np.where(ST[g].ok, ST[g].atr_usd, 1.0), 0.0)[dev] for g in ST}
n = len(Rr["noise"]); mu = {g: Rr[g].mean() for g in Rr}
pooled = lambda a, b: (3 * a + 2 * b) / 5.0
Pd = pooled(Rr["noise"], Rr["gap"])
print(f"dev sessions {n}; pooled R per session: mean {Pd.mean():.4f}, sd {Pd.std():.4f}  (verdict.py constants: mean {V.MU_DEV}, sd {V.SIGMA_DEV})")
assert abs(Pd.mean() - V.MU_DEV) < 5e-4 and abs(Pd.std() - V.SIGMA_DEV) < 2e-3, "verdict.py constants do not match the development data"
T_, B, BLK = V.HORIZON, 20000, 21
looks = np.arange(V.FIRST_LOOK, T_ + 1, V.EVERY)
thr = -V.H19 + V.SLOPE19 * looks
rng = np.random.default_rng(19)
rows = []
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    nb = int(np.ceil(T_ / BLK)); st_ = rng.integers(0, n - BLK, size=(B, nb))
    ix = (st_[:, :, None] + np.arange(BLK)[None, None, :]).reshape(B, -1)[:, :T_]
    p = pooled(Rr["noise"][ix] - (1 - lam) * mu["noise"], Rr["gap"][ix] - (1 - lam) * mu["gap"])
    cp = np.cumsum(p, 1)
    hit = cp[:, looks - 1] <= thr[None, :]
    first = np.where(hit.any(1), looks[hit.argmax(1)], 10 ** 9)
    conf = (first > T_) & (cp[:, T_ - 1] >= V.CONFIRM_R)
    rows.append(dict(edge_share_of_dev=lam, kill_by_3m=(first <= 63).mean(), kill_by_6m=(first <= 126).mean(), kill_by_12m=(first <= 252).mean(), kill_by_24m=(first <= 504).mean(),
                     kill_by_36m=(first <= 756).mean(), confirm_at_36m=conf.mean(), cannot_tell_at_36m=((first > T_) & ~conf).mean(), median_cum_R_36m=float(np.median(cp[:, T_ - 1])), paths=B))
df = pd.DataFrame(rows); df.to_csv("results/pr19_operating_characteristics.csv", index=False)
pd.set_option("display.width", 220); print(df.round(3).to_string(index=False))
print("kill line at sessions 63 / 126 / 252 / 504 / 756:", np.round(-V.H19 + V.SLOPE19 * np.array([63, 126, 252, 504, 756]), 2), "| confirm line at 756:", round(V.CONFIRM_R, 2))
