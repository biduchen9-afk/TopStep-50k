"""Forward-test v2 (PR18) operating characteristics: how often do the kill rules fire when the edge is the full development edge, 75%, 50% or zero?
R = daily P&L per micro / ATR in dollars (what the farm sizes on), sleeves re-charged at today's cost-to-ATR (retest_cost.rebase), development days only, 21-day block bootstrap."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import retest as T
import retest_cost as C
from portfolio import DEV_END

D = T.load(); ST = C.rebase(D["ST"], 1.0)
dev = (ST["noise"].dates <= np.datetime64(DEV_END)) & ST["noise"].ok & ST["gap"].ok
R = {g: np.where(dev, ST[g].pnl / np.where(ST[g].ok, ST[g].atr_usd, 1.0), 0.0)[dev] for g in ST}
n = len(R["noise"]); mu = {g: R[g].mean() for g in R}
print(f"dev days used (both sleeves tradable): {n}; mean R/day noise {mu['noise']:.4f}, gap {mu['gap']:.4f}")
pooled = lambda a, b: (3 * a + 2 * b) / 5.0
P = pooled(R["noise"], R["gap"])
LS = (60, 120, 250)
th = {}
for L in LS:
    c = np.r_[0.0, np.cumsum(P)]; w = c[L:] - c[:-L]
    th[L] = float(np.percentile(w, 5)); print(f"L={L}: dev rolling-window pooled R sum: p5 {th[L]:.3f}, median {np.median(w):.3f}, p95 {np.percentile(w, 95):.3f}, share > 0 {np.mean(w > 0):.1%}")
rng = np.random.default_rng(18); B, blk = 20000, 21
def paths(scale):
    nb = int(np.ceil(max(LS) / blk)); st_ = rng.integers(0, n - blk, size=(B, nb))
    ix = (st_[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :max(LS)]
    a = R["noise"][ix] - (1 - scale) * mu["noise"]; b = R["gap"][ix] - (1 - scale) * mu["gap"]
    return np.cumsum(a, 1), np.cumsum(b, 1), np.cumsum(pooled(a, b), 1)
rows = []
for scale in (1.0, 0.75, 0.5, 0.0):
    ca, cb, cp = paths(scale)
    r60 = cp[:, 59] < th[60]; r120 = cp[:, 119] < th[120]; r250 = cp[:, 249] < th[250]
    both120 = (ca[:, 119] < 0) & (cb[:, 119] < 0); both250 = (ca[:, 249] < 0) & (cb[:, 249] < 0)
    stop60 = r60; stop120 = r60 | r120 | both120; stop250 = stop120 | r250 | both250
    rows.append(dict(edge=scale, stop_by_60=float(stop60.mean()), stop_by_120=float(stop120.mean()), stop_by_250=float(stop250.mean()), rule_ii_120_only=float(r120.mean()), both_sleeves_negative_120=float(both120.mean()),
                     pooled_R_sum_120_median=float(np.median(cp[:, 119])), share_positive_120=float((cp[:, 119] > 0).mean())))
df = pd.DataFrame(rows); df.to_csv("results/forward_power.csv", index=False); json.dump(th, open("results/forward_thresholds_R.json", "w")); print(df.round(3).to_string(index=False))
