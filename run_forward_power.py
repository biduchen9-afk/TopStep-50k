"""Forward-test v2 (PR18) operating characteristics with WEEKLY looks: how often do the kill rules fire when the edge is the full development edge, 75%, 50% or zero?
R = daily P&L per micro / ATR in dollars (what the farm sizes on), sleeves re-charged at today's cost-to-ATR (retest_cost.rebase; R is identical in the 'exact' and 'cost_only' bases), development days only,
21-day block bootstrap, looks at sessions 60, 65, ..., 250; rule (ii) uses the 5th percentile of equal-length development windows at EACH look, rule (iii) applies from session 120."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import retest as T
import retest_cost as C
from portfolio import DEV_END

D = T.load(); ST = C.rebase(D["ST"], 1.0, "exact")
dev = (ST["noise"].dates <= np.datetime64(DEV_END)) & ST["noise"].ok & ST["gap"].ok
R = {g: np.where(dev, ST[g].pnl / np.where(ST[g].ok, ST[g].atr_usd, 1.0), 0.0)[dev] for g in ST}
n = len(R["noise"]); mu = {g: R[g].mean() for g in R}
print(f"dev days used (both sleeves tradable): {n}; mean R/day noise {mu['noise']:.4f}, gap {mu['gap']:.4f}")
pooled = lambda a, b: (3 * a + 2 * b) / 5.0
P = pooled(R["noise"], R["gap"])
LOOKS = list(range(60, 251, 5))
c = np.r_[0.0, np.cumsum(P)]
th = {L: float(np.percentile(c[L:] - c[:-L], 5)) for L in LOOKS}
for L in (60, 120, 250):
    w = c[L:] - c[:-L]; print(f"L={L}: dev rolling-window pooled R sum: p5 {th[L]:.3f}, median {np.median(w):.3f}, p95 {np.percentile(w, 95):.3f}, share > 0 {np.mean(w > 0):.1%}")
rng = np.random.default_rng(18); B, blk = 20000, 21
def paths(scale):
    nb = int(np.ceil(250 / blk)); st_ = rng.integers(0, n - blk, size=(B, nb))
    ix = (st_[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :250]
    a = R["noise"][ix] - (1 - scale) * mu["noise"]; b = R["gap"][ix] - (1 - scale) * mu["gap"]
    return np.cumsum(a, 1), np.cumsum(b, 1), np.cumsum(pooled(a, b), 1)
rows = []
for scale in (1.0, 0.75, 0.5, 0.0):
    ca, cb, cp = paths(scale)
    fire = np.zeros((B, len(LOOKS)), bool)
    for j, L in enumerate(LOOKS):
        r2 = cp[:, L - 1] < th[L]
        r3 = (L >= 120) & (ca[:, L - 1] < 0) & (cb[:, L - 1] < 0)
        fire[:, j] = r2 | r3
    ever = np.maximum.accumulate(fire, axis=1)
    at = lambda T_: float(ever[:, LOOKS.index(T_)].mean())
    rows.append(dict(edge=scale, stop_by_60=at(60), stop_by_120=at(120), stop_by_250=at(250), pooled_R_sum_120_median=float(np.median(cp[:, 119])), share_positive_120=float((cp[:, 119] > 0).mean()), pooled_R_sum_250_median=float(np.median(cp[:, 249]))))
df = pd.DataFrame(rows); df.to_csv("results/forward_power.csv", index=False); json.dump({str(k): v for k, v in th.items()}, open("results/forward_thresholds_R.json", "w")); print(df.round(3).to_string(index=False))
f1, f0 = df.loc[df.edge == 1.0, "stop_by_250"].iloc[0], df.loc[df.edge == 0.0, "stop_by_250"].iloc[0]
print(f"likelihood ratio of a STOP by session 250 (zero edge vs full edge): {f0 / f1:.2f}")
