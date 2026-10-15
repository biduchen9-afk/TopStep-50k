"""Dashboard data: the search for > 80% of funded accounts paid and a first payout within 90 days (pre-registration 14). Writes results/dashboard_v4/fast_paid_search.json."""
import json, sys
import numpy as np, pandas as pd
from portfolio import DEV_END
import run_kelly as K
import run_paid_share as P
import run_paid_share_d as D

G = pd.read_csv("results/search_fast_paid_all.csv")
gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}; ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
AC = {k: P.Acc(ST[k], n_dev) for k in ST}
CASES = [("Today: both groups size 1.2, first payout at $2,000", (1.2, 2000.0), (1.2, 2000.0), "Baseline"),
         ("Option A: noise-area 1.2, first payout $750; gap-VWAP 0.4, first payout $1,000", (1.2, 750.0), (0.4, 1000.0), "Qualifies on the pooled sample, not stable (2010-16 starts take 112 days; 2023+ paid 69%)"),
         ("Option B: noise-area 0.6, first payout $750; gap-VWAP 0.8, first payout $750", (0.6, 750.0), (0.8, 750.0), "Qualifies on the pooled sample, not stable (2010-16 starts only 62% paid), profit about 30%"),
         ("Earlier two-tier: noise-area 1.2, first payout $1,000; gap-VWAP 0.4, first payout $2,000", (1.2, 1000.0), (0.4, 2000.0), "More than 80% paid but 104 days")]
rows = []
for label, nz, gp, verdict in CASES:
    a = P.pooled([(AC["noise"].run(nz[0], nz[1], 0.8), 3), (AC["gap"].run(gp[0], gp[1], 0.8), 2)]); f = a[a.passed]; p = f[f.paid]
    g = G[(G.noise_a1 == nz[0]) & (G.noise_t1 == nz[1]) & (G.gap_a1 == gp[0]) & (G.gap_t1 == gp[1])].iloc[0]
    k = K.evaluate(D.plans_pg(ST, nz, gp), ref, n_dev, n); seen = json.loads(k["seen"]) if isinstance(k["seen"], str) else k["seen"]
    rows.append(dict(setting=label, verdict=verdict, paid_pooled=round(len(p) / len(f), 3), paid_noise=round(g.paid_noise, 3), paid_gap=round(g.paid_gap, 3), days_med=round(p.d_total.median()), days_p25=round(p.d_total.quantile(.25)), days_p75=round(p.d_total.quantile(.75)),
                     within_90=round((p.d_total <= 90).sum() / len(f), 3), first_payout=round(p["first"].mean()), paid_per_paid=round(p.total.mean()), farm_mean=round(k["dev_mean"]), farm_p25=round(k["dev_p25"]), farm_worst=round(k["dev_worst"]), green=int(k["green"]),
                     farm_days=round(k["days_first_payout"]), farm_paid=round(k["paid_share"], 3), early_paid=round(g.paid_early, 3), early_days=round(g.days_early), late_paid=round(g.paid_late, 3), late_days=round(g.days_late), paid_2023=round(g.paid_2023, 3),
                     days_2023=round(g.days_2023), seen_2021=seen[0], seen_2023=seen[1], seen_2025=seen[2]))
json.dump(rows, open("results/dashboard_v4/fast_paid_search.json", "w")); print(pd.DataFrame(rows)[["setting", "paid_pooled", "days_med", "within_90", "first_payout", "farm_mean", "farm_p25", "farm_days"]].to_string(index=False))
