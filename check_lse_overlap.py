"""PR15/PR18 data check 5: LSE NQ.F against the stored Databento NQ over the minutes both cover (2026-05-31 .. 2026-07-02). Price-level, roll and join-continuity.
   TOPSTEP_RAW_DIR=/home/user/data_raw python check_lse_overlap.py data_new/nq_lse_1m.pkl"""
import os, pickle, sys
os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw")
import numpy as np, pandas as pd
import market_data as md

bars = pickle.load(open(sys.argv[1], "rb"))
L = pd.DataFrame(dict(o=[b.o for b in bars], h=[b.h for b in bars], l=[b.l for b in bars], c=[b.c for b in bars], v=[b.v for b in bars]), index=pd.DatetimeIndex([b.ts for b in bars]).tz_convert("UTC"))
D = md.load_minute("nq"); D.index = D.index.tz_convert("UTC"); D = D.loc[L.index[0]:]
J = D.join(L, how="outer", lsuffix="_db", rsuffix="_lse")
both = J.dropna(subset=["c_db", "c_lse"])
print(f"Databento minutes in the window {D.index.min()} .. {D.index.max()}: {len(D)}; LSE minutes in the same window: {int((L.index <= D.index.max()).sum())}; both: {len(both)}")
print(f"share of Databento minutes also in LSE {len(both) / len(D):.1%}; share of LSE minutes (in window) also in Databento {len(both) / int((L.index <= D.index.max()).sum()):.1%}")
d = (both.c_lse - both.c_db)
print(f"close difference LSE - Databento over matching minutes: median {d.median():.2f}, mean {d.mean():.2f}, p5 {d.quantile(.05):.2f}, p95 {d.quantile(.95):.2f}, share |diff| <= 1 point {(d.abs() <= 1).mean():.1%}")
day = d.groupby(both.index.tz_convert("America/New_York").strftime("%Y-%m-%d")).agg(["median", "count"])
day["median"] = day["median"].round(2)
print("by ET calendar day (median diff, minutes):")
print(day.T.to_string())
r = pd.concat([both.c_lse.pct_change(), both.c_db.pct_change()], axis=1).dropna(); r.columns = ["lse", "db"]
print(f"1-minute return correlation: {r.corr().iloc[0, 1]:.4f}; mean abs return gap {(r.lse - r.db).abs().mean() * 1e4:.3f} bp")
