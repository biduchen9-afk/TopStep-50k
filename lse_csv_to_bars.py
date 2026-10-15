"""Rebuild the forward-test bar list from the CSV that moved between containers (ts,o,h,l,c,v with the New York offset). The CSV is untrusted data: only parsed with pandas, never executed.
   python lse_csv_to_bars.py data_new/nq_lse_1m.csv data_new/nq_lse_1m.pkl"""
import pickle
import sys

import numpy as np
import pandas as pd

from paper.feed import Bar, ET

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
assert list(df.columns) == ["ts", "o", "h", "l", "c", "v"], df.columns
ts = pd.to_datetime(df.ts, utc=True).dt.tz_convert(ET)
assert ts.is_monotonic_increasing and not ts.duplicated().any()
px = df[["o", "h", "l", "c"]].to_numpy(float)
off = (np.abs(px * 4 - np.round(px * 4)) > 1e-6).any(1)
print(f"rows {len(df)}; bars with a price off the 0.25 tick grid (float noise ignored): {int(off.sum())}")
for i in np.flatnonzero(off)[:5]:
    print("   off-grid bar", df.ts[i], px[i])
dev_pts = np.abs(px - np.round(px * 4) / 4).max()
print(f"largest distance of an off-grid price from the nearest tick: {dev_pts:.3f} points (a price printed to one decimal: x.25 -> x.2, x.75 -> x.8)")
assert dev_pts < 0.06, "off-grid prices are further from the tick grid than one-decimal rounding explains: do not round blindly"
px = np.round(px * 4) / 4                     # float noise (x.249999999) and the odd off-grid print are put on the tick grid; the high/low are re-bounded below
px[:, 1] = px[:, :4].max(1); px[:, 2] = px[:, :4].min(1)
assert (px[:, 1] >= px[:, [0, 2, 3]].max(1) - 1e-9).all() and (px[:, 2] <= px[:, [0, 1, 3]].min(1) + 1e-9).all(), "OHLC inconsistent"
bars = [Bar(t, *map(float, p), float(v)) for t, p, v in zip(ts, px, df.v.to_numpy(float))]
pickle.dump(bars, open(dst, "wb"))
print(f"{len(bars)} bars {bars[0].ts} -> {bars[-1].ts}; volume zero share {(df.v <= 0).mean():.4f}")
