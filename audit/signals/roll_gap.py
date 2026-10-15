import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from load_raw import load
from market_data import _third_friday
mkt = sys.argv[1] if len(sys.argv) > 1 else 'nq'
yrs = [int(a) for a in sys.argv[2:]] or [2013]
df = load(mkt)
prev_c = df['c'].shift(1)
gapm = df.index.to_series().diff() / pd.Timedelta(minutes=1)
jump = df['o'] - prev_c
for y in yrs:
    for mth in (3, 6, 9, 12):
        tf = _third_friday(y, mth)
        lo, hi = tf - pd.Timedelta(days=12), tf + pd.Timedelta(days=1)
        w = pd.DataFrame({'jump': jump[lo:hi], 'gap_min': gapm[lo:hi]})
        w['rel_bp'] = w.jump / df['c'][lo:hi] * 1e4
        top = w.reindex(w.jump.abs().sort_values(ascending=False).index).head(4)
        print(y, mth, 'expiry', tf.date())
        print(top.round(2).to_string())
