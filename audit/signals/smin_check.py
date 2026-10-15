import sys
sys.path.insert(0, '..')
import numpy as np, pandas as pd
from market_data import build_sessions
for mkt in ('nq', 'es', 'gc'):
    s = build_sessions(mkt)
    pk = []
    for i in range(len(s)):
        o, h, l, c, v, m = s.day(i)
        w = (m >= 870) & (m <= 990)           # 09:00-11:00 ET
        if w.sum() < 60:
            pk.append(-1); continue
        pk.append(int(m[w][v[w].argmax()]))
    pk = pd.Series(pk, index=pd.DatetimeIndex(s.dates))
    print(mkt, 'share of sessions whose max-volume minute in 09:00-11:00 ET is smin=930 (09:30):', round((pk == 930).mean(), 3))
    bad = pk[(pk != 930) & (pk != -1)]
    print('   others by smin:', bad.value_counts().head(6).to_dict())
    if mkt == 'gc':
        w2 = pk.groupby(pk.index.year).apply(lambda x: x.value_counts().index[0])
        print('   gc modal peak minute by year (GC cash-hour peak is 08:20 COMEX = smin 860 so 930 not expected):', w2.to_dict())
    # DST-transition weeks: the Monday sessions after each transition
    idx = pk.index
    for y in (2012, 2016, 2019, 2023, 2025):
        mar = pd.Timestamp(y, 3, 1); s2 = mar + pd.Timedelta(days=(6 - mar.dayofweek) % 7 + 7)
        nov = pd.Timestamp(y, 11, 1); s1 = nov + pd.Timedelta(days=(6 - nov.dayofweek) % 7)
        for lab, dsun in (('spring', s2), ('fall', s1)):
            wk = pk[(idx > dsun) & (idx <= dsun + pd.Timedelta(days=5))]
            print('   ', mkt, y, lab, 'forward week peak smin:', wk.to_list())
    # first bar of every session is smin==0..?
    first = s.smin[s.offsets[:-1]]
    print('   first-bar smin distribution:', pd.Series(first).value_counts().head(3).to_dict(), ' last bar smin max', s.smin.max())
