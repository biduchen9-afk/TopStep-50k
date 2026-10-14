import sys
sys.path.insert(0, '..')
import numpy as np, pandas as pd
from load_raw import load
from market_data import build_sessions
df = load('nq')
hr = df.index.hour
d = df[(hr >= 18) | (hr < 16)]
lab = (d.index.normalize() + pd.to_timedelta(np.where(d.index.hour >= 18, 1, 0), unit='D'))
cnt = pd.Series(1, index=lab).groupby(level=0).sum()
cnt = cnt[cnt.index.dayofweek < 5]
sess = build_sessions('nq')
kept = set(pd.DatetimeIndex(sess.dates))
dropped = cnt[~cnt.index.isin(kept)]
print('label-days with bars but dropped (<500 bars):', len(dropped))
print(dropped.to_string())
# the daily close-to-close move of NQ around dropped days
c = df['c'].resample('1D').last().dropna()
ret = c.pct_change()
print('abs daily ret on/around dropped days (pct):')
for dt in dropped.index:
    r = ret.reindex([dt, dt + pd.Timedelta(days=1)])
    print(dt.date(), r.round(4).to_list())
# weekdays with no bars at all
alld = pd.bdate_range(sess.dates.min(), sess.dates.max())
missing = [x for x in alld if x not in kept and x not in dropped.index]
print('weekdays with no bars at all:', len(missing), [str(x.date()) for x in missing][:60])
# sessions with early close (last bar before 15:50)
early = []
for i in range(len(sess)):
    m = sess.smin[sess.offsets[i]:sess.offsets[i + 1]]
    if m[-1] < 21 * 60 + 50:
        early.append((pd.Timestamp(sess.dates[i]).date(), int(m[-1])))
print('sessions whose last bar is before 15:50 ET:', len(early), early[:40])
