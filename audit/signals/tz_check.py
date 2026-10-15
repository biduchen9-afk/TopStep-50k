import pandas as pd, numpy as np
from load_raw import load
df = load('nq')
mod = df.index.hour * 60 + df.index.minute
df['md'] = mod
df['yr'] = df.index.year
df['mo'] = df.index.month
# 1. gap structure: which minutes-of-day have bars at all (halt)
cnt = df.groupby('md').size()
print('minutes of day with <5% of max count:', [f'{m//60:02d}:{m%60:02d}' for m in cnt.index[cnt < 0.05*cnt.max()]][:5], '...')
hrs = cnt.groupby(cnt.index//60).sum()
print('bars per hour-of-day:\n', hrs.to_string())
# 2. volume peak for each regime: Jan (std time) vs July (DST) vs the weeks around transitions
for label, sel in [('Jan-Feb (EST)', df.mo.isin([1, 2])), ('Jul-Aug (EDT)', df.mo.isin([7, 8])), ('Dec', df.mo == 12), ('Jun', df.mo == 6)]:
    d = df[sel & (df.md >= 8*60) & (df.md <= 11*60)]
    g = d.groupby('md').v.mean()
    top = g.sort_values(ascending=False).head(4)
    print(label, 'top vol minutes', [f'{m//60:02d}:{m%60:02d}' for m in top.index])
# 3. volume at the 09:29/09:30/09:31 minutes by year
d = df[(df.md >= 9*60+28) & (df.md <= 9*60+32)]
print((d.groupby(['yr', 'md']).v.mean().unstack().astype(int)).to_string())
# 4. around US DST transitions: weeks between US and EU DST differ -> check 9:30 peak in March 8-28 2022 etc
for a, b in [('2022-03-14', '2022-03-25'), ('2022-10-31', '2022-11-04'), ('2023-03-13', '2023-03-24'), ('2015-03-09','2015-03-20'), ('2015-10-26','2015-10-30')]:
    d = df[a:b]; d = d[(d.md >= 8*60) & (d.md <= 11*60)]
    g = d.groupby('md').v.mean().sort_values(ascending=False).head(2)
    print(a, b, [f'{m//60:02d}:{m%60:02d}' for m in g.index])
# 5. daily first bar after 17:00-18:00 halt: distribution of minute-of-day of first bar at/after 17:00
d = df[(df.md >= 16*60+55)]
first = d.groupby(d.index.normalize()).md.min()
print('first bar >=16:55 each day (mod->count):', (first.value_counts().head(6)))
last = df[(df.md >= 15*60) & (df.md < 17*60+30)]
lastb = last.groupby(last.index.normalize()).md.max()
print('last bar before 17:30 each day:', lastb.value_counts().head(6))
# bars at 16:00 minute: volume spike?
print('mean vol 15:59, 16:00, 16:01 :', [int(df[df['md']==m].v.mean()) for m in (15*60+59, 16*60, 16*60+1)])
# 6. Sunday open
sun = df[df.index.dayofweek == 6]
fb = sun.groupby(sun.index.normalize()).md.min()
print('first bar on Sundays (mod->count):', fb.value_counts().head(5))
print('Sunday first bar by year:', fb.groupby(fb.index.year).agg(lambda x: x.value_counts().index[0]))
