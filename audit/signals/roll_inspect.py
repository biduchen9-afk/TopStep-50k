import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from market_data import build_sessions

for mkt in ('nq',):
    s = build_sessions(mkt)
    idx = pd.DatetimeIndex(s.dates)
    rows = []
    for i in range(len(s)):
        o, h, l, c, v, m = s.day(i)
        d = np.diff(c)
        rel = np.abs(d) / c[1:]
        j = rel.argmax()
        wall_min = (m[j + 1] + 1080) % 1440
        rows.append((idx[i], s.roll[i], d[j], rel[j] * 1e4, f'{wall_min//60:02d}:{wall_min%60:02d}', h.max() - l.min(), c[-1]))
    df = pd.DataFrame(rows, columns=['date', 'roll', 'jump', 'jump_bp', 'time', 'range', 'close']).set_index('date')
    fl = df[df.roll]
    print(mkt, 'flagged', len(fl))
    print(fl.head(70).round(2).to_string())
    print('\nflagged: median |jump| bp', fl.jump_bp.median(), ' median unflagged session max jump bp', df[~df.roll].jump_bp.median())
    print('flagged range/ATR-ish: median range', fl['range'].median(), 'vs overall', df['range'].median())
    print('time-of-day histogram of flagged jumps:', fl.time.value_counts().head(8).to_dict())
    df.to_pickle('/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/jumps_nq.pkl')
