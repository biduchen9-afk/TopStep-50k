import sys
sys.path.insert(0, '..')
from pathlib import Path
import numpy as np
import pandas as pd
import market_data as md
SCR = Path('/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/rawlink')
for m in ('nq', 'es', 'gc'):
    fresh = md.build_sessions(m, SCR)               # builds from pkl, writes cache only in scratch dir
    old = md.build_sessions(m, Path('/home/user/data_raw'))  # reads existing cache (read-only)
    same = all(np.array_equal(getattr(fresh, k), getattr(old, k)) for k in ('dates', 'offsets', 'o', 'h', 'l', 'c', 'v', 'smin', 'roll'))
    print(m, 'sessions', len(fresh), len(old), 'identical to cache:', same, 'rolls', int(fresh.roll.sum()))
    # compare the pkl with the txt (first 1e5 + random rows)
    if m == 'nq':
        from load_raw import load
        raw = load('nq')
        pk = pd.read_pickle(SCR / 'nq_1m.pkl')
        print('pkl rows', len(pk), 'txt rows', len(raw), 'index equal', (pk.index.tz_localize(None) if pk.index.tz is not None else pk.index).equals(raw.index))
        print('values equal', np.allclose(pk[['o', 'h', 'l', 'c', 'v']].to_numpy(), raw[['o', 'h', 'l', 'c', 'v']].to_numpy()))
