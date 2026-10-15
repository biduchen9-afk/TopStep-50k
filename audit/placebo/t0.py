import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from market_data import build_sessions
import academic_strategies as ac
t = time.time()
s = build_sessions('nq'); print(len(s), s.dates[0], s.dates[-1], len(s.c), time.time() - t)
t = time.time(); ctx = ac.context(s); print('ctx', time.time() - t)
t = time.time(); d = ac.run_sleeve(s, 'noise_area', ctx=ctx); print('run', time.time() - t)
print(d.sharpe(), d.pnl.sum(), d.traded_ok.sum(), d.trades.sum())
r = d.pnl / np.where(d.traded_ok, d.atr * d.mult, 1); m = d.traded_ok
print('ATR-norm sharpe', r[m].mean() / r[m].std(ddof=1) * np.sqrt(252))
