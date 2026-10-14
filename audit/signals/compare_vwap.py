import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions
from replicate_vwap import run_all

sess = build_sessions('nq')
dates, out = run_all(sess.roll)
assert set(pd.DatetimeIndex(dates)) == set(pd.DatetimeIndex(sess.dates))
d0 = ac.run_sleeve(sess, 'vwap_atr', costs=st.Costs(0.0, 0.0), k=0.3, s=0.2, start_min=st.CASH_OPEN)
proj = pd.Series(d0.pnl / sess.mult, index=pd.DatetimeIndex(sess.dates))
ptr = pd.Series(d0.trades, index=proj.index)
mine = pd.Series({d: r[0] for d, r in out.items()}); mtr = pd.Series({d: r[1] for d, r in out.items()})
J = pd.concat([mine.rename('mine'), proj.reindex(mine.index).rename('proj'), mtr.rename('ntr_mine'), ptr.reindex(mine.index).rename('ntr_proj')], axis=1, sort=True)
J['diff'] = J.mine - J.proj
bad = J[J['diff'].abs() > 1e-9]
print('sessions', len(J), 'mismatch', len(bad), 'trade-count mismatch', int((J.ntr_mine != J.ntr_proj).sum()))
print(bad.head(15).to_string())
print('sum pts mine', J.mine.sum(), 'proj', J.proj.sum())
rng = np.random.default_rng(777)
cand = J[J.index >= '2012-01-01']
pick = list(rng.choice(cand[cand.index >= '2023-01-01'].index.values, 6, replace=False)) + list(rng.choice(cand[cand.index < '2023-01-01'].index.values, 16, replace=False))
print(cand.loc[sorted(pick)][['mine', 'proj', 'diff', 'ntr_mine', 'ntr_proj']].round(3).to_string())
