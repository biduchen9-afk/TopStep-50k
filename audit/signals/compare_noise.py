import sys, pickle
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions

SCR = '/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad'
dates, res = pickle.load(open(f'{SCR}/my_noise_nq.pkl', 'rb'))
sess = build_sessions('nq')
d0 = ac.run_sleeve(sess, 'noise_area', costs=st.Costs(0.0, 0.0))
proj_pts = pd.Series(d0.pnl / sess.mult, index=pd.DatetimeIndex(sess.dates))
ok = pd.Series(d0.traded_ok, index=proj_pts.index)
roll = pd.Series(sess.roll, index=proj_pts.index)
mine = pd.Series({d: r[0] for d, r in res.items()})
ntr = pd.Series({d: len(r[1]) for d, r in res.items()})
print('my sessions', len(dates), 'project sessions', len(proj_pts), 'same dates:', set(pd.DatetimeIndex(dates)) == set(proj_pts.index))
J = pd.concat([mine.rename('mine'), proj_pts.rename('proj'), ok.rename('ok'), roll.rename('roll'), ntr.rename('ntr_mine'), pd.Series(d0.trades, index=proj_pts.index).rename('ntr_proj')], axis=1)
J = J.dropna(subset=['mine', 'proj'])
J['diff'] = J.mine - J.proj
okd = J[J.ok]
print('sessions compared (project ok):', len(okd), ' exact matches (<1e-9):', int((okd['diff'].abs() < 1e-9).sum()), ' mismatches:', int((okd['diff'].abs() >= 1e-9).sum()))
print('trade count mismatches:', int((okd.ntr_mine != okd.ntr_proj).sum()))
print('max |diff| pts', okd['diff'].abs().max())
mm = okd[okd['diff'].abs() >= 1e-9]
print(mm.head(20).to_string())
print('sum of my pnl pts (ok days):', okd.mine.sum(), ' project:', okd.proj.sum())
# random sample, >=20 across 2012-2026, >=5 in 2023-26
rng = np.random.default_rng(12345)
cand = okd[(okd.index >= '2012-01-01')]
hold = cand[cand.index >= '2023-01-01']
pick = list(rng.choice(hold.index.values, 8, replace=False)) + list(rng.choice(cand[cand.index < '2023-01-01'].index.values, 20, replace=False))
S = cand.loc[sorted(pick)]
print('\nRANDOM SAMPLE (28 sessions):')
print(S[['mine', 'proj', 'diff', 'ntr_mine', 'ntr_proj']].round(3).to_string())
print('traded sessions in sample:', int((S.ntr_proj > 0).sum()), ' max |diff|:', S['diff'].abs().max())
print('\nflagged roll sessions (project skipped): compare my pnl there:')
R = J[J.roll]
print(len(R), 'sum my pts on roll sessions', R.mine.sum(), 'abs', R.mine.abs().sum())
