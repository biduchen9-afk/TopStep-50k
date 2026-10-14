import sys
sys.path.insert(0, '..')
import numpy as np, pandas as pd
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions
DEV_END = np.datetime64('2022-12-31')


def sh(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.size > 20 and x.std() > 0 else 0.0


sess = build_sessions('nq'); ctx = ac.context(sess)
early = np.array([sess.smin[sess.offsets[i + 1] - 1] < 21 * 60 + 50 for i in range(len(sess))])
print('early-close/holiday sessions:', early.sum())
for name, p in [('noise_area', {}), ('vwap_atr', dict(k=0.3, s=0.2, start_min=st.CASH_OPEN))]:
    d = ac.run_sleeve(sess, name, ctx=ctx, **p)
    r = np.where(d.traded_ok, d.pnl / (d.atr * d.mult), 0.0)
    ok = d.traded_ok
    ex = ok & ~early
    dev = sess.dates <= DEV_END
    print(name, 'all', round(sh(r[ok]), 3), ' excluding early-close sessions', round(sh(r[ex]), 3), ' (hold %.3f)' % sh(r[ex & ~dev]),
          ' sum normalised R on early-close days: %.2f of total %.2f' % (r[ok & early].sum(), r[ok].sum()), ' trades there', int(d.trades[ok & early].sum()))
