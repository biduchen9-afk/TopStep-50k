import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions

DEV_END = np.datetime64('2022-12-31')


def sh(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.size > 20 and x.std() > 0 else 0.0


if __name__ == '__main__':
    sess = build_sessions('nq')
    ctx = ac.context(sess)
    for name, p in [('noise_area', {}), ('vwap_atr', dict(k=0.3, s=0.2, start_min=st.CASH_OPEN))]:
        d = ac.run_sleeve(sess, name, ctx=ctx, **p)
        ok = d.traded_ok
        r = np.where(ok, d.pnl / (d.atr * d.mult), 0.0)
        dev = sess.dates <= DEV_END
        print(name, 'N ok', ok.sum(), 'full', round(sh(r[ok]), 3), 'dev', round(sh(r[ok & dev]), 3), 'hold', round(sh(r[ok & ~dev]), 3),
              'trades/day', d.trades[ok].mean().round(3), 'mean $/day/mini', d.pnl[ok].mean().round(2))
        yrs = pd.Series(r[ok], index=pd.DatetimeIndex(sess.dates[ok])).groupby(lambda t: t.year).agg(lambda x: sh(x.to_numpy()) if len(x) > 20 else np.nan)
        print(yrs.round(2).to_dict())
        # zero cost
        d0 = ac.run_sleeve(sess, name, costs=st.Costs(0.0, 0.0), ctx=ctx, **p)
        r0 = np.where(ok, d0.pnl / (d0.atr * d0.mult), 0.0)
        print('  zero cost sharpe', round(sh(r0[ok]), 3))
        # include roll days as zeros (denominator keeps all days with atr)
        atrok = ~np.isnan(d.atr)
        print('  roll sessions', sess.roll.sum(), ' include roll days as 0 pnl:', round(sh(r[atrok]), 3))
