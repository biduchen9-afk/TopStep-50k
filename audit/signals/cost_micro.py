import sys
sys.path.insert(0, '..')
import numpy as np
import academic_strategies as ac, intraday_strategies as st
from market_data import build_sessions


def sh(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


sess = build_sessions('nq'); ctx = ac.context(sess); dev = sess.dates <= np.datetime64('2022-12-31')
for lab, c in [('base 2.50 + 1t', st.Costs(2.5, 1)), ('mini all-in 3.78 + 1t', st.Costs(3.78, 1)),
               ('MNQ all-in 1.22*10=12.2 + 1t', st.Costs(12.2, 1)), ('MNQ 12.2 + 2t', st.Costs(12.2, 2))]:
    out = []
    for n, p in [('noise_area', {}), ('vwap_atr', dict(k=0.3, s=0.2, start_min=st.CASH_OPEN))]:
        d = ac.run_sleeve(sess, n, costs=c, ctx=ctx, **p); ok = d.traded_ok; r = np.where(ok, d.pnl / (d.atr * d.mult), 0)
        out.append('%s full %.2f hold %.2f' % (n, sh(r[ok]), sh(r[ok & ~dev])))
    print(lab.ljust(32), ' | '.join(out))
