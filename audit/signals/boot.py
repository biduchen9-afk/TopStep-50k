import sys
sys.path.insert(0, '..')
import numpy as np, pandas as pd
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions


def sh(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


sess = build_sessions('nq'); ctx = ac.context(sess)
rng = np.random.default_rng(1)
for name, p in [('noise_area', {}), ('vwap_atr', dict(k=0.3, s=0.2, start_min=st.CASH_OPEN))]:
    d = ac.run_sleeve(sess, name, ctx=ctx, **p)
    ok = d.traded_ok
    r = pd.Series(np.where(ok, d.pnl / (d.atr * d.mult), 0.0), index=pd.DatetimeIndex(sess.dates))[ok]
    # month-block bootstrap
    months = r.groupby([r.index.year, r.index.month])
    blocks = [g.to_numpy() for _, g in months]
    S = []
    for _ in range(3000):
        pick = rng.integers(0, len(blocks), len(blocks))
        S.append(sh(np.concatenate([blocks[k] for k in pick])))
    S = np.array(S)
    top = r.sort_values(ascending=False)
    print(name, 'Sharpe %.2f  month-block bootstrap 90%% CI [%.2f, %.2f]' % (sh(r.to_numpy()), *np.percentile(S, [5, 95])),
          ' top-10 days share of total R: %.0f%%' % (100 * top.head(10).sum() / r.sum()),
          ' Sharpe without top-10 days: %.2f' % sh(r.drop(top.head(10).index).to_numpy()))
    print('   P(Sharpe<=0) bootstrap: %.4f' % (S <= 0).mean())
