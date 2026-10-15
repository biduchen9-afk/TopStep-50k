import sys
sys.path.insert(0, '..')
import numpy as np, pandas as pd
from market_data import build_sessions
import replicate_vwap_delay as rv
import replicate_noise as rn
DEV_END = np.datetime64('2022-12-31')
def sh(x): return float(x.mean()/x.std(ddof=1)*np.sqrt(252)) if x.size>20 and x.std()>0 else 0.0
sess = build_sessions('nq')
atr_cache = {}
cost = 2.5/20 + 0.25
for dly in (0, 1, 2):
    dates, out = rv.run_all(sess.roll, delay=dly)
    # atr per session
    days = None
    s = pd.Series({d: r[0] - cost*r[1] for d, r in out.items()})
    # normalise by atr (recompute from project's atr, identical to replication)
    import intraday_strategies as st
    ranges = st.session_ranges(sess); atr = st.trailing_atr(ranges, sess.roll)
    A = pd.Series(atr, index=pd.DatetimeIndex(sess.dates))
    r = (s / A.reindex(s.index))
    dev = r.index <= pd.Timestamp(DEV_END)
    print('vwap_atr entry/exit delay %d min: Sharpe full %.3f dev %.3f hold %.3f' % (dly, sh(r.to_numpy()), sh(r.to_numpy()[dev]), sh(r.to_numpy()[~dev])))
