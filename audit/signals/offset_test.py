"""Does the noise-area edge depend on decisions being exactly on the :00/:30 boundary? Shift the whole decision grid by `off` minutes."""
import sys
sys.path.insert(0, '..')
import numpy as np, pandas as pd
import intraday_strategies as st
from market_data import build_sessions
from load_raw import load

DEV_END = np.datetime64('2022-12-31')


def sh(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.size > 20 and x.std() > 0 else 0.0


src = open('replicate_noise.py').read().split("if __name__")[0]
src = src.replace("T = 570 + STEP * k", "T = 570 + STEP * k + OFF")
ns = {'OFF': 0}
exec(src, ns)
df = load('nq'); days = ns['build_days'](df); dates = sorted(days)
sess = build_sessions('nq'); ranges = st.session_ranges(sess); atr = st.trailing_atr(ranges, sess.roll)
A = pd.Series(atr, index=pd.DatetimeIndex(sess.dates)); ok = pd.Series(~sess.roll & ~np.isnan(atr), index=A.index)
cost = 2.5 / 20 + 0.25
for off in (0, 1, 3, 7, 13, 29):
    ns['OFF'] = off
    mv = np.array([ns['moves'](days[d]) for d in dates])
    out = {}
    for i, d in enumerate(dates):
        if i < 15:
            continue
        pc = days[dates[i - 1]]['c'][-1]; sg = np.nanmean(mv[i - 14:i], axis=0)
        p, tr = ns['noise_day'](days[d], pc, sg); out[d] = p - cost * len(tr)
    s = pd.Series(out); r = (s / A.reindex(s.index))[ok.reindex(s.index)]
    dev = r.index <= pd.Timestamp(DEV_END)
    print('decision grid offset +%2d min: Sharpe full %.3f dev %.3f hold %.3f' % (off, sh(r.to_numpy()), sh(r.to_numpy()[dev]), sh(r.to_numpy()[~dev])))
