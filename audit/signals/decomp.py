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


def runfn(sess, ctx, fn, cp, atr, ok, **kw):
    n = len(sess); pnl = np.zeros(n)
    for i in np.flatnonzero(ok):
        o, h, l, c, v, m = sess.day(i)
        pnl[i] = fn(o, h, l, c, v, m, i, atr, ctx, cp, **kw)
    return pnl


if __name__ == '__main__':
    mkt = sys.argv[1] if len(sys.argv) > 1 else 'nq'
    sess = build_sessions(mkt); ctx = ac.context(sess)
    atr = st.trailing_atr(ctx['ranges'], sess.roll)
    ok = ~sess.roll & ~np.isnan(atr)
    dev = sess.dates <= DEV_END
    cp = st.Costs().points(sess)
    fns = {
        'noise': lambda o, h, l, c, v, m, i, atr, ctx, cp, allow=0: ac.noise_area_session(o, h, l, c, v, m, ctx['prev_close'][i], ctx['sigma'][i], cp, allow=allow)[0],
        'vwap': lambda o, h, l, c, v, m, i, atr, ctx, cp, allow=0: st.vwap_atr_session(o, h, l, c, v, m, atr[i], 0.3, 0.2, cp, start_min=st.CASH_OPEN, allow=allow)[0],
    }
    for nm, fn in fns.items():
        for lab, allow in (('both', 0), ('long only', 1), ('short only', -1)):
            p = runfn(sess, ctx, fn, cp, atr, ok, allow=allow)
            r = np.where(ok, p / atr, 0)
            print(mkt, nm, lab.ljust(11), 'sharpe full %.2f dev %.2f hold %.2f' % (sh(r[ok]), sh(r[ok & dev]), sh(r[ok & ~dev])), ' mean pts/day %.2f' % (p[ok].mean()))
    # long-only buy open 9:30 -> close 15:55 benchmark (1 RT cost)
    p = np.zeros(len(sess))
    for i in np.flatnonzero(ok):
        o, h, l, c, v, m = sess.day(i)
        a = int(np.searchsorted(m, st.CASH_OPEN)); e = int(np.searchsorted(m, st.FLAT_BY))
        if e - a > 100:
            p[i] = c[e - 1] - o[a] - cp
    r = np.where(ok, p / atr, 0)
    print(mkt, 'BENCH long 09:30->15:55 each day: sharpe full %.2f dev %.2f hold %.2f' % (sh(r[ok]), sh(r[ok & dev]), sh(r[ok & ~dev])))
    # cost stress
    for lab, cst in [('slip 1t comm 2.5 (base)', st.Costs(2.5, 1)), ('slip 2t', st.Costs(2.5, 2)), ('slip 3t', st.Costs(2.5, 3)), ('slip 1t, comm 7.4 (10 micros @0.74)', st.Costs(7.4, 1)), ('slip 2t, comm 7.4', st.Costs(7.4, 2))]:
        out = []
        for name, pp in [('noise_area', {}), ('vwap_atr', dict(k=0.3, s=0.2, start_min=st.CASH_OPEN))]:
            d = ac.run_sleeve(sess, name, costs=cst, ctx=ctx, **pp)
            rr = np.where(d.traded_ok, d.pnl / (d.atr * d.mult), 0.0)
            out.append('%s %.2f/%.2f' % (name, sh(rr[d.traded_ok]), sh(rr[d.traded_ok & ~dev])))
        print(mkt, lab.ljust(38), '  '.join(out), ' (full/hold)')
