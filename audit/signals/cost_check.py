import sys
sys.path.insert(0, '..')
import numpy as np
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions
from portfolio import make_sleeve
for mkt in ('nq', 'es', 'gc'):
    sess = build_sessions(mkt); ctx = ac.context(sess)
    cp = st.Costs().points(sess)
    for name, p in [('noise_area', {}), ('vwap_atr', dict(k=0.3, s=0.2, start_min=st.CASH_OPEN)), ('orb', dict(or_min=30, buf=0.05, s=0.2)),
                    ('gap', dict(mode=1, thr=0.2, s=0.6)), ('intraday_mom', dict(use_r12=True, thr=0.0)), ('vwap_fade', dict(k=0.3, s=0.3))]:
        d = ac.run_sleeve(sess, name, ctx=ctx, **p)
        d0 = ac.run_sleeve(sess, name, costs=st.Costs(0, 0), ctx=ctx, **p)
        diff = (d0.pnl - d.pnl) / sess.mult
        exp = d.trades * cp
        # minpnl: should be <= net pnl and <= 0
        viol = int(((d.minpnl > d.pnl + 1e-9) | (d.minpnl > 1e-9)).sum())
        # minpnl vs zero-cost minpnl: cost deduction
        print(mkt, name.ljust(13), 'cost applied == trades*cp on all days:', bool(np.allclose(diff, exp, atol=1e-9)), ' minpnl violations:', viol,
              ' mean (minpnl0-minpnl)/cp per trade: %.2f' % (((d0.minpnl - d.minpnl) / sess.mult)[d.trades > 0] / cp).mean())
# micro conversion
s = make_sleeve('nq', 'noise_area', {})
sess = build_sessions('nq'); d = ac.run_sleeve(sess, 'noise_area')
print('micro conversion pnl ratio mini->micro:', np.nanmax(np.abs(s.pnl - d.pnl * np.where(d.traded_ok, 0.1, 0))), 'atr_usd example', np.nanmedian(s.atr))
print('NQ cost per RT: mini $%.2f, micro $%.2f (a real MNQ round turn is typically >= $0.5-0.75 commission+fees + slippage $0.50)' % (st.Costs().points(sess) * 20, st.Costs().points(sess) * 2))
