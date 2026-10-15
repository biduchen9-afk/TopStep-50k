from common import *
from lib import stats
import academic_strategies as ac, intraday_strategies as st
from market_data import build_sessions
from real_engine import Stream
sess = build_sessions("nq"); ctx = ac.context(sess)
DEV = sess.dates <= DEV_END
n_h = int(np.searchsorted(sess.dates, DEV_END, side="right"))
spec = FarmSpec(max_evals=20, max_funded=5, p_est=1.0); offs = range(n_h, n_h + 60, 10)
starts = np.arange(n_h, len(sess) - 300); q = pd.DatetimeIndex(sess.dates[starts]); grp = (q.year * 4 + (q.month - 1) // 3).to_numpy()
for comm, slip in ((2.5, 1.0), (2.5, 2.0), (7.4, 2.0), (7.4, 3.0), (7.4, 4.0)):
    d = ac.run_sleeve(sess, "noise_area", costs=st.Costs(commission_rt=comm, slippage_ticks_rt=slip), ctx=ctx)
    s = Stream.from_series("x", d)
    res = pd.DataFrame([run_farm([s], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, o, spec) for o in offs])
    mm = contract_metrics(s, s, SZ, plan(0.4, 0.4), None, cfg, rules, starts, grp)
    print(f"comm/mini RT ${comm} slippage {slip} ticks RT: Sharpe(ATR) full {stats(d)['sh_atr']:.3f} dev {stats(d, DEV)['sh_atr']:.3f} hold {stats(d, ~DEV)['sh_atr']:.3f} | farm hold-out net {res.net.mean():,.0f} | ev_att {mm['ev_att']:.0f}")
