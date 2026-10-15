"""Pre-registration 2, Part B: turn-of-month long sleeve (see docs/PREREGISTRATION_2.md). Two tests only: NQ and ES."""
from __future__ import annotations
import numpy as np, pandas as pd
import intraday_strategies as st
from combine_policy import CombinePolicy
from final_strategy import build_stream
from market_data import build_sessions
from portfolio import DEV_END, Sleeve, basket, dev_weights, make_sleeve, r_series, sharpe
from real_engine import MICRO_PER_MINI, costs_from_config
from pricing_config import load_config
from run_funded_payout_policy import farm_plan
from run_regime_study import windows
from real_engine import FundedPlan

pd.set_option("display.width", 250)


def tom_sleeve(market: str) -> Sleeve:
    sess = build_sessions(market); costs = costs_from_config(load_config()); cp = costs.points(sess)
    ranges = st.session_ranges(sess); atr = st.trailing_atr(ranges, sess.roll)
    d = pd.DatetimeIndex(sess.dates); ym = d.year * 12 + d.month
    n = len(sess); pnl, low = np.zeros(n), np.zeros(n); on = np.zeros(n, bool)
    for i in range(n):
        last_of_month = i + 1 < n and ym[i + 1] != ym[i]
        first3 = sum(1 for j in range(max(0, i - 2), i + 1) if ym[j] == ym[i]) == (i - np.searchsorted(ym, ym[i]) + 1) and (i - np.searchsorted(ym, ym[i])) < 3
        on[i] = last_of_month or first3
    ok = ~sess.roll & ~np.isnan(atr)
    for i in np.flatnonzero(ok & on):
        o, h, l, c, v, m = sess.day(i)
        p, w, _ = st.random_session(o, h, l, c, v, m, 1, cp)
        pnl[i], low[i] = p * sess.mult, w * sess.mult
    k = 1.0 / MICRO_PER_MINI
    act = ok & on
    return Sleeve(f"{market}_tom", sess.dates, np.where(act, pnl * k, 0.0), np.where(act, low * k, 0.0), np.where(ok, atr * sess.mult * k, np.nan), act, np.full(n, np.nan))


def main() -> None:
    base = build_stream(); rb = r_series(base); dev = base.dates <= DEV_END
    res = []
    for mkt in ("nq", "es"):
        sl = tom_sleeve(mkt)
        b = basket([sl], name=sl.name); r = r_series(b)
        ser_b = pd.Series(rb, index=pd.DatetimeIndex(base.dates)); ser_t = pd.Series(r, index=pd.DatetimeIndex(b.dates)).reindex(ser_b.index).fillna(0.0)
        dm = ser_b.index <= pd.Timestamp(DEV_END)
        corr = float(np.corrcoef(ser_b[dm], ser_t[dm])[0, 1])
        okk = b.ok; dv = b.dates <= DEV_END
        sh_dev, sh_hold = sharpe(r[okk & dv]), sharpe(r[okk & ~dv])
        sh_dev_all = sharpe(r[dv & (b.dates >= b.dates[np.flatnonzero(okk)[0]])])      # incl. non-TOM zero days
        res.append(dict(market=mkt, active_days=int(okk.sum()), sharpe_dev_TOM_days=sh_dev, sharpe_dev_all_days=sh_dev_all, sharpe_hold_TOM_days=sh_hold, corr_dev_with_basket=corr))
    g = pd.DataFrame(res); print(g.round(3).to_string(index=False)); g.to_csv("results/tom_sleeve.csv", index=False)
    print("\nAcceptance rule: Sharpe(all days, dev) >= 0.5 AND corr < 0.3 AND dev farm objective not lower. (Farm test only if the first two pass.)")


if __name__ == "__main__":
    main()
