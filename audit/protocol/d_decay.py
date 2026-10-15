"""Audit: NQ noise-area / vwap_atr performance by sub-period, long vs short, and a drift check."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions

sess = build_sessions("nq"); ctx = ac.context(sess)
cp = st.Costs().points(sess)
atr = st.trailing_atr(ctx["ranges"], sess.roll)
n = len(sess); ok = ~sess.roll & ~np.isnan(atr)
dates = pd.DatetimeIndex(sess.dates)


def sh(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(252) if len(x) > 20 and x.std() > 0 else np.nan


def run(allow):
    r = np.zeros(n)
    for i in np.flatnonzero(ok):
        o, h, l, c, v, m = sess.day(i)
        p = ac.noise_area_session(o, h, l, c, v, m, ctx["prev_close"][i], ctx["sigma"][i], cp, allow=allow)
        r[i] = p[0] / atr[i]
    return pd.Series(r, index=dates)[ok]


both, lng, sht = run(0), run(1), run(-1)
# daily NQ open->close drift (09:30 -> 15:55) for scale
print("Sharpe (ATR-norm) noise_area: both / long-only / short-only")
for lab, a, b in (("dev 2010-2022", "2010", "2022"), ("2023-2024", "2023", "2024"), ("2025-2026", "2025", "2026"), ("hold 2023-2026", "2023", "2026")):
    print(f"  {lab:<16} both {sh(both[a:b]):5.2f}  long {sh(lng[a:b]):5.2f}  short {sh(sht[a:b]):5.2f}   n={len(both[a:b])}  mean R/day both {both[a:b].mean():.4f}")
x = both["2025":"2026"]
print("2025-2026 t-stat", round(sh(x) * np.sqrt(len(x) / 252), 2))
x = both["2023":"2026"]
print("hold-out t-stat", round(sh(x) * np.sqrt(len(x) / 252), 2))
x = both[:"2022"]
print("dev t-stat", round(sh(x) * np.sqrt(len(x) / 252), 2))
# rolling 1y Sharpe at the end
print("trailing 252d Sharpe at end:", round(sh(both.iloc[-252:]), 2), " last 126d:", round(sh(both.iloc[-126:]), 2))
