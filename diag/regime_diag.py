"""Why did the NQ noise-area edge hold in 2023-24 and fade in 2025-26? Look at what the strategy lives on:
intraday trendiness (variance ratio / return autocorrelation of 30-min bars), volatility level, and gap size."""
import numpy as np, pandas as pd
from market_data import build_sessions
from intraday_strategies import CASH_OPEN, FLAT_BY
from run_corrected import NEW, streams
pd.set_option("display.width", 250)
S = streams(NEW)["NQ noise-area"]
sess = build_sessions("nq")
n = len(sess)
rows = []
for i in range(n):
    o, h, l, c, v, m = sess.day(i)
    a = int(np.searchsorted(m, CASH_OPEN)); e = int(np.searchsorted(m, FLAT_BY))
    if e - a < 300:
        rows.append((np.nan,) * 6); continue
    px = np.r_[o[a], c[a + 29:e:30]]                      # 30-minute marks
    r = np.diff(np.log(px))
    net = np.log(px[-1] / px[0])
    rows.append((net, (r ** 2).sum(), (r[1:] * r[:-1]).sum(), np.abs(r).sum(), np.log(c[a - 1] / (o[0])) if a > 0 else np.nan, h[a:e].max() / l[a:e].min() - 1))
f = pd.DataFrame(rows, columns=["net", "ss", "cross", "absum", "gap", "rng"], index=pd.DatetimeIndex(sess.dates))
f["R"] = pd.Series(np.where(S.ok, S.pnl / np.where(S.atr_usd > 0, S.atr_usd, np.nan), np.nan), index=pd.DatetimeIndex(S.dates)).reindex(f.index)
f["pnl"] = pd.Series(S.pnl, index=pd.DatetimeIndex(S.dates)).reindex(f.index)
f["atr_usd"] = pd.Series(S.atr_usd, index=pd.DatetimeIndex(S.dates)).reindex(f.index)
g = f.groupby(f.index.year)
y = pd.DataFrame({
    "R_sharpe": g.R.mean() / g.R.std() * np.sqrt(252),
    "raw_$/micro_day": g.pnl.mean(),
    "atr_$/micro": g.atr_usd.mean(),
    "range_%": g.rng.mean() * 100,
    "VR (net^2/sumsq)": g.apply(lambda d: (d.net ** 2).mean() / d.ss.mean()),
    "lag1 autocorr": g.apply(lambda d: d.cross.sum() / d.ss.sum()),
    "|gap| % ": g.gap.apply(lambda s: s.abs().mean() * 100),
})
print(y.round(2).to_string())
# is trailing trendiness predictive of next-day strategy return?  (strictly past information)
for w in (20, 60, 120):
    vr = (f.net ** 2).rolling(w, min_periods=w // 2).mean() / f.ss.rolling(w, min_periods=w // 2).mean()
    ac = f.cross.rolling(w, min_periods=w // 2).sum() / f.ss.rolling(w, min_periods=w // 2).sum()
    rng = f.rng.rolling(w, min_periods=w // 2).mean()
    ok = f.R.notna()
    for nm, sgn in (("trailing VR", vr), ("trailing autocorr", ac), ("trailing range", rng)):
        x = sgn.shift(1)[ok]; r = f.R[ok]
        q = pd.qcut(x, 3, labels=["low", "mid", "high"], duplicates="drop")
        sh = r.groupby(q, observed=True).apply(lambda s: s.mean() / s.std() * np.sqrt(252))
        print(f"w={w:<3} {nm:<18} corr(next R)={np.corrcoef(x.dropna(), r[x.notna()])[0,1]:+.3f}   Sharpe by tercile low/mid/high: {sh.round(2).tolist()}")
# dollar sizing check: the strategy's R-units vs. raw dollars per fixed micro size
print("\nraw $ per 1 micro per day by year (what a fixed-size account earns):")
print(g.pnl.agg(["mean", "std"]).assign(sharpe=lambda d: d["mean"] / d["std"] * np.sqrt(252)).round(2).tail(8).to_string())
f.to_pickle("/home/user/data_raw/regime_features.pkl")
