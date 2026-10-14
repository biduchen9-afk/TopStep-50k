"""Does sizing by ATR (inverse-volatility) hurt the Combine in high-vol years? Combine pass rate by start-year for several risk units."""
import numpy as np, pandas as pd
from run_corrected import NEW, streams, rules
from real_engine import run_combine
from risk_units import with_risk_unit
pd.set_option("display.width", 250)
S = streams(NEW)["NQ noise-area"]
f = pd.read_pickle("/home/user/data_raw/regime_features.pkl")
p = pd.Series(S.pnl, index=pd.DatetimeIndex(S.dates)); p = p[S.ok]
top = p.groupby(p.index.year).apply(lambda s: s.nlargest(5).sum() / s.sum() if s.sum() > 0 else np.nan)
print("share of a year's profit from its 5 best days:", top.tail(6).round(2).to_dict())
d = pd.DatetimeIndex(S.dates)
variants = {"ATR R=6400": ("atr", 40, 6400), "ATR R=3200": ("atr", 40, 3200), "pnlstd40 R=6400": ("pnlstd", 40, 6400), "pnlstd40 R=3200": ("pnlstd", 40, 3200),
            "pnlstd80 R=3200": ("pnlstd", 80, 3200), "blend40 R=3200": ("blend", 40, 3200)}
out = {}
for name, (kind, w, R) in variants.items():
    s = with_risk_unit(S, kind, w)
    nd = s.desired_micros(R); x = s.pnl.tolist(); lo = s.low.tolist()
    first = int(np.flatnonzero(s.ok)[0]) + 60
    res = []
    for i0 in range(first, len(d) - 260, 5):
        st, tau = run_combine(x, lo, nd, i0, rules)
        res.append((d[i0].year, st == 1, tau))
    r = pd.DataFrame(res, columns=["y", "p", "tau"])
    out[name] = r.groupby("y").p.mean()
    out[name]["ALL"] = r.p.mean(); out[name]["dev"] = r[r.y <= 2022].p.mean(); out[name]["2023+"] = r[r.y >= 2023].p.mean()
print(pd.DataFrame(out).round(2).T.to_string())
