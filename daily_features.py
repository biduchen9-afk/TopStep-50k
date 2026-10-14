"""Causal daily features for the regime filters and the intraday-only versions of daily systems (pre-registration 8).
A daily bar is a full session (18:00 ET open to 16:00 ET close). feats[name][i] is known BEFORE session i opens: it is computed from sessions <= i-1."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import Sessions


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _hurst(r: np.ndarray, w: int = 100) -> np.ndarray:
    out = np.full(len(r), np.nan)
    for t in range(w, len(r) + 1):
        x = r[t - w:t]
        if not np.all(np.isfinite(x)):
            continue
        sizes, rs = [], []
        for n in (10, 20, 25, 50):
            k = w // n; vals = []
            for j in range(k):
                seg = x[j * n:(j + 1) * n]; z = np.cumsum(seg - seg.mean()); s = seg.std()
                if s > 0:
                    vals.append((z.max() - z.min()) / s)
            if vals:
                sizes.append(np.log(n)); rs.append(np.log(np.mean(vals)))
        if len(sizes) >= 3:
            out[t - 1] = np.polyfit(sizes, rs, 1)[0]
    return out


def build_daily(sess: Sessions, ranges: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(sess)
    o, h, l, c = (np.zeros(n) for _ in range(4))
    for i in range(n):
        oo, hh, ll, cc, v, m = sess.day(i)
        o[i], h[i], l[i], c[i] = oo[0], hh.max(), ll.min(), cc[-1]
    O, H, L, C = (pd.Series(x) for x in (o, h, l, c))
    atr = pd.Series(st.trailing_atr(ranges, sess.roll))          # same range-ATR used for stops everywhere (already uses sessions < i)
    atr_c = atr.shift(-1)                                          # ATR computed THROUGH session i (so that shift(1) below aligns it with the other features)
    F: Dict[str, pd.Series] = {}
    ret = C.pct_change(); lr = np.log(C).diff()
    for span in (12, 20, 26, 50, 100, 200):
        F[f"ema{span}"] = C.ewm(span=span, adjust=False).mean()
    for w in (20, 50, 200):
        F[f"sma{w}"] = C.rolling(w).mean()
    F["sma200_up"] = (F["sma200"] > F["sma200"].shift(5)).astype(float)
    F["ema20_up"] = (F["ema20"] > F["ema20"].shift(5)).astype(float); F["ema50_up"] = (F["ema50"] > F["ema50"].shift(5)).astype(float); F["ema200_up"] = (F["ema200"] > F["ema200"].shift(5)).astype(float)
    F["ema100_slope"] = F["ema100"] - F["ema100"].shift(5); F["sma50_slope"] = F["sma50"] - F["sma50"].shift(5)
    pc = C.shift(1)
    tr = pd.concat([H - L, (H - pc).abs(), (L - pc).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up_m, dn_m = H.diff(), -L.diff()
    pdm = pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)); mdm = pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0))
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr_w; mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    F["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean(); F["adx_rising"] = (F["adx"] > F["adx"].shift(3)).astype(float)
    F["er20"] = (C - C.shift(20)).abs() / C.diff().abs().rolling(20).sum()
    F["rsi2"], F["rsi14"] = _rsi(C, 2), _rsi(C, 14)
    sd20 = C.rolling(20).std(); mid20 = F["sma20"]
    F["bb_up2"], F["bb_dn2"], F["bb_up25"], F["bb_dn25"] = mid20 + 2 * sd20, mid20 - 2 * sd20, mid20 + 2.5 * sd20, mid20 - 2.5 * sd20
    F["bbw"] = (F["bb_up2"] - F["bb_dn2"]) / mid20
    ema20 = F["ema20"]; kel_up, kel_dn = ema20 + 1.5 * atr_c, ema20 - 1.5 * atr_c
    sq = ((F["bb_up2"] < kel_up) & (F["bb_dn2"] > kel_dn)).astype(float)
    run = sq.groupby((sq != sq.shift()).cumsum()).cumsum() * sq
    F["squeeze_run"] = run
    F["atr_pct"] = atr_c.rolling(252).apply(lambda x: (x[:-1] < x[-1]).mean() if np.isfinite(x[-1]) else np.nan, raw=True)
    F["roc5"], F["roc10"], F["roc20"] = C / C.shift(5) - 1, C / C.shift(10) - 1, C / C.shift(20) - 1
    for w in (10, 20):
        F[f"hh{w}"] = H.rolling(w).max().shift(1); F[f"ll{w}"] = L.rolling(w).min().shift(1)       # the w sessions BEFORE the signal session
    F["hv20"] = lr.rolling(20).std(); F["hv_pct"] = F["hv20"].rolling(252).apply(lambda x: (x[:-1] < x[-1]).mean() if np.isfinite(x[-1]) else np.nan, raw=True)
    rng = H - L
    F["nr7"] = (rng <= rng.rolling(7).min()).astype(float); F["nr4"] = (rng <= rng.rolling(4).min()).astype(float)
    sgn = np.sign(C.diff()); grp = (sgn != sgn.shift()).cumsum(); F["streak"] = sgn.groupby(grp).cumcount() + 1; F["streak_sign"] = sgn
    macd = F["ema12"] - F["ema26"]; F["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    F["hurst"] = pd.Series(_hurst(lr.to_numpy()))
    r1 = lr; var1 = r1.rolling(100).var(); var5 = np.log(C / C.shift(5)).rolling(100).var()
    F["vr5"] = var5 / (5 * var1)
    F["shock"] = ret / ret.rolling(60).std()
    F["close"], F["open"], F["high"], F["low"], F["range"] = C, O, H, L, rng
    F["close_loc"] = (C - L) / rng.replace(0, np.nan)
    F["atr_c"] = atr_c
    F["close5"] = C.shift(5)
    out = {k: v.shift(1).to_numpy(dtype=float) for k, v in F.items()}        # known before session i = value through session i-1
    out["_dates"] = sess.dates
    out["_H"], out["_L"], out["_C"] = h, l, c
    return out
