"""Sleeves for pre-registration 8 (the user's strategy list). Regular trading hours only: entries >= 09:30 ET, exit <= 15:55 ET.
`F` = daily features known before the session (daily_features.py, plus raw daily arrays _H/_L/_C that are only ever indexed at sessions <= i-1)."""
from __future__ import annotations

from typing import Dict

import numpy as np

import academic_strategies as ac
import intraday_strategies as st
from intraday_strategies import CASH_OPEN, FLAT_BY, Costs, DailySeries, _first_true
from lit_sleeves import M, _hold
from market_data import Sessions

I = lambda m, x: int(np.searchsorted(m, x))
_sg = lambda x: 0 if not np.isfinite(x) else int(np.sign(x))
BLACKOUT = [(M(9, 55), M(10, 10)), (M(13, 55), M(14, 15))]


def prepass(sess: Sessions) -> Dict[str, np.ndarray]:
    """Strictly-past (previous 14 sessions) means of the 09:30 15-minute range, the 15:25-15:55 range and the average 60-minute range, plus the 30-minute-bucket volatility profile."""
    n = len(sess); f15, f30, h60 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    prof = np.full((n, 12), np.nan)
    for i in range(n):
        o, h, l, c, v, m = sess.day(i)
        a, b = I(m, CASH_OPEN), I(m, CASH_OPEN + 15)
        if b - a >= 5:
            f15[i] = h[a:b].max() - l[a:b].min()
        a2, b2 = I(m, M(15, 25)), I(m, FLAT_BY)
        if b2 - a2 >= 5:
            f30[i] = h[a2:b2].max() - l[a2:b2].min()
        rs = []
        for k in range(6):
            x, y = I(m, CASH_OPEN + 60 * k), I(m, CASH_OPEN + 60 * (k + 1))
            if y - x >= 20:
                rs.append(h[x:y].max() - l[x:y].min())
        if rs:
            h60[i] = np.mean(rs)
        for k in range(12):
            x, y = I(m, CASH_OPEN + 30 * k), I(m, CASH_OPEN + 30 * (k + 1))
            if y - x >= 20:
                seg = c[x:y]; step = 5
                if len(seg) > step:
                    prof[i, k] = np.mean(np.abs(np.diff(seg[::step]) / seg[:-step:step]))
    tm = lambda a: np.array([np.nanmean(a[max(0, i - 14):i]) if i > 3 else np.nan for i in range(n)])
    pr = np.full((n, 12), np.nan)
    for i in range(n):
        if i > 10:
            pr[i] = np.nanmean(prof[max(0, i - 90):i], axis=0)
    return dict(f15=tm(f15), f30=tm(f30), h60=tm(h60), prof=pr)


def hourly_state(sess: Sessions) -> Dict[str, np.ndarray]:
    """Hourly (09:30-based, 60-minute) bars: EMA50 and stochastic(14) computed across sessions, value after each of the first 5 hourly bars of every session."""
    n = len(sess); ema = np.full((n, 5), np.nan); stoch = np.full((n, 5), np.nan)
    e = None; hist_h, hist_l, hist_c = [], [], []
    for i in range(n):
        o, h, l, c, v, m = sess.day(i)
        for k in range(6):
            x, y = I(m, CASH_OPEN + 60 * k), I(m, CASH_OPEN + 60 * (k + 1))
            if y - x < 20:
                continue
            cl, hh, ll = c[y - 1], h[x:y].max(), l[x:y].min()
            e = cl if e is None else e + (2 / 51) * (cl - e)
            hist_h.append(hh); hist_l.append(ll); hist_c.append(cl)
            if k < 5:
                ema[i, k] = e
                if len(hist_c) >= 14:
                    hi, lo = max(hist_h[-14:]), min(hist_l[-14:])
                    stoch[i, k] = 100 * (cl - lo) / (hi - lo) if hi > lo else 50.0
    return dict(ema=ema, stoch=stoch)


def _break(o, h, l, c, m, d_allowed, up, dn, t0, end, stop_fn, cp, target=None):
    iu = _first_true(c[t0:end - 1] > up) if d_allowed >= 0 else -1
    idn = _first_true(c[t0:end - 1] < dn) if d_allowed <= 0 else -1
    if iu < 0 and idn < 0:
        return 0.0, 0.0, 0
    d, sig = (1, t0 + iu) if (idn < 0 or (iu >= 0 and iu < idn)) else (-1, t0 + idn)
    e = sig + 1
    sd = stop_fn(o[e], d)
    if sd is None or sd <= 0:
        return 0.0, 0.0, 0
    return _hold(o, h, l, c, m, d, e, end, sd, cp, target)


def daily_signal(name: str, F: Dict[str, np.ndarray], i: int, A: float):
    """(kind, direction, params) for the intraday-only versions of the daily systems; signals use data through session i-1."""
    g = lambda k: F[k][i]
    close, hi, lo, op, rg = g("close"), g("high"), g("low"), g("open"), g("range")
    if not (np.isfinite(close) and np.isfinite(A)) or i < 30:
        return None
    adx, j = g("adx"), i - 1
    HH, LL, CC = F["_H"], F["_L"], F["_C"]
    if name == "b1":
        if adx > 25 and np.isfinite(g("ema50")):
            return ("hold", _sg((g("ema20") - g("ema50"))), dict(stop=2.0))
    elif name == "b2":
        if close > g("hh20") and close > g("sma200"):
            return ("hold", 1, dict(stop=2.0))
        if close < g("ll20") and close < g("sma200"):
            return ("hold", -1, dict(stop=2.0))
    elif name == "b4":
        if close > g("hh10") + 0.25 * A and g("roc20") > 0:
            return ("hold", 1, dict(stop=2.0))
        if close < g("ll10") - 0.25 * A and g("roc20") < 0:
            return ("hold", -1, dict(stop=2.0))
    elif name == "b5":
        up = g("ema20") > g("ema50") > g("ema200") and g("ema20_up") and g("ema50_up") and g("ema200_up")
        dn = g("ema20") < g("ema50") < g("ema200") and not g("ema20_up") and not g("ema50_up") and not g("ema200_up")
        if up and lo <= g("ema20") and close > op and g("close_loc") >= 2 / 3:
            return ("hold", 1, dict(stop=2.0))
        if dn and hi >= g("ema20") and close < op and g("close_loc") <= 1 / 3:
            return ("hold", -1, dict(stop=2.0))
    elif name == "b6":
        if g("er20") > 0.35:
            d = _sg((g("ema20") - g("ema50")))
            return ("hold", d, dict(stop=2.0)) if d else None
    elif name == "b8":
        if F["squeeze_run"][i - 1] >= 6 and g("adx_rising"):
            if close > g("bb_up2"):
                return ("hold", 1, dict(stop=2.0))
            if close < g("bb_dn2"):
                return ("hold", -1, dict(stop=2.0))
    elif name == "b9":
        if g("nr7"):
            d = _sg((g("roc20")))
            if d != 0:
                return ("break", d, dict(up=hi, dn=lo, cap=2.0))
    elif name == "b10":
        if np.isfinite(g("atr_pct")) and g("atr_pct") < 0.40:
            if close > g("hh20"):
                return ("hold", 1, dict(stop=2.0))
            if close < g("ll20"):
                return ("hold", -1, dict(stop=2.0))
    elif name == "b12":
        p_hi, p_lo = max(HH[j - 2], HH[j - 1]), min(LL[j - 2], LL[j - 1])
        half = (max(HH[j], p_hi) - min(LL[j], p_lo)) / 2
        if HH[j] > p_hi and CC[j] < p_hi and g("rsi14") > 65:
            return ("hold", -1, dict(stop=2.0, target=half))
        if LL[j] < p_lo and CC[j] > p_lo and g("rsi14") < 35:
            return ("hold", 1, dict(stop=2.0, target=half))
    elif name == "b13":
        if close > HH[j - 1] and g("close_loc") >= 0.75 and g("roc20") > 0:
            return ("hold", 1, dict(stop_mid=(hi + lo) / 2))
        if close < LL[j - 1] and g("close_loc") <= 0.25 and g("roc20") < 0:
            return ("hold", -1, dict(stop_mid=(hi + lo) / 2))
    elif name == "b14":
        bh, bl = np.nanmax(HH[j - 10:j]), np.nanmin(LL[j - 10:j])
        if bh - bl < 1.5 * A:
            if CC[j] > bh:
                return ("hold", 1, dict(stop_mid=(bh + bl) / 2))
            if CC[j] < bl:
                return ("hold", -1, dict(stop_mid=(bh + bl) / 2))
    elif name == "b15":
        if adx < 20:
            if close > g("bb_up25") and g("rsi14") > 75:
                return ("hold", -1, dict(stop=1.2, target=abs(close - g("sma20"))))
            if close < g("bb_dn25") and g("rsi14") < 25:
                return ("hold", 1, dict(stop=1.2, target=abs(close - g("sma20"))))
    elif name == "b16":
        if close > g("sma200") and g("sma200_up") and g("rsi2") < 10:
            return ("hold", 1, dict(stop=2.0))
    elif name == "b19":
        z = (close - g("sma20")) / A
        if abs(z) >= 2 and adx < 25:
            return ("hold", -_sg((z)), dict(stop=1.5, target=abs(close - g("sma20"))))
    elif name == "b20":
        if g("streak") >= 5 and g("streak_sign") > 0 and g("rsi14") > 75:
            return ("hold", -1, dict(stop=1.0))
        if g("streak") >= 5 and g("streak_sign") < 0 and g("rsi14") < 25:
            return ("hold", 1, dict(stop=1.0))
    elif name == "b23":
        if rg > 1.5 * A:
            d = _sg((close - op))
            return ("hold", d, dict(stop=1.0)) if d else None
    elif name == "b25":
        bw = F["bbw"]; w120 = np.nanmin(bw[max(0, i - 120):i])
        if np.isfinite(w120) and bw[i] > bw[i - 1] and bw[i - 1] <= w120 * 1.15:
            d = _sg((g("macd_hist")))
            return ("hold", d, dict(stop=2.0)) if d else None
    elif name == "b26":
        if rg > 1.5 * A:
            d = _sg((close - op))
            if d:
                return ("inside", d, dict(mid=(hi + lo) / 2, hi=hi, lo=lo))
    elif name == "b27":
        ma = np.nanmean(CC[j - 10:j])
        if close > ma + A:
            return ("hold", 1, dict(stop=2.0))
        if close < ma - A:
            return ("hold", -1, dict(stop=2.0))
    elif name == "b37":
        if adx >= 25:
            d = _sg((close - g("sma50")))
            return ("hold", d, dict(stop=2.0)) if d else None
        if adx < 20:
            d = -_sg((close - g("sma20")))
            return ("hold", d, dict(stop=1.5)) if d else None
    elif name == "b38":
        sl, bw = g("sma50_slope"), F["bbw"]
        if sl > 0 and bw[i] > bw[i - 3]:
            return ("hold", 1, dict(stop=2.0))
        if sl < 0 and bw[i] > bw[i - 3]:
            return ("hold", -1, dict(stop=2.0))
        if abs(sl) < 0.1 * A and bw[i] < bw[i - 3]:
            d = -_sg((close - op))
            return ("hold", d, dict(stop=1.5)) if d else None
    elif name == "b39":
        for back in (0, 1, 2):
            s_ = F["shock"][i - back]
            if np.isfinite(s_) and abs(s_) > 2:
                return ("hold", -_sg((s_)), dict(stop=1.5))
    return None


def run_ext(sess: Sessions, name: str, costs: Costs, ctx: dict, F: Dict, PP: Dict, HS: Dict, **p) -> DailySeries:
    n = len(sess); cp = costs.points(sess)
    atr = st.trailing_atr(ctx["ranges"], sess.roll)
    pnl, mn, tr = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int32)
    ok = ~sess.roll & ~np.isnan(atr)
    for i in np.flatnonzero(ok):
        o, h, l, c, v, m = sess.day(i)
        end = I(m, FLAT_BY); a0 = I(m, CASH_OPEN)
        if end < 100 or a0 < 30 or a0 >= end - 40:
            continue
        A = atr[i]; pc = F["close"][i]; r = (0.0, 0.0, 0)
        if name == "a1":
            b = I(m, CASH_OPEN + p["or_min"])
            if b - a0 < 5 or not np.isfinite(pc) or abs(o[a0] - pc) > 1.0 * A:
                continue
            tdir = _sg((F["roc5"][i])) if np.isfinite(F["roc5"][i]) else 0
            if tdir == 0:
                continue
            hi, lo = h[a0:b].max(), l[a0:b].min()
            r = _break(o, h, l, c, m, tdir, hi, lo, b, end, (lambda pr, d, hi=hi, lo=lo: (pr - lo) if d > 0 else (hi - pr)), cp)
        elif name == "a2":
            b = I(m, CASH_OPEN + 15); ref = PP["f15"][i]
            if b - a0 < 5 or not np.isfinite(ref):
                continue
            hi, lo = h[a0:b].max(), l[a0:b].min(); rng_ = hi - lo
            if rng_ < p["rm"] * ref or rng_ <= 0:
                continue
            loc = (c[b - 1] - lo) / rng_
            d = 1 if loc >= 0.8 else (-1 if loc <= 0.2 else 0)
            if d == 0:
                continue
            sd = abs(o[b] - (hi + lo) / 2) if b < len(o) else 0
            r = _hold(o, h, l, c, m, d, b, end, sd, cp) if sd > 0 else r
        elif name == "a3":
            b1, b2 = I(m, CASH_OPEN + 15), I(m, CASH_OPEN + 30)
            if b2 - b1 < 5 or b1 - a0 < 5:
                continue
            d1 = _sg((c[b1 - 1] - o[a0])); d2 = _sg((c[b2 - 1] - o[b1])); mid1 = (h[a0:b1].max() + l[a0:b1].min()) / 2
            if d1 == 0 or d1 != d2 or not ((c[b2 - 1] > mid1) if d1 > 0 else (c[b2 - 1] < mid1)):
                continue
            sd = abs(o[b2] - mid1)
            r = _hold(o, h, l, c, m, d1, b2, end, sd, cp) if sd > 0 else r
        elif name == "a4":
            t_ = I(m, M(11, 30)); t0_ = I(m, M(10, 30))
            if t_ - t0_ < 20:
                continue
            if h[t0_:t_].max() > h[a0:t0_].max() or l[t0_:t_].min() < l[a0:t0_].min():
                continue                               # a new extreme was made in 10:30-11:30
            d = -_sg((c[t_ - 1] - o[a0]))
            if d == 0:
                continue
            r = _hold(o, h, l, c, m, d, t_, I(m, M(13, 30)), (1.0 * A if p["stop"] else None), cp)
        elif name == "a5":
            e = I(m, M(15, 25)); d = _sg((c[e - 1] - pc)) if np.isfinite(pc) else 0
            if d == 0 or not np.isfinite(PP["f30"][i]):
                continue
            r = _hold(o, h, l, c, m, d, e, end, (PP["f30"][i] if p["stop"] else None), cp)
        elif name == "a8":
            if not np.isfinite(pc) or not np.isfinite(F["roc5"][i]):
                continue
            gap = o[a0] - pc; tdir = _sg((F["roc5"][i]))
            if p["gate"] and F["adx"][i] > 25:
                continue
            if tdir == 0 or abs(gap) > 0.5 * A or abs(gap) < 0.05 * A or np.sign(gap) != -tdir:
                continue
            r = _hold(o, h, l, c, m, -_sg((gap)), a0, end, 0.75 * A, cp, target=abs(gap))
        elif name == "a9":
            if not np.isfinite(F["high"][i]) or not np.isfinite(PP["h60"][i]):
                continue
            mid = (F["high"][i] + F["low"][i]) / 2; thr = p["mult"] * PP["h60"][i]; done = False
            for hh_ in (M(10, 30), M(11, 30), M(12, 30), M(13, 30), M(14, 30)):
                t_ = I(m, hh_)
                if t_ <= a0 + 5 or done:
                    continue
                dev = c[t_ - 1] - mid; rng_ = h[a0:t_].max() - l[a0:t_].min()
                if abs(dev) > thr and rng_ > 0 and abs(c[t_ - 1] - o[a0]) / rng_ < 0.5:
                    r = _hold(o, h, l, c, m, -_sg((dev)), t_, end, 1.0 * A, cp, target=abs(dev)); done = True
        elif name == "a10":
            if not np.isfinite(F["low"][i]):
                continue
            pl_, ph_ = F["low"][i], F["high"][i]; mid = (pl_ + ph_) / 2; bars = []
            for k in range(26):
                x, y = I(m, CASH_OPEN + 15 * k), I(m, CASH_OPEN + 15 * (k + 1))
                if y - x < 5 or y > end:
                    break
                bars.append((x, y, o[x], h[x:y].max(), l[x:y].min(), c[y - 1]))
            for k in range(14, len(bars)):
                x, y, bo, bh, bl, bc = bars[k]
                hh14 = max(b[3] for b in bars[k - 13:k + 1]); ll14 = min(b[4] for b in bars[k - 13:k + 1])
                stoch = 100 * (bc - ll14) / (hh14 - ll14) if hh14 > ll14 else 50
                body = abs(bc - bo)
                low_w, up_w = min(bo, bc) - bl, bh - max(bo, bc)
                if bl <= pl_ and low_w >= 2 * max(body, 1e-9) and bc > bo and stoch < 20 and y < end - 5:
                    sd = o[y] - bl
                    if sd > 0:
                        r = _hold(o, h, l, c, m, 1, y, end, sd, cp, target=max(mid - o[y], 0.25 * A)); break
                if bh >= ph_ and up_w >= 2 * max(body, 1e-9) and bc < bo and stoch > 80 and y < end - 5:
                    sd = bh - o[y]
                    if sd > 0:
                        r = _hold(o, h, l, c, m, -1, y, end, sd, cp, target=max(o[y] - mid, 0.25 * A)); break
        elif name == "a6":
            prof = PP["prof"][i]
            if not np.all(np.isfinite(prof[:11])) or not np.isfinite(pc):
                continue
            top = np.argsort(prof[:11])[::-1][:2]
            tp = (h + l + c) / 3.0; vw = np.cumsum(tp * v) / np.maximum(np.cumsum(v), 1.0); tot = [0.0, 0.0, 0]
            for k in sorted(top):
                x, y = I(m, CASH_OPEN + 30 * k), I(m, CASH_OPEN + 30 * (k + 1))
                if x < 1 or y - x < 10:
                    continue
                d = _sg((c[x - 1] - vw[x - 1]))
                rr = _hold(o, h, l, c, m, d, x, y, None, cp)
                tot = [tot[0] + rr[0], min(tot[1], tot[0] + rr[1]), tot[2] + rr[2]]
            r = (tot[0], min(tot[1], tot[0]), tot[2])
        elif name == "a7v":
            r = st.vwap_atr_session(o, h, l, c, v, m, A, 0.3, 0.2, cp, start_min=CASH_OPEN, blackout=BLACKOUT)
        elif name == "a7n":
            r = ac.noise_area_session(o, h, l, c, v, m, ctx["prev_close"][i], ctx["sigma"][i], cp, blackout=BLACKOUT)
        elif name == "b36":
            ad, cl_, e200 = F["adx"][i], F["close"][i], F["ema200"][i]
            if not (ad > 25 and np.isfinite(e200)):
                continue
            dd = 1 if cl_ > e200 else -1
            for k in range(5):
                e_h, s_h = HS["ema"][i, k], HS["stoch"][i, k]
                x, y = I(m, CASH_OPEN + 60 * k), I(m, CASH_OPEN + 60 * (k + 1))
                if not (np.isfinite(e_h) and np.isfinite(s_h)) or y - x < 20 or y >= end - 5:
                    continue
                bl, bh = l[x:y].min(), h[x:y].max()
                if (dd > 0 and bl <= e_h and s_h < 20) or (dd < 0 and bh >= e_h and s_h > 80):
                    r = _hold(o, h, l, c, m, dd, y, end, 1.5 * A, cp); break
        else:
            sig = daily_signal(name, F, i, A)
            if sig is None:
                continue
            kind, d, q = sig
            if d == 0:
                continue
            if kind == "hold":
                if "stop_mid" in q:
                    sd = float(np.clip(abs(o[a0] - q["stop_mid"]), 0.25 * A, 2.0 * A))
                else:
                    sd = q["stop"] * A
                r = _hold(o, h, l, c, m, d, a0, end, sd, cp, target=q.get("target"))
            elif kind == "inside":
                if q["lo"] <= o[a0] <= q["hi"]:
                    sd = float(np.clip(abs(o[a0] - q["mid"]), 0.25 * A, 1.5 * A))
                    r = _hold(o, h, l, c, m, d, a0, end, sd, cp)
            elif kind == "break":
                capA = q["cap"] * A
                r = _break(o, h, l, c, m, d, q["up"], q["dn"], I(m, CASH_OPEN + 5), end,
                           (lambda pr, dd_, q=q, capA=capA: min((pr - q["dn"]) if dd_ > 0 else (q["up"] - pr), capA)), cp)
        pnl[i], mn[i], tr[i] = r[0] * sess.mult, r[1] * sess.mult, r[2]
    return DailySeries(sess.dates, pnl, mn, tr, ok, atr, sess.mult)
