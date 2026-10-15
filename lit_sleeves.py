"""Sleeves from the SSRN / arXiv search (pre-registration 7). Every entry is at or after 09:30 ET and every exit by 15:55 ET; signals only use data available at the entry time.
  lt_mom      last-half-hour momentum (Gao et al.; Baltussen et al.)         lt_reversal   fade of the 09:30-10:00 move (Fung; Grant et al.)
  lt_lunch    long 11:00-14:00 (Vojtko-Dujava)                               lt_candle5    5-minute opening-candle breakout (Zarattini-Aziz)
  lt_slot     same-half-hour continuation (Heston-Korajczyk-Sadka)           lt_xasset     ES/NQ 09:30-10:00 return -> GC 10:00-11:00
  lt_nr       narrow-range-day breakout (Crabel)                             (L5, the volatility-gated ORB, reuses intraday_strategies.orb_session)"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

import intraday_strategies as st
from intraday_strategies import CASH_OPEN, FLAT_BY, Costs, DailySeries, _first_true
from market_data import Sessions, build_sessions

M = lambda hh, mm: (hh * 60 + mm - 18 * 60) % 1440          # ET clock -> minutes after the 18:00 ET session open
T1000, T1100, T1200, T1400, T1325, T1330 = M(10, 0), M(11, 0), M(12, 0), M(14, 0), M(13, 25), M(13, 30)
_CACHE: Dict[str, dict] = {}


def _hold(o, h, l, c, smin, direction, e, xend, stop_d, cp, target=None):
    """Enter at the open of bar e, hold to the close of bar xend-1; optional stop distance / target distance in points."""
    if direction == 0 or xend - e < 3 or e >= len(o):
        return 0.0, 0.0, 0
    price = o[e]
    last, px = xend - 1, c[xend - 1]
    js = jt = -1
    if direction > 0:
        if stop_d:
            js = _first_true(l[e:xend] <= price - stop_d)
        if target:
            jt = _first_true(h[e:xend] >= price + target)
    else:
        if stop_d:
            js = _first_true(h[e:xend] >= price + stop_d)
        if target:
            jt = _first_true(l[e:xend] <= price - target)
    if js >= 0 and (jt < 0 or js <= jt):
        j = e + js
        px = min(price - stop_d, o[j]) if direction > 0 else max(price + stop_d, o[j])
        last = j
    elif jt >= 0:
        j = e + jt
        px = max(price + target, o[j]) if direction > 0 else min(price - target, o[j])
        last = j
    st._trace(smin, e, max(last, e))
    if direction > 0:
        worst, pnl = l[e:last + 1].min() - price, px - price
    else:
        worst, pnl = price - h[e:last + 1].max(), price - px
    net = pnl - cp
    return net, min(worst - cp, net, 0.0), 1


def _slot_returns(sess: Sessions, s0: int, s1: int) -> np.ndarray:
    n = len(sess); out = np.full(n, np.nan)
    for i in range(n):
        o, h, l, c, v, m = sess.day(i)
        a, b = int(np.searchsorted(m, s0)), int(np.searchsorted(m, s1))
        if b - a >= 10 and b <= len(c):
            out[i] = c[b - 1] / o[a] - 1.0
    return out


def _xret(market: str) -> Dict[str, float]:
    key = "x_" + market
    if key not in _CACHE:
        s = build_sessions(market); d = {}
        for i in range(len(s)):
            o, h, l, c, v, m = s.day(i)
            a, b = int(np.searchsorted(m, CASH_OPEN)), int(np.searchsorted(m, T1000))
            if b - a >= 10 and not s.roll[i]:
                d[str(s.dates[i])] = c[b - 1] / o[a] - 1.0
        _CACHE[key] = d
    return _CACHE[key]


def run_lit(sess: Sessions, name: str, costs: Costs, ctx: dict, **p) -> DailySeries:
    n = len(sess); cp = costs.points(sess)
    atr = st.trailing_atr(ctx["ranges"], sess.roll)
    pnl, mn, tr = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int32)
    ok = ~sess.roll & ~np.isnan(atr)
    slot_ret = None
    if name == "lt_slot":
        slot_ret = _slot_returns(sess, p["s0"], p["s1"])
    xr = _xret(p["src"]) if name == "lt_xasset" else None
    rng = ctx["ranges"]
    for i in np.flatnonzero(ok):
        o, h, l, c, v, m = sess.day(i)
        end = int(np.searchsorted(m, FLAT_BY)); a0 = int(np.searchsorted(m, CASH_OPEN))
        if end < 100 or a0 < 30 or a0 >= end - 40:
            continue
        A = atr[i]; r = (0.0, 0.0, 0)
        if name == "lt_mom":
            pc = ctx["prev_close"][i]
            if not np.isfinite(pc):
                continue
            if p["filter"] == "hivol":
                vr = ctx["volratio"][i]
                if not (np.isfinite(vr) and vr >= 1.0):
                    continue
            ws, we = (T1325 - 0, T1330) if p["window"] == "comex" else (M(15, 25), FLAT_BY)
            if p["window"] == "comex":
                ws = M(13, 0)
            e, xe = int(np.searchsorted(m, ws)), min(int(np.searchsorted(m, we)), end)
            if p["signal"] == "gao":
                i10 = int(np.searchsorted(m, T1000))
                sig = c[i10 - 1] / pc - 1.0
                if p["window"] == "close":
                    ia, ib = int(np.searchsorted(m, M(13, 0))), int(np.searchsorted(m, M(13, 30)))
                    if ib > ia > 0:
                        sig += c[ib - 1] / c[ia - 1] - 1.0
            else:
                sig = c[e - 1] / pc - 1.0
            r = _hold(o, h, l, c, m, int(np.sign(sig)), e, xe, None, cp)
        elif name == "lt_reversal":
            i1 = int(np.searchsorted(m, T1000)); move = c[i1 - 1] - o[a0]
            if abs(move) < p["k"] * A:
                continue
            xe = min(int(np.searchsorted(m, T1200)), end) if p["exit"] == "noon" else end
            r = _hold(o, h, l, c, m, -int(np.sign(move)), i1, xe, 1.0 * A, cp)
        elif name == "lt_lunch":
            e, xe = int(np.searchsorted(m, T1100)), int(np.searchsorted(m, T1400))
            if p["cond"] == "down" and not (c[e - 1] - o[a0] < 0):
                continue
            r = _hold(o, h, l, c, m, 1, e, xe, None, cp)
        elif name == "lt_candle5":
            e = int(np.searchsorted(m, CASH_OPEN + 5))
            if e - a0 < 3:
                continue
            lo_c, hi_c = l[a0:e].min(), h[a0:e].max(); d = int(np.sign(c[e - 1] - o[a0]))
            if d == 0:
                continue
            stop_d = (o[e] - lo_c) if d > 0 else (hi_c - o[e])
            if stop_d <= 0:
                continue
            r = _hold(o, h, l, c, m, d, e, end, stop_d, cp, target=10 * stop_d if p["target"] == "10R" else None)
        elif name == "lt_orbvol":
            vr = ctx["volratio"][i]
            if not (np.isfinite(vr) and vr >= 1.0):
                continue
            r = st.orb_session(o, h, l, c, v, m, A, 0.1, 0.5, cp, or_min=p["or_min"])
        elif name == "lt_slot":
            past = slot_ret[max(0, i - 10):i]; past = past[np.isfinite(past)]
            if len(past) < 8:
                continue
            e, xe = int(np.searchsorted(m, p["s0"])), int(np.searchsorted(m, p["s1"]))
            r = _hold(o, h, l, c, m, int(np.sign(past.mean())), e, xe, None, cp)
        elif name == "lt_xasset":
            sig = xr.get(str(sess.dates[i]))
            if sig is None:
                continue
            e, xe = int(np.searchsorted(m, T1000)), int(np.searchsorted(m, T1100))
            r = _hold(o, h, l, c, m, int(np.sign(sig)) * (1 if p["sign"] == "same" else -1), e, xe, 1.0 * A, cp)
        elif name == "lt_nr":
            N = p["N"]
            if i < N + 1 or not (rng[i - 1] < np.min(rng[i - N:i - 1])):
                continue
            t0 = int(np.searchsorted(m, CASH_OPEN + 5)); lvl = o[a0]; band = 0.5 * A
            iu, idn = _first_true(c[t0:end - 1] > lvl + band), _first_true(c[t0:end - 1] < lvl - band)
            if iu < 0 and idn < 0:
                continue
            d, sig = (1, t0 + iu) if (idn < 0 or (iu >= 0 and iu < idn)) else (-1, t0 + idn)
            r = _hold(o, h, l, c, m, d, sig + 1, end, 1.0 * A, cp)
        else:
            raise ValueError(name)
        pnl[i], mn[i], tr[i] = r[0] * sess.mult, r[1] * sess.mult, r[2]
    return DailySeries(sess.dates, pnl, mn, tr, ok, atr, sess.mult)
