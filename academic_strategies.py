"""Sleeves from the academic literature, plus trend / volatility gating, on full futures sessions.

Papers (checked by web search, Oct 2026):
  * Gao, Han, Li, Zhou, "Market Intraday Momentum", JFE 2018 (SSRN 2440866): the first half-hour return
    (from the previous close) predicts the last half-hour return; stronger on volatile days.
  * Zarattini, Aziz, Barbon, "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)",
    SSRN 4824172 (2024): trade breakouts of a time-of-day "Noise Area" around the open, positions checked every
    30 minutes, VWAP trailing stop. Independent replications report the edge shrinking recently.
  * Zarattini, Barbon, Aziz, "A Profitable Day Trading Strategy For The U.S. Equity Market", SSRN 4729284 (2024):
    opening-range breakout; evidence is for single stocks in play, not index futures.
  * Moskowitz, Ooi, Pedersen, "Time Series Momentum", JFE 2012 (SSRN 2089463): the sign of the trailing
    return predicts the next period in 58 futures markets -> used here as a direction gate.
All sleeves are flat by 15:55 ET (Topstep requires flat by 16:10 ET). Fills: signal at a bar close, entry at the
next open; stops fill at the stop or the gap open.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

import intraday_strategies as st
from intraday_strategies import CASH_OPEN, FLAT_BY, Costs, DailySeries, _first_true
from market_data import Sessions

STEP = 30
K_DEC = 12                 # decision times 10:00 .. 15:30 ET
LAST_HALF_HOUR = 21 * 60 + 25   # 15:25 ET as minutes after the 18:00 ET open
R12_START, R12_END = 19 * 60, 19 * 60 + 30     # 13:00-13:30 ET


def _px_before(c, smin, t):
    i = int(np.searchsorted(smin, t))
    return (c[i - 1] if i > 0 else np.nan), i


def context(sess: Sessions) -> Dict[str, np.ndarray]:
    """Per-session features known BEFORE the trade (no look-ahead)."""
    n = len(sess)
    prev_close = np.full(n, np.nan)
    ranges = st.session_ranges(sess)
    move = np.full((n, K_DEC), np.nan)
    last_close = np.full(n, np.nan)
    for i in range(n):
        o, h, l, c, v, m = sess.day(i)
        last_close[i] = c[-1]
        a = int(np.searchsorted(m, CASH_OPEN))
        if a >= len(c) - 5:
            continue
        op = o[a]
        for k in range(1, K_DEC + 1):
            p, idx = _px_before(c, m, CASH_OPEN + STEP * k)
            if idx > a:
                move[i, k - 1] = abs(p / op - 1.0)
    prev_close[1:] = last_close[:-1]
    sigma = np.full((n, K_DEC), np.nan)
    for i in range(14, n):
        sigma[i] = np.nanmean(move[i - 14:i], axis=0)
    out = {"prev_close": prev_close, "sigma": sigma, "ranges": ranges}
    for N in (20, 60, 250):
        tr = np.zeros(n)
        for i in range(N + 1, n):
            a, b = prev_close[i], prev_close[i - N]
            if np.isfinite(a) and np.isfinite(b):
                tr[i] = np.sign(a / b - 1.0)
        out[f"trend{N}"] = tr
    v5 = np.full(n, np.nan); v60 = np.full(n, np.nan)
    for i in range(60, n):
        v5[i] = ranges[i - 5:i].mean(); v60[i] = ranges[i - 60:i].mean()
    out["volratio"] = v5 / v60
    return out


# ---------------------------------------------------------------------------------------------
def intraday_mom_session(o, h, l, c, v, smin, prev_close, atr, cost_pts, thr=0.0, use_r12=True, allow=0):
    """Gao et al.: sign of the first-half-hour return (from the previous close), plus the 13:00-13:30 return,
    predicts the last half hour. Trade 15:25 -> 15:55 ET."""
    p10, i10 = _px_before(c, smin, CASH_OPEN + 30)
    if not np.isfinite(prev_close) or i10 < 5:
        return 0.0, 0.0, 0
    sig = p10 / prev_close - 1.0
    if use_r12:
        pa, ia = _px_before(c, smin, R12_START)
        pb, ib = _px_before(c, smin, R12_END)
        if ia > 0 and ib > ia:
            sig += pb / pa - 1.0
    if abs(sig) * prev_close < thr * atr or sig == 0:
        return 0.0, 0.0, 0
    direction = 1 if sig > 0 else -1
    if allow != 0 and direction != allow:
        return 0.0, 0.0, 0
    a = int(np.searchsorted(smin, LAST_HALF_HOUR))
    end = int(np.searchsorted(smin, FLAT_BY))
    if end - a < 10:
        return 0.0, 0.0, 0
    price, px = o[a], c[end - 1]
    if direction > 0:
        worst, pnl = l[a:end].min() - price, px - price
    else:
        worst, pnl = price - h[a:end].max(), price - px
    net = pnl - cost_pts
    return net, min(worst - cost_pts, net, 0.0), 1


def noise_area_session(o, h, l, c, v, smin, prev_close, sigma, cost_pts, allow=0, max_trades=3):
    """Zarattini-Aziz-Barbon style: Noise Area = [min(open, prev_close)*(1-sigma_t), max(open, prev_close)*(1+sigma_t)]
    where sigma_t is the 14-day average |move from the open| at the same time of day. Every 30 minutes: flat and price above
    the upper edge -> long, below the lower edge -> short; exit when price falls back through max(upper, VWAP)
    (long) / min(lower, VWAP) (short); flat by 15:55 ET."""
    a0 = int(np.searchsorted(smin, CASH_OPEN))
    end = int(np.searchsorted(smin, FLAT_BY))
    if not np.isfinite(prev_close) or a0 > len(c) - 20 or end - a0 < 100:
        return 0.0, 0.0, 0
    op = o[a0]
    hi_b, lo_b = max(op, prev_close), min(op, prev_close)
    tp = (h[a0:] + l[a0:] + c[a0:]) / 3.0
    cv = np.cumsum(v[a0:])
    vw = np.cumsum(tp * v[a0:]) / np.maximum(cv, 1.0)
    realized, minp, trades = 0.0, 0.0, 0
    pos, entry, e_idx = 0, 0.0, 0

    def close_out(exit_px, last_idx):
        nonlocal realized, minp, pos
        if pos > 0:
            worst, pnl = l[e_idx:last_idx + 1].min() - entry, exit_px - entry
        else:
            worst, pnl = entry - h[e_idx:last_idx + 1].max(), entry - exit_px
        minp = min(minp, realized + min(worst, pnl))
        realized += pnl - cost_pts
        pos = 0

    for k in range(1, K_DEC + 1):
        t = CASH_OPEN + STEP * k
        idx = int(np.searchsorted(smin, t))
        if idx >= end - 1:
            break
        sg = sigma[k - 1]
        if not np.isfinite(sg):
            continue
        px = c[idx - 1]
        vwap = vw[idx - 1 - a0] if idx - 1 >= a0 else op
        ub, lb = hi_b * (1 + sg), lo_b * (1 - sg)
        if pos == 0 and trades < max_trades:
            if px > ub and allow >= 0:
                pos, entry, e_idx, trades = 1, o[idx], idx, trades + 1
            elif px < lb and allow <= 0:
                pos, entry, e_idx, trades = -1, o[idx], idx, trades + 1
        elif pos > 0 and px < max(ub, vwap):
            close_out(o[idx], max(idx - 1, e_idx))
        elif pos < 0 and px > min(lb, vwap):
            close_out(o[idx], max(idx - 1, e_idx))
    if pos != 0:
        close_out(c[end - 1], end - 1)
    return realized, min(minp, realized), trades


def vwap_fade_session(o, h, l, c, v, smin, atr, k, s, cost_pts, allow=0, max_trades=2):
    """Mean reversion: fade a stretch of k*ATR away from the (18:00-anchored) VWAP during cash hours; exit at VWAP or stop s*ATR."""
    n = len(c)
    end = int(np.searchsorted(smin, FLAT_BY))
    t0 = int(np.searchsorted(smin, CASH_OPEN + 30))
    if end - t0 < 60:
        return 0.0, 0.0, 0
    tp = (h + l + c) / 3.0
    cv = np.cumsum(v)
    vw = np.cumsum(tp * v) / np.maximum(cv, 1.0)
    band, stop_d = k * atr, s * atr
    up, dn = c > vw + band, c < vw - band          # stretched above -> short, below -> long
    if allow > 0:
        up = np.zeros_like(up)
    elif allow < 0:
        dn = np.zeros_like(dn)
    realized, minp, trades, t = 0.0, 0.0, 0, t0
    while trades < max_trades and t < end - 2:
        iu, idn = _first_true(up[t:end - 1]), _first_true(dn[t:end - 1])
        if iu < 0 and idn < 0:
            break
        if idn < 0 or (iu >= 0 and iu < idn):
            direction, sig = -1, t + iu
        else:
            direction, sig = 1, t + idn
        e = sig + 1
        price = o[e]
        stop = price - stop_d if direction > 0 else price + stop_d
        # exit: price reverts to VWAP (close crosses it) or stop
        if direction > 0:
            js = _first_true(l[e:end] <= stop)
            jc = _first_true(c[e:end] >= vw[e:end])
        else:
            js = _first_true(h[e:end] >= stop)
            jc = _first_true(c[e:end] <= vw[e:end])
        js = e + js if js >= 0 else 10 ** 9
        jc_exit = e + jc + 1 if jc >= 0 and e + jc + 1 < end else 10 ** 9
        if jc_exit <= js and jc_exit < 10 ** 9:
            px, last = o[jc_exit], jc_exit - 1
        elif js < 10 ** 9:
            px = min(stop, o[js]) if direction > 0 else max(stop, o[js])
            last = js
        else:
            px, last = c[end - 1], end - 1
        last = max(last, e)
        if direction > 0:
            worst, pnl = l[e:last + 1].min() - price, px - price
        else:
            worst, pnl = price - h[e:last + 1].max(), price - px
        minp = min(minp, realized + min(worst, pnl))
        realized += pnl - cost_pts
        trades += 1
        t = last + 2
    return realized, min(minp, realized), trades


# ---------------------------------------------------------------------------------------------
SLEEVES = ("vwap_atr", "orb", "gap", "intraday_mom", "noise_area", "vwap_fade")


def run_sleeve(sess: Sessions, name: str, costs: Costs = Costs(), ctx: Optional[dict] = None, trend: Optional[int] = None,
               vol: Optional[str] = None, vol_thr: float = 1.0, **p) -> DailySeries:
    """Run one sleeve with optional gates. trend=N: only trade WITH the sign of the trailing N-session return (TSMOM gate).
    vol='hi'/'lo': trade only when the 5/60-session range ratio is >= / < vol_thr (Gao et al.: momentum stronger on volatile days)."""
    ctx = ctx or context(sess)
    n = len(sess)
    cp = costs.points(sess)
    atr = st.trailing_atr(ctx["ranges"], sess.roll)
    pnl, mn, tr = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int32)
    ok = ~sess.roll & ~np.isnan(atr)
    for i in np.flatnonzero(ok):
        if vol is not None:
            vr = ctx["volratio"][i]
            if not np.isfinite(vr) or (vol == "hi" and vr < vol_thr) or (vol == "lo" and vr >= vol_thr):
                continue
        allow = int(ctx[f"trend{trend}"][i]) if trend else 0
        if trend and allow == 0:
            continue
        o, h, l, c, v, m = sess.day(i)
        if name == "vwap_atr":
            r = st.vwap_atr_session(o, h, l, c, v, m, atr[i], p["k"], p["s"], cp, start_min=p.get("start_min", CASH_OPEN), allow=allow)
        elif name == "orb":
            r = st.orb_session(o, h, l, c, v, m, atr[i], p["buf"], p["s"], cp, or_min=p.get("or_min", 30), allow=allow)
        elif name == "gap":
            r = st.gap_session(o, h, l, c, v, m, atr[i], p["thr"], p["s"], p["mode"], cp, allow=allow)
        elif name == "intraday_mom":
            r = intraday_mom_session(o, h, l, c, v, m, ctx["prev_close"][i], atr[i], cp, thr=p.get("thr", 0.0), use_r12=p.get("use_r12", True), allow=allow)
        elif name == "noise_area":
            r = noise_area_session(o, h, l, c, v, m, ctx["prev_close"][i], ctx["sigma"][i], cp, allow=allow)
        elif name == "vwap_fade":
            r = vwap_fade_session(o, h, l, c, v, m, atr[i], p["k"], p["s"], cp, allow=allow)
        else:
            raise ValueError(name)
        pnl[i], mn[i], tr[i] = r[0] * sess.mult, r[1] * sess.mult, r[2]
    return DailySeries(sess.dates, pnl, mn, tr, ok, atr, sess.mult)
