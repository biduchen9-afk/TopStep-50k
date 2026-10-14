"""VWAP-combination sleeves (all flat by 15:55 ET; signal at a bar close, fill at the next open; costs as the other sleeves).

  vv_filt   VWAP+ATR momentum, but only in the direction an independent pre-trade signal agrees with:
              gap    sign of (09:30 open - previous close)
              first30 sign of the first-half-hour move (signals start after 10:00)
              gap30  both agree, otherwise no trade
              volhi  only when the 5/60-session range ratio >= 1 (Gao et al.: momentum is stronger on volatile days)
  vv_blend  switch between VWAP momentum and VWAP fade each session by a pre-trade regime indicator: the trailing 10-session mean
            efficiency (|close-open| / range) above its own trailing 250-session median -> momentum, else fade.
  vv_multi  VWAP momentum that needs agreement between several anchored VWAPs: 'c' cash-open (09:30) anchored, 'd' two-session
            (previous + current), 'w' week-to-date. The exit is a close back through the first listed VWAP.
Every indicator uses data from before the decision time only.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

import intraday_strategies as st
import academic_strategies as ac
from intraday_strategies import CASH_OPEN, FLAT_BY, Costs, DailySeries, _first_true, _exit_long_short
from market_data import Sessions
import pandas as pd


def extra_context(sess: Sessions) -> Dict[str, np.ndarray]:
    n = len(sess)
    pv_tot, v_tot, eff = np.zeros(n), np.zeros(n), np.full(n, np.nan)
    for i in range(n):
        o, h, l, c, v, m = sess.day(i)
        tp = (h + l + c) / 3.0
        pv_tot[i], v_tot[i] = float((tp * v).sum()), float(v.sum())
        rg = h.max() - l.min()
        eff[i] = abs(c[-1] - o[0]) / rg if rg > 0 else np.nan
    idx = pd.DatetimeIndex(sess.dates)
    week = idx.to_period("W-SUN") if False else (idx - pd.Timedelta(days=0)).isocalendar()
    wk_key = (week["year"].to_numpy() * 100 + week["week"].to_numpy())
    skip = sess.roll
    prev_pv, prev_v = np.zeros(n), np.zeros(n)
    wk_pv, wk_v = np.zeros(n), np.zeros(n)
    last_pv = last_v = 0.0
    a_pv = a_v = 0.0
    cur = None
    for i in range(n):
        if wk_key[i] != cur:
            cur, a_pv, a_v = wk_key[i], 0.0, 0.0
        prev_pv[i], prev_v[i] = last_pv, last_v
        wk_pv[i], wk_v[i] = a_pv, a_v
        if not skip[i]:
            last_pv, last_v = pv_tot[i], v_tot[i]
            a_pv += pv_tot[i]; a_v += v_tot[i]
    e = pd.Series(eff)
    eff10 = e.rolling(10, min_periods=8).mean().shift(1)
    med250 = eff10.rolling(250, min_periods=120).median()
    return dict(prev_pv=prev_pv, prev_v=prev_v, wk_pv=wk_pv, wk_v=wk_v, eff10=eff10.to_numpy(), eff_med=med250.to_numpy())


def vwap_multi_session(o, h, l, c, v, smin, atr, k, s, cost_pts, anchors, ex, start_min=CASH_OPEN, allow=0, max_trades=3):
    end = int(np.searchsorted(smin, st.FLAT_BY))
    a = int(np.searchsorted(smin, CASH_OPEN))
    if end < 100 or a >= end - 30:
        return 0.0, 0.0, 0
    tp = (h + l + c) / 3.0
    pvc, vc = np.cumsum(tp * v), np.cumsum(v)
    vws = []
    for ch in anchors:
        if ch == "c":
            pv0, v0 = pvc[a - 1] if a > 0 else 0.0, vc[a - 1] if a > 0 else 0.0
            w = (pvc - pv0) / np.maximum(vc - v0, 1.0)
            w[:a] = pvc[:a] / np.maximum(vc[:a], 1.0)           # before 09:30 fall back to the session VWAP (no signals then)
        elif ch == "d":
            w = (ex["prev_pv"] + pvc) / np.maximum(ex["prev_v"] + vc, 1.0)
        elif ch == "w":
            w = (ex["wk_pv"] + pvc) / np.maximum(ex["wk_v"] + vc, 1.0)
        elif ch == "s":
            w = pvc / np.maximum(vc, 1.0)
        else:
            raise ValueError(ch)
        vws.append(w)
    vex = vws[0]
    band, stop_d = k * atr, s * atr
    up = np.ones(len(c), dtype=bool); dn = np.ones(len(c), dtype=bool)
    for w in vws:
        up &= c > w + band
        dn &= c < w - band
    if allow > 0:
        dn[:] = False
    elif allow < 0:
        up[:] = False
    up_edge = up & ~np.r_[False, up[:-1]]
    dn_edge = dn & ~np.r_[False, dn[:-1]]
    t = int(np.searchsorted(smin, start_min))
    realized, minp, trades, first = 0.0, 0.0, 0, True
    while trades < max_trades and t < end - 2:
        mu = up[t:end - 1] if first else up_edge[t:end - 1]
        md = dn[t:end - 1] if first else dn_edge[t:end - 1]
        iu, idn = _first_true(mu), _first_true(md)
        if iu < 0 and idn < 0:
            break
        if idn < 0 or (iu >= 0 and iu < idn):
            direction, sig = 1, t + iu
        else:
            direction, sig = -1, t + idn
        first = False
        e = sig + 1
        price = o[e]
        px, last = _exit_long_short(direction, price, stop_d, e, o, h, l, c, vex, end)
        last = max(last, e)
        if direction > 0:
            worst, pnl_pts = l[e:last + 1].min() - price, px - price
        else:
            worst, pnl_pts = price - h[e:last + 1].max(), price - px
        minp = min(minp, realized + min(worst, pnl_pts))
        realized += pnl_pts - cost_pts
        trades += 1
        t = last + 1
    return realized, min(minp, realized), trades


def run_variant(sess: Sessions, name: str, costs: Costs, ctx: dict, ex: dict, trend: Optional[int] = None, **p) -> DailySeries:
    n = len(sess)
    cp = costs.points(sess)
    atr = st.trailing_atr(ctx["ranges"], sess.roll)
    pnl, mn, tr = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int32)
    ok = ~sess.roll & ~np.isnan(atr)
    for i in np.flatnonzero(ok):
        o, h, l, c, v, m = sess.day(i)
        a = int(np.searchsorted(m, CASH_OPEN)); b = int(np.searchsorted(m, CASH_OPEN + 30))
        allow = int(ctx[f"trend{trend}"][i]) if trend else 0
        if trend and allow == 0:
            continue
        r = (0.0, 0.0, 0)
        if name == "vv_filt":
            f, k, s = p["filt"], p["k"], p["s"]
            pc = ctx["prev_close"][i]
            sg_gap = int(np.sign(o[a] - pc)) if (np.isfinite(pc) and a < len(o)) else 0
            sg_30 = int(np.sign(c[b - 1] - o[a])) if (b > a + 5 and b <= len(c)) else 0
            start = CASH_OPEN + 30 if "30" in f else CASH_OPEN
            if f == "gap":
                al = sg_gap
            elif f == "first30":
                al = sg_30
            elif f == "gap30":
                al = sg_gap if (sg_gap == sg_30 and sg_gap != 0) else 0
            elif f == "volhi":
                vr = ctx["volratio"][i]
                al = 2 if (np.isfinite(vr) and vr >= 1.0) else 0           # 2 = any direction allowed
            else:
                raise ValueError(f)
            if al == 0:
                continue
            if allow != 0 and al in (-1, 1) and al != allow:
                continue
            use = allow if al == 2 else al
            r = st.vwap_atr_session(o, h, l, c, v, m, atr[i], k, s, cp, start_min=start, allow=use)
        elif name == "vv_blend":
            e10, med = ex["eff10"][i], ex["eff_med"][i]
            if not (np.isfinite(e10) and np.isfinite(med)):
                continue
            if e10 > med:
                r = st.vwap_atr_session(o, h, l, c, v, m, atr[i], p.get("k_m", 0.3), p.get("s_m", 0.2), cp, start_min=CASH_OPEN, allow=allow)
            else:
                r = ac.vwap_fade_session(o, h, l, c, v, m, atr[i], p.get("k_f", 0.6), p.get("s_f", 0.4), cp, allow=allow)
        elif name == "vv_multi":
            exi = dict(prev_pv=ex["prev_pv"][i], prev_v=ex["prev_v"][i], wk_pv=ex["wk_pv"][i], wk_v=ex["wk_v"][i])
            r = vwap_multi_session(o, h, l, c, v, m, atr[i], p["k"], p["s"], cp, p["anchors"], exi, start_min=CASH_OPEN, allow=allow)
        else:
            raise ValueError(name)
        pnl[i], mn[i], tr[i] = r[0] * sess.mult, r[1] * sess.mult, r[2]
    return DailySeries(sess.dates, pnl, mn, tr, ok, atr, sess.mult)
