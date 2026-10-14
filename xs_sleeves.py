"""ES / GC component ideas (pre-registration 6). All flat by 15:55 ET, signals at a bar close, fills at the next open, stops fill at the stop or the gap open.
  xs_prevday   prior-day-range breakout: close beyond yesterday's high / low +- buf*ATR after anchor+30 min
  xs_onrange   overnight-range breakout: close beyond the 18:00-to-anchor high / low +- buf*ATR after anchor+15 min
  xs_orb       opening-range breakout from `anchor` (GC: 08:20 ET COMEX open), range of `or_min` minutes, no buffer
`anchor` is 09:30 ET (smin 930) unless given; the COMEX gold open 08:20 ET is smin 860 (minutes after the 18:00 ET session open)."""
from __future__ import annotations

from typing import Dict

import numpy as np

import intraday_strategies as st
from intraday_strategies import CASH_OPEN, FLAT_BY, Costs, DailySeries, _first_true
from market_data import Sessions

COMEX_OPEN = 14 * 60 + 20


def xs_context(sess: Sessions) -> Dict[str, np.ndarray]:
    n = len(sess)
    hi, lo = np.zeros(n), np.zeros(n)
    for i in range(n):
        o, h, l, c, v, m = sess.day(i)
        hi[i], lo[i] = h.max(), l.min()
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    last_h = last_l = np.nan
    for i in range(n):
        ph[i], pl[i] = last_h, last_l
        if not sess.roll[i]:
            last_h, last_l = hi[i], lo[i]
    return dict(prev_hi=ph, prev_lo=pl)


def _breakout(o, h, l, c, smin, atr, up_lvl, dn_lvl, buf, s, cp, start_min, end):
    t0 = int(np.searchsorted(smin, start_min))
    if end - t0 < 30 or not (np.isfinite(up_lvl) and np.isfinite(dn_lvl)):
        return 0.0, 0.0, 0
    bu = buf * atr
    iu = _first_true(c[t0:end - 1] > up_lvl + bu)
    idn = _first_true(c[t0:end - 1] < dn_lvl - bu)
    if iu < 0 and idn < 0:
        return 0.0, 0.0, 0
    if idn < 0 or (iu >= 0 and iu < idn):
        direction, sig = 1, t0 + iu
    else:
        direction, sig = -1, t0 + idn
    e = sig + 1
    price, stop_d = o[e], s * atr
    js = _first_true(l[e:end] <= price - stop_d) if direction > 0 else _first_true(h[e:end] >= price + stop_d)
    if js >= 0:
        j = e + js
        px = min(price - stop_d, o[j]) if direction > 0 else max(price + stop_d, o[j])
        last = j
    else:
        px, last = c[end - 1], end - 1
    if direction > 0:
        worst, pnl = l[e:last + 1].min() - price, px - price
    else:
        worst, pnl = price - h[e:last + 1].max(), price - px
    net = pnl - cp
    return net, min(worst - cp, net, 0.0), 1


def run_xs(sess: Sessions, name: str, costs: Costs, ctx: dict, xs: dict, anchor: int = CASH_OPEN, **p) -> DailySeries:
    n = len(sess)
    cp = costs.points(sess)
    atr = st.trailing_atr(ctx["ranges"], sess.roll)
    pnl, mn, tr = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int32)
    ok = ~sess.roll & ~np.isnan(atr)
    for i in np.flatnonzero(ok):
        o, h, l, c, v, m = sess.day(i)
        end = int(np.searchsorted(m, FLAT_BY))
        a = int(np.searchsorted(m, anchor))
        if end < 100 or a < 60 or a >= end - 30:
            continue
        if name == "xs_prevday":
            r = _breakout(o, h, l, c, m, atr[i], xs["prev_hi"][i], xs["prev_lo"][i], p["buf"], p["s"], cp, anchor + 30, end)
        elif name == "xs_onrange":
            r = _breakout(o, h, l, c, m, atr[i], h[:a].max(), l[:a].min(), p["buf"], p["s"], cp, anchor + 15, end)
        elif name == "xs_orb":
            b = int(np.searchsorted(m, anchor + p["or_min"]))
            if b - a < max(5, p["or_min"] // 2):
                continue
            r = _breakout(o, h, l, c, m, atr[i], h[a:b].max(), l[a:b].min(), 0.0, p["s"], cp, anchor + p["or_min"], end)
        else:
            raise ValueError(name)
        pnl[i], mn[i], tr[i] = r[0] * sess.mult, r[1] * sess.mult, r[2]
    return DailySeries(sess.dates, pnl, mn, tr, ok, atr, sess.mult)
