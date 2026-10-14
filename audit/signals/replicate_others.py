"""Independent bar-by-bar (causal) replications of orb / gap / intraday_mom / vwap_fade on NQ from the RAW file; compare to project."""
import sys
sys.path.insert(0, '..')
import numpy as np, pandas as pd
from replicate_noise import build_days
from load_raw import load
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions


def prep(day):
    md = day['md']
    key = np.where(md >= 1080, md - 1440, md)         # chronological minutes, 18:00 = -360
    end = int(np.sum(key < 955))
    return key, end


def first_ge(key, T):
    return int(np.searchsorted(key, T))


def orb(day, atr, buf=0.05, s=0.2, or_min=30):
    key, end = prep(day); o, h, l, c = day['o'], day['h'], day['l'], day['c']
    a, b = first_ge(key, 570), first_ge(key, 570 + or_min)
    if b - a < max(5, or_min // 2) or end - b < 30:
        return 0.0, 0
    hi = lo = None
    pos = 0; entry = stop = 0.0; pend = 0; pnl = 0.0; trades = 0
    for t in range(a, end):
        # start of bar
        if pend and pos == 0 and trades == 0:
            pos = pend; entry = o[t]; stop = entry - s * atr if pos > 0 else entry + s * atr; trades = 1; pend = 0
        if pos > 0 and l[t] <= stop:
            return pnl + min(stop, o[t]) - entry, 1
        if pos < 0 and h[t] >= stop:
            return pnl + entry - max(stop, o[t]), 1
        # close of bar
        if t < b:
            hi = h[t] if hi is None else max(hi, h[t]); lo = l[t] if lo is None else min(lo, l[t])
        elif pos == 0 and trades == 0 and t <= end - 2:
            if c[t] > hi + buf * atr:
                pend = 1
            elif c[t] < lo - buf * atr:
                pend = -1
    if pos != 0:
        pnl += pos * (c[end - 1] - entry)
    return pnl, trades


def gap(day, atr, thr=0.2, s=0.6, mode=1):
    key, end = prep(day); o, h, l, c = day['o'], day['h'], day['l'], day['c']
    a = first_ge(key, 570)
    if a < 100 or end - a < 100:
        return 0.0, 0
    move = c[a - 1] - o[0]                       # known at 09:30
    if abs(move) < thr * atr:
        return 0.0, 0
    d = int(np.sign(move)) * mode
    entry = o[a]; stop = entry - s * atr if d > 0 else entry + s * atr
    for t in range(a, end):
        if d > 0 and l[t] <= stop:
            return min(stop, o[t]) - entry, 1
        if d < 0 and h[t] >= stop:
            return entry - max(stop, o[t]), 1
    return d * (c[end - 1] - entry), 1


def mom(day, prev_close, atr, use_r12=True):
    key, end = prep(day); o, h, l, c = day['o'], day['h'], day['l'], day['c']
    def before(T):
        i = first_ge(key, T); return (c[i - 1] if i > 0 else np.nan), i
    p10, i10 = before(570 + 30)
    if not np.isfinite(prev_close) or i10 < 5:
        return 0.0, 0
    sig = p10 / prev_close - 1.0
    if use_r12:
        pa, ia = before(13 * 60); pb, ib = before(13 * 60 + 30)
        if ia > 0 and ib > ia:
            sig += pb / pa - 1.0
    if sig == 0:
        return 0.0, 0
    d = 1 if sig > 0 else -1
    a = first_ge(key, 15 * 60 + 25)
    if end - a < 10:
        return 0.0, 0
    return d * (c[end - 1] - o[a]), 1


def fade(day, atr, k=0.3, s=0.3, max_trades=2):
    key, end = prep(day); o, h, l, c, v = day['o'], day['h'], day['l'], day['c'], day['v']
    t0 = first_ge(key, 600)
    if end - t0 < 60:
        return 0.0, 0
    band, sd = k * atr, s * atr
    cpv = cv = 0.0; pos = 0; entry = stop = 0.0; pend = 0; pend_exit = False; trades = 0; pnl = 0.0; blocked = -1
    for t in range(end):
        stopped = False; exited_open = False
        if pos != 0 and pend_exit:
            pnl += pos * (o[t] - entry); pos = 0; blocked = t + 1       # last = t-1 -> next signal bar >= last+2 = t+1
        pend_exit = False
        if pos == 0 and pend:
            pos = pend; entry = o[t]; stop = entry - sd if pos > 0 else entry + sd; trades += 1; pend = 0
        if pos > 0 and l[t] <= stop:
            pnl += min(stop, o[t]) - entry; pos = 0; blocked = t + 2; stopped = True
        elif pos < 0 and h[t] >= stop:
            pnl += entry - max(stop, o[t]); pos = 0; blocked = t + 2; stopped = True
        cpv += (h[t] + l[t] + c[t]) / 3.0 * v[t]; cv += v[t]
        vw = cpv / max(cv, 1.0)
        if pos != 0:
            if ((pos > 0 and c[t] >= vw) or (pos < 0 and c[t] <= vw)) and t + 1 < end:
                pend_exit = True
        elif trades < max_trades and t >= t0 and t >= blocked and t <= end - 2 and not pend:
            if c[t] < vw - band:
                pend = 1
            elif c[t] > vw + band:
                pend = -1
    if pos != 0:
        pnl += pos * (c[end - 1] - entry)
    return pnl, trades


if __name__ == '__main__':
    df = load('nq'); days = build_days(df); dates = sorted(days)
    sess = build_sessions('nq'); ctx = ac.context(sess)
    atr = st.trailing_atr(ctx['ranges'], sess.roll); ok = ~sess.roll & ~np.isnan(atr)
    assert [pd.Timestamp(x) for x in sess.dates] == dates
    zero = st.Costs(0.0, 0.0)
    specs = {
        'orb30': ('orb', dict(or_min=30, buf=0.05, s=0.2), lambda i, d: orb(d, atr[i], 0.05, 0.2, 30)),
        'orb60': ('orb', dict(or_min=60, buf=0.05, s=0.2), lambda i, d: orb(d, atr[i], 0.05, 0.2, 60)),
        'gap_mom': ('gap', dict(mode=1, thr=0.2, s=0.6), lambda i, d: gap(d, atr[i], 0.2, 0.6, 1)),
        'gap_fade': ('gap', dict(mode=-1, thr=0.2, s=0.6), lambda i, d: gap(d, atr[i], 0.2, 0.6, -1)),
        'mom': ('intraday_mom', dict(use_r12=True, thr=0.0), lambda i, d: mom(d, ctx['prev_close'][i], atr[i], True)),
        'mom_nor12': ('intraday_mom', dict(use_r12=False, thr=0.0), lambda i, d: mom(d, ctx['prev_close'][i], atr[i], False)),
        'fade': ('vwap_fade', dict(k=0.3, s=0.3), lambda i, d: fade(d, atr[i], 0.3, 0.3)),
        'fade2': ('vwap_fade', dict(k=0.5, s=0.6), lambda i, d: fade(d, atr[i], 0.5, 0.6)),
    }
    for nm, (sl, p, f) in specs.items():
        d0 = ac.run_sleeve(sess, sl, costs=zero, ctx=ctx, **p)
        proj = d0.pnl / sess.mult
        mine = np.zeros(len(sess)); ntr = np.zeros(len(sess), int)
        for i in np.flatnonzero(ok):
            mine[i], ntr[i] = f(i, days[dates[i]])
        diff = np.abs(mine - proj)
        bad = np.flatnonzero(diff > 1e-9)
        print(nm.ljust(10), 'sessions', int(ok.sum()), 'pnl mismatches', len(bad), ' trade-count mismatches', int((ntr != d0.trades).sum()),
              ' sum pts mine %.2f proj %.2f' % (mine.sum(), proj.sum()), ' e.g.', [(str(pd.Timestamp(sess.dates[j]).date()), round(mine[j], 2), round(proj[j], 2)) for j in bad[:3]])
