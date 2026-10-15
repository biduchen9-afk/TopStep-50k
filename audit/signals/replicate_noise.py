"""Independent first-principles replication of the NQ noise-area sleeve from the RAW txt file.
Does NOT import any project strategy code (only used afterwards, in compare(), to read project output)."""
import numpy as np
import pandas as pd
from load_raw import load

STEP, K = 30, 12


def build_days(df):
    """dict: label date -> dict(md=minute-of-day array (ET wall clock), o,h,l,c,v) for the 18:00(prev)->16:00 session."""
    ts = df.index
    hr = ts.hour.to_numpy()
    keep = (hr >= 18) | (hr < 16)
    d = df[keep]
    ts = d.index
    lab = (ts.normalize() + pd.to_timedelta(np.where(ts.hour >= 18, 1, 0), unit='D'))
    wd = lab.dayofweek.to_numpy()
    m = wd < 5
    d, lab = d[m], lab[m]
    days = {}
    labn = lab.to_numpy()
    cuts = np.flatnonzero(labn[1:] != labn[:-1]) + 1
    st = np.r_[0, cuts]; en = np.r_[cuts, len(d)]
    arr = {k: d[k].to_numpy(float) for k in 'ohlcv'}
    md = (d.index.hour * 60 + d.index.minute).to_numpy()
    for a, b in zip(st, en):
        if b - a < 500:
            continue
        days[pd.Timestamp(labn[a])] = dict(md=md[a:b], o=arr['o'][a:b], h=arr['h'][a:b], l=arr['l'][a:b], c=arr['c'][a:b], v=arr['v'][a:b])
    return days


def cash_part(day):
    """bars of the cash day: minute-of-day in [9:30, 15:55) -> excludes the overnight part."""
    md = day['md']
    s = (md >= 570) & (md < 955)
    return {k: v[s] for k, v in day.items()}


def moves(day):
    """|close of last bar before T / open(9:30) - 1| for T = 10:00 .. 15:30, nan if not available."""
    cp = cash_part(day)
    out = np.full(K, np.nan)
    if len(cp['md']) == 0:
        return out
    op = cp['o'][0]
    for k in range(1, K + 1):
        T = 570 + STEP * k
        sel = np.flatnonzero(cp['md'] < T)
        if len(sel) > 0 and cp['md'][0] >= 570:
            out[k - 1] = abs(cp['c'][sel[-1]] / op - 1.0)
    return out


def noise_day(day, prev_close, sigma, max_trades=3):
    """Return (points pnl, trades list) for one session using only the documented rules."""
    cp = cash_part(day)
    md, o, h, l, c, v = (cp[k] for k in ('md', 'o', 'h', 'l', 'c', 'v'))
    if len(md) < 100 or not np.isfinite(prev_close):
        return 0.0, []
    op = o[0]
    hi_b, lo_b = max(op, prev_close), min(op, prev_close)
    tp = (h + l + c) / 3.0
    cum_pv = np.cumsum(tp * v)
    cum_v = np.maximum(np.cumsum(v), 1.0)
    pos, entry, ei = 0, 0.0, 0
    trades, pnl = [], 0.0
    for k in range(1, K + 1):
        T = 570 + STEP * k
        j = int(np.searchsorted(md, T))            # first bar at/after T = the bar we would trade at
        if j >= len(md) - 1 or j == 0:
            break
        sg = sigma[k - 1]
        if not np.isfinite(sg):
            continue
        last = j - 1                                # last completed bar
        px = c[last]
        vwap = cum_pv[last] / cum_v[last]
        ub, lb = hi_b * (1 + sg), lo_b * (1 - sg)
        if pos == 0:
            if len(trades) < max_trades:
                if px > ub:
                    pos, entry, ei = 1, o[j], j
                elif px < lb:
                    pos, entry, ei = -1, o[j], j
        elif pos > 0 and px < max(ub, vwap):
            trades.append((1, entry, o[j], md[ei], md[j])); pnl += o[j] - entry; pos = 0
        elif pos < 0 and px > min(lb, vwap):
            trades.append((-1, entry, o[j], md[ei], md[j])); pnl += entry - o[j]; pos = 0
    if pos != 0:
        px = c[-1]                                   # close of the 15:54 bar
        trades.append((pos, entry, px, md[ei], md[-1] + 1)); pnl += pos * (px - entry)
    return pnl, trades


def run_all(mkt='nq'):
    df = load(mkt)
    days = build_days(df)
    dates = sorted(days)
    mv = np.array([moves(days[d]) for d in dates])
    res = {}
    for i, d in enumerate(dates):
        if i == 0:
            continue
        prev_close = days[dates[i - 1]]['c'][-1]
        sigma = np.full(K, np.nan)
        if i >= 14:
            with np.errstate(all='ignore'):
                sigma = np.nanmean(mv[i - 14:i], axis=0)
        res[d] = noise_day(days[d], prev_close, sigma)
    return dates, res


if __name__ == '__main__':
    dates, res = run_all('nq')
    import pickle
    pickle.dump((dates, res), open('/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/my_noise_nq.pkl', 'wb'))
    print(len(dates), len(res))
