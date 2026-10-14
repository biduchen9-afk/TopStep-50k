"""Independent BAR-BY-BAR (streaming; decisions use only bars <= current bar) replication of NQ vwap_atr
(k=0.3, s=0.2, cash start 09:30, max 3 trades). Reads the RAW txt. Roll flags come from the project (analysed separately)."""
import sys, pickle
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from replicate_noise import build_days
from load_raw import load, SCR


def stream_day(day, atr, k=0.3, s=0.2, start_key=570, max_trades=3):
    md, o, h, l, c, v = (day[x] for x in ('md', 'o', 'h', 'l', 'c', 'v'))
    key = np.where(md >= 18 * 60, md - 1440, md)           # chronological minute key, 18:00 -> -360
    end = int(np.sum(key < 955))                            # tradable bars: before 15:55
    if end < 100:
        return 0.0, 0
    band, stop_d = k * atr, s * atr
    cum_pv = cum_v = 0.0
    pos = 0; entry = stop = 0.0
    pend_exit = False; pend_dir = 0
    trades = 0; pnl = 0.0; first = True
    prev_up = prev_dn = False
    for b in range(end):
        stopped_now = False
        # ---- start of bar b: execute orders decided at close of b-1
        if pos != 0 and pend_exit:
            pnl += pos * (o[b] - entry); pos = 0
        pend_exit = False
        if pos == 0 and pend_dir != 0:
            pos = pend_dir; entry = o[b]; stop = entry - stop_d if pos > 0 else entry + stop_d
            trades += 1
        pend_dir = 0
        # ---- during bar b: resting stop
        if pos > 0 and l[b] <= stop:
            pnl += min(stop, o[b]) - entry; pos = 0; stopped_now = True
        elif pos < 0 and h[b] >= stop:
            pnl += entry - max(stop, o[b]); pos = 0; stopped_now = True
        # ---- close of bar b: VWAP now includes bar b
        cum_pv += (h[b] + l[b] + c[b]) / 3.0 * v[b]; cum_v += v[b]
        vw = cum_pv / max(cum_v, 1.0)
        up = c[b] > vw + band; dn = c[b] < vw - band
        if pos != 0:
            if ((pos > 0 and c[b] < vw) or (pos < 0 and c[b] > vw)) and b + 1 < end:
                pend_exit = True
        elif (not stopped_now) and trades < max_trades and key[b] >= start_key and b <= end - 2:
            su, sd = (up, dn) if first else (up and not prev_up, dn and not prev_dn)
            if su:
                pend_dir = 1; first = False
            elif sd:
                pend_dir = -1; first = False
        prev_up, prev_dn = up, dn
    if pos != 0:
        pnl += pos * (c[end - 1] - entry)
    return pnl, trades


def run_all(roll):
    df = load('nq')
    days = build_days(df)
    dates = sorted(days)
    rng = np.array([days[d]['h'].max() - days[d]['l'].min() for d in dates])
    atr = np.full(len(dates), np.nan); hist = []
    for i in range(len(dates)):
        if len(hist) >= 14:
            atr[i] = np.mean(hist[-14:])
        if not roll[i]:
            hist.append(rng[i])
    out = {}
    for i, d in enumerate(dates):
        if roll[i] or np.isnan(atr[i]):
            continue
        out[d] = stream_day(days[d], atr[i])
    return dates, out
