"""Quantify the roll-handling choice for the NQ headline sleeves.
Variants: A project (skip flagged), B skip TRUE splice sessions (00:00 UTC bar), C no skipping at all (raw, contaminated),
D back-adjust the splice and trade EVERY session (live-feasible: the roll date is known in advance)."""
import sys, copy, dataclasses
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions, _third_friday

DEV_END = np.datetime64('2022-12-31')


def sh(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.size > 20 and x.std() > 0 else 0.0


def is_edt(d):
    y = d.year
    mar = pd.Timestamp(y, 3, 1); s2 = mar + pd.Timedelta(days=(6 - mar.dayofweek) % 7 + 7)
    nov = pd.Timestamp(y, 11, 1); s1 = nov + pd.Timedelta(days=(6 - nov.dayofweek) % 7)
    return (d >= s2) and (d < s1)


def true_splices(sess, months=(3, 6, 9, 12)):
    """per expiry window: (session index, bar index, jump) of the largest 00:00-UTC jump."""
    idx = pd.DatetimeIndex(sess.dates)
    out = []
    for y in range(idx.year.min(), idx.year.max() + 1):
        for mth in months:
            tf = _third_friday(y, mth)
            lo, hi = tf - pd.Timedelta(days=12), tf + pd.Timedelta(days=1)
            best = None
            for i in np.flatnonzero((idx >= lo) & (idx <= hi)):
                eve = idx[i] - pd.Timedelta(days=1)
                tgt = 120 if is_edt(eve) else 60            # smin of 20:00 / 19:00 ET
                a, b = sess.offsets[i], sess.offsets[i + 1]
                m = sess.smin[a:b]
                j = a + int(np.searchsorted(m, tgt))
                if j <= a or j >= b or sess.smin[j] - sess.smin[j - 1] > 10:
                    continue
                jump = sess.o[j] - sess.c[j - 1]
                if best is None or abs(jump) > abs(best[2]):
                    best = (i, j, jump)
            if best is not None:
                out.append(best)
    return out


def variant(sess, kind, splices):
    s2 = dataclasses.replace(sess, o=sess.o.copy(), h=sess.h.copy(), l=sess.l.copy(), c=sess.c.copy(), roll=sess.roll.copy())
    if kind == 'B':
        s2.roll[:] = False
        for i, j, jump in splices:
            s2.roll[i] = True
    elif kind == 'C':
        s2.roll[:] = False
    elif kind == 'D':
        s2.roll[:] = False
        for i, j, jump in splices:
            a = sess.offsets[i]
            for arr in (s2.o, s2.h, s2.l, s2.c):
                arr[a:j] += jump
                if i > 0:
                    arr[sess.offsets[i - 1]:a] += jump          # previous session shifted too (prev_close consistent)
    return s2


def evaluate(sess, label, extra_gate=None):
    ctx = ac.context(sess)
    res = {}
    for name, p in [('noise_area', {}), ('vwap_atr', dict(k=0.3, s=0.2, start_min=st.CASH_OPEN))]:
        d = ac.run_sleeve(sess, name, ctx=ctx, **p)
        ok = d.traded_ok
        r = np.where(ok, d.pnl / (d.atr * d.mult), 0.0)
        dev = sess.dates <= DEV_END
        res[name] = (sh(r[ok]), sh(r[ok & dev]), sh(r[ok & ~dev]), int(ok.sum()))
    return res


if __name__ == '__main__':
    sess = build_sessions('nq')
    sp = true_splices(sess)
    flagged = set(np.flatnonzero(sess.roll)); truth = set(i for i, j, k in sp)
    print('windows', len(sp), 'true-splice sessions that project flagged:', len(flagged & truth), ' project flags not true splice:', len(flagged - truth))
    jumps = pd.Series({pd.Timestamp(sess.dates[i]): round(k, 2) for i, j, k in sp})
    print('splice jump (pts) by year (mean abs):', jumps.abs().groupby(jumps.index.year).mean().round(1).to_dict())
    for kind, lab in [('A', 'project flags (skip 65)'), ('B', 'skip TRUE splice sessions'), ('C', 'no skipping, raw'), ('D', 'back-adjusted, trade all')]:
        s2 = sess if kind == 'A' else variant(sess, kind, sp)
        r = evaluate(s2, lab)
        print(kind, lab.ljust(28), {k: tuple(round(x, 3) if isinstance(x, float) else x for x in v) for k, v in r.items()})
