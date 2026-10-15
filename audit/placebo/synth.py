"""Synthetic zero-edge NQ minute data with the SAME per-minute point-volatility profile (and same volume, time stamps, roll flags)."""
import numpy as np
from market_data import Sessions


def make_synth(sess: Sessions, seed: int, mode: str = "bar") -> Sessions:
    """mode:
      'bar'     : every 1-minute bar (gap+body+wicks) gets an iid random sign  -> pure random walk, no autocorrelation, no drift
      'session' : one random sign per whole session path (keeps each session's own intraday dependence structure), gap sign independent
      'permute' : sessions are drawn in random order (with replacement-free permutation) AND sign-flipped per session
    """
    rng = np.random.default_rng(seed)
    n = len(sess)
    off = sess.offsets
    N = len(sess.c)
    order = np.arange(n)
    if mode == "permute":
        order = rng.permutation(n)
    # assemble flat arrays in the chosen session order
    idx = np.concatenate([np.arange(off[i], off[i + 1]) for i in order]) if mode == "permute" else np.arange(N)
    o, h, l, c = sess.o[idx], sess.h[idx], sess.l[idx], sess.c[idx]
    lens = np.diff(off)[order]
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    first = np.zeros(N, dtype=bool); first[starts] = True
    # real previous close in the (possibly permuted) order -> gap component
    c_prev = np.r_[c[0], c[:-1]]
    # inside a permuted order the first bar's "gap" is the real own inter-session gap (o_0 - own prev session close)
    if mode == "permute":
        own_prev_close = np.r_[sess.c[0], sess.c[off[1:-1] - 1]]          # per session (real order)
        gap_first = sess.o[off[order]] - own_prev_close[order]
        c_prev = np.where(first, 0.0, c_prev)
        g = o - c_prev
        g[starts] = gap_first
    else:
        g = o - c_prev
        g[0] = 0.0
    body = c - o
    up = h - np.maximum(o, c)
    dn = np.minimum(o, c) - l
    if mode == "bar":
        e = rng.choice([-1.0, 1.0], size=N)
        f = rng.choice([-1.0, 1.0], size=N)
    else:
        se = rng.choice([-1.0, 1.0], size=n)
        e = np.repeat(se, lens)
        f = rng.choice([-1.0, 1.0], size=N)
    gsign = np.where(first, f, e)                      # session-gap sign independent of the intraday path sign
    gp = gsign * g
    bp = e * body
    step = gp + bp
    cn = np.cumsum(step)
    on = cn - bp
    upn = np.where(e > 0, up, dn)
    dnn = np.where(e > 0, dn, up)
    hn = np.maximum(on, cn) + upn
    ln = np.minimum(on, cn) - dnn
    shift = 3000.0 - min(ln.min(), cn.min())           # keep price positive; levels only matter through ratios
    on, cn, hn, ln = on + shift, cn + shift, hn + shift, ln + shift
    if mode == "permute":
        dates = sess.dates                            # calendar labels unchanged, paths are re-assigned to dates
        # sessions now hold whichever path was drawn; need smin/v/roll aligned with the NEW order of lengths -> use drawn session's own
        smin = sess.smin[idx]; v = sess.v[idx]; roll = sess.roll[order]
        offsets = np.r_[0, np.cumsum(lens)]
        return Sessions(sess.market, dates, offsets, on, hn, ln, cn, v, smin, roll, sess.mult, sess.tick)
    return Sessions(sess.market, sess.dates, off, on, hn, ln, cn, sess.v, sess.smin, sess.roll, sess.mult, sess.tick)
