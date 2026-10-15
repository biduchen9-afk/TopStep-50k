"""Independent re-implementation of the noise_area sleeve with knobs for placebo tests.

na_day(...) mirrors academic_strategies.noise_area_session but supports:
  dirmult  : -1 reverses every position (sign-flip placebo)
  delay    : execute every entry/exit `delay` bars after the decision bar (delay test)
  events   : if a list is given, decisions are recorded as (smin_of_decision_bar, action) action in {+1,-1 enter, 0 exit}
  replay   : if a list of events is given, ignore own signals and execute these (time-shift placebo)
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import intraday_strategies as st
import academic_strategies as ac
from intraday_strategies import CASH_OPEN, FLAT_BY, Costs, DailySeries
from academic_strategies import STEP, K_DEC
from market_data import Sessions


def _pnl_leg(side, entry, exit_px, l, h, e_idx, last_idx):
    last_idx = max(last_idx, e_idx)
    if side > 0:
        return l[e_idx:last_idx + 1].min() - entry, exit_px - entry
    return entry - h[e_idx:last_idx + 1].max(), entry - exit_px


def na_day(o, h, l, c, v, smin, prev_close, sigma, cost_pts, allow=0, max_trades=3, dirmult=1, delay=0, events=None, replay=None):
    a0 = int(np.searchsorted(smin, CASH_OPEN))
    end = int(np.searchsorted(smin, FLAT_BY))
    if not np.isfinite(prev_close) or a0 > len(c) - 20 or end - a0 < 100:
        return 0.0, 0.0, 0
    op = o[a0]
    realized, minp, trades = 0.0, 0.0, 0
    state = dict(pos=0, entry=0.0, e_idx=0)

    def enter(side, idx):
        j = min(idx + delay, end - 2)
        state.update(pos=side, entry=o[j], e_idx=j)

    def leave(idx):
        nonlocal realized, minp
        j = min(idx + delay, end - 1)
        j = max(j, state["e_idx"] + 1) if j <= state["e_idx"] else j
        worst, pnl = _pnl_leg(state["pos"], state["entry"], o[j], l, h, state["e_idx"], j - 1)
        minp = min(minp, realized + min(worst, pnl))
        realized += pnl - cost_pts
        state["pos"] = 0

    if replay is not None:
        for (tm, act) in replay:
            idx = int(np.searchsorted(smin, tm))
            if idx >= end - 1:
                break
            if act != 0 and state["pos"] == 0:
                enter(act * dirmult, idx); trades += 1
            elif act == 0 and state["pos"] != 0:
                leave(idx)
        if state["pos"] != 0:
            worst, pnl = _pnl_leg(state["pos"], state["entry"], c[end - 1], l, h, state["e_idx"], end - 1)
            minp = min(minp, realized + min(worst, pnl)); realized += pnl - cost_pts
        return realized, min(minp, realized), trades

    hi_b, lo_b = max(op, prev_close), min(op, prev_close)
    tp = (h[a0:] + l[a0:] + c[a0:]) / 3.0
    cv = np.cumsum(v[a0:])
    vw = np.cumsum(tp * v[a0:]) / np.maximum(cv, 1.0)
    sig_pos = 0   # position in SIGNAL space (before dirmult)
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
        if sig_pos == 0 and trades < max_trades:
            if px > ub and allow >= 0:
                sig_pos = 1
            elif px < lb and allow <= 0:
                sig_pos = -1
            if sig_pos != 0:
                trades += 1
                enter(sig_pos * dirmult, idx)
                if events is not None:
                    events.append((smin[idx], sig_pos))
        elif sig_pos > 0 and px < max(ub, vwap):
            sig_pos = 0; leave(idx)
            if events is not None:
                events.append((smin[idx], 0))
        elif sig_pos < 0 and px > min(lb, vwap):
            sig_pos = 0; leave(idx)
            if events is not None:
                events.append((smin[idx], 0))
    if state["pos"] != 0:
        worst, pnl = _pnl_leg(state["pos"], state["entry"], c[end - 1], l, h, state["e_idx"], end - 1)
        minp = min(minp, realized + min(worst, pnl)); realized += pnl - cost_pts
        if events is not None:
            events.append((smin[end - 1], 0))
    return realized, min(minp, realized), trades


def run_na(sess: Sessions, ctx=None, costs=Costs(), shift=0, perm=None, **kw) -> DailySeries:
    """Run my noise_area over all sessions. shift=s: use the signal events of session i-s on session i.
    perm: array mapping session i -> session whose events to use (overrides shift)."""
    ctx = ctx or ac.context(sess)
    n = len(sess)
    cp = costs.points(sess)
    atr = st.trailing_atr(ctx["ranges"], sess.roll)
    ok = ~sess.roll & ~np.isnan(atr)
    pnl, mn, tr = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int32)
    ev = None
    if shift or perm is not None:
        ev = [None] * n
        for i in np.flatnonzero(ok):
            o, h, l, c, v, m = sess.day(i)
            e = []
            na_day(o, h, l, c, v, m, ctx["prev_close"][i], ctx["sigma"][i], cp, events=e, **{k: x for k, x in kw.items() if k in ("allow", "max_trades")})
            ev[i] = e
    for i in np.flatnonzero(ok):
        o, h, l, c, v, m = sess.day(i)
        if ev is not None:
            j = perm[i] if perm is not None else i - shift
            if j < 0 or ev[j] is None:
                continue
            r = na_day(o, h, l, c, v, m, ctx["prev_close"][i], ctx["sigma"][i], cp, replay=ev[j], **{k: x for k, x in kw.items() if k in ("dirmult", "delay")})
        else:
            r = na_day(o, h, l, c, v, m, ctx["prev_close"][i], ctx["sigma"][i], cp, **kw)
        pnl[i], mn[i], tr[i] = r[0] * sess.mult, r[1] * sess.mult, r[2]
    return DailySeries(sess.dates, pnl, mn, tr, ok, atr, sess.mult)


def stats(d: DailySeries, mask=None):
    """Return dict: ATR-normalised annual Sharpe, raw $ Sharpe, mean $/day (per mini), total $ , t-stat, trades/day."""
    m = d.traded_ok if mask is None else (d.traded_ok & mask)
    x = d.pnl[m]
    r = d.pnl[m] / (d.atr[m] * d.mult)
    sd = r.std(ddof=1)
    return dict(sh_atr=float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                sh_raw=float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std() > 0 else 0.0,
                mean_usd=float(x.mean()), total_usd=float(x.sum()), t=float(r.mean() / sd * np.sqrt(len(r))) if sd > 0 else 0.0,
                trades_day=float(d.trades[m].mean()), n=int(m.sum()))
