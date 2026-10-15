"""Intraday strategies on full futures sessions (18:00 ET open -> 16:00 ET), per 1 full-size contract.

Every strategy is flat at the end of the session and returns, per session:
  pnl    net $ P&L per 1 mini (after commission + slippage)
  minpnl worst intraday mark-to-market $ P&L per 1 mini (<= 0), used for the daily loss limit
  trades number of round trips
Signals are read at a bar close and filled at the next bar open (no look-ahead). Stops fill at the
stop price, or at the open if the bar gaps through it. Sessions flagged as containing a contract
roll are not traded (pnl = 0).
Sizing is linear: multiply pnl / minpnl by the number of minis (micros = 0.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from market_data import Sessions

# Eastern-clock anchors expressed as minutes after the 18:00 ET open
CASH_OPEN = 15 * 60 + 30      # 09:30 ET
OR_END = CASH_OPEN + 30       # opening range 09:30-10:00 ET
FLAT_BY = 22 * 60 - 5         # 15:55 ET
ATR_LOOKBACK = 14
TRACE = None            # set to a list to record (entry_smin, exit_smin) of every trade (used by audit_trade_hours.py)


def _trace(smin, e, last):
    if TRACE is not None:
        TRACE.append((int(smin[e]), int(smin[last]) + 1))        # entry bar open .. close of the last bar held


@dataclass
class Costs:
    commission_rt: float = 2.50      # $ per round trip per mini (config friction)
    slippage_ticks_rt: float = 1.0   # ticks lost per round trip

    def points(self, sess: Sessions) -> float:
        return self.commission_rt / sess.mult + self.slippage_ticks_rt * sess.tick


@dataclass
class DailySeries:
    dates: np.ndarray
    pnl: np.ndarray
    minpnl: np.ndarray
    trades: np.ndarray
    traded_ok: np.ndarray     # False for sessions skipped (roll / not enough history)
    atr: np.ndarray           # trailing 14-session range in points (NaN until available)
    mult: float = 1.0

    def sharpe(self, mask: np.ndarray | None = None) -> float:
        m = self.traded_ok if mask is None else (self.traded_ok & mask)
        x = self.pnl[m]
        return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.size > 2 and x.std() > 0 else 0.0


def session_ranges(sess: Sessions) -> np.ndarray:
    r = np.empty(len(sess))
    for i in range(len(sess)):
        o, h, l, c, v, m = sess.day(i)
        r[i] = h.max() - l.min()
    return r


def trailing_atr(ranges: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Mean of the previous 14 non-roll session ranges (no look-ahead); NaN until available."""
    atr = np.full(len(ranges), np.nan)
    good = [i for i in range(len(ranges)) if not roll[i]]
    hist = []
    for i in range(len(ranges)):
        if len(hist) >= ATR_LOOKBACK:
            atr[i] = np.mean(hist[-ATR_LOOKBACK:])
        if not roll[i]:
            hist.append(ranges[i])
    return atr


def _first_true(mask: np.ndarray) -> int:
    return int(mask.argmax()) if mask.size and mask.any() else -1


def _exit_long_short(direction, price, stop_dist, e, o, h, l, c, vw, end):
    """Find the exit bar for a position entered at open of bar `e`. Returns (exit_price, last_bar_inclusive)."""
    n = len(c)
    if direction > 0:
        stop = price - stop_dist
        js = _first_true(l[e:end] <= stop)
        jc = _first_true(c[e:end] < vw[e:end])
    else:
        stop = price + stop_dist
        js = _first_true(h[e:end] >= stop)
        jc = _first_true(c[e:end] > vw[e:end])
    js = e + js if js >= 0 else 10 ** 9
    jc_exit = e + jc + 1 if jc >= 0 and e + jc + 1 < end else 10 ** 9   # cross at close j -> exit at open j+1
    if jc_exit <= js and jc_exit < 10 ** 9:
        return o[jc_exit], jc_exit - 1            # exits at the open: bar jc_exit not in the position
    if js < 10 ** 9:
        px = min(stop, o[js]) if direction > 0 else max(stop, o[js])
        return px, js
    return c[end - 1], end - 1


def vwap_atr_session(o, h, l, c, v, smin, atr, k, s, cost_pts, start_min=60, max_trades=3, allow=0, blackout=None):
    """Momentum: trade in the direction of price leaving an 18:00-anchored VWAP by k*ATR.

    Long when close > VWAP + k*ATR, short when close < VWAP - k*ATR (fresh crossings after the first
    trade). Exit on a stop of s*ATR or when price closes back through VWAP; flat by 15:55 ET.
    """
    n = len(c)
    end = int(np.searchsorted(smin, FLAT_BY))              # first bar at/after 15:55 ET is not traded
    if end < 100:
        return 0.0, 0.0, 0
    tp = (h + l + c) / 3.0
    cv = np.cumsum(v)
    vw = np.cumsum(tp * v) / np.maximum(cv, 1.0)
    band, stop_d = k * atr, s * atr
    up = c > vw + band
    dn = c < vw - band
    if allow > 0:
        dn = np.zeros_like(dn)
    elif allow < 0:
        up = np.zeros_like(up)
    for b0, b1 in (blackout or ()):
        bm = (smin >= b0) & (smin < b1)
        up = up & ~bm
        dn = dn & ~bm
    up_edge = up & ~np.r_[False, up[:-1]]
    dn_edge = dn & ~np.r_[False, dn[:-1]]
    t = int(np.searchsorted(smin, start_min))
    realized, minp, trades, first = 0.0, 0.0, 0, True
    while trades < max_trades and t < end - 2:
        mu = (up[t:end - 1] if first else up_edge[t:end - 1])
        md = (dn[t:end - 1] if first else dn_edge[t:end - 1])
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
        px, last = _exit_long_short(direction, price, stop_d, e, o, h, l, c, vw, end)
        last = max(last, e)
        _trace(smin, e, last)
        if direction > 0:
            worst = l[e:last + 1].min() - price
            pnl_pts = px - price
        else:
            worst = price - h[e:last + 1].max()
            pnl_pts = price - px
        minp = min(minp, realized + min(worst, pnl_pts))
        realized += pnl_pts - cost_pts
        trades += 1
        t = last + 1
    return realized, min(minp, realized), trades


def orb_session(o, h, l, c, v, smin, atr, buf, s, cost_pts, or_min=30, allow=0):
    """Opening-range breakout (range of the first `or_min` minutes after 09:30 ET), one trade, stop s*ATR,
    flat by 15:55 ET."""
    end = int(np.searchsorted(smin, FLAT_BY))
    a, b = int(np.searchsorted(smin, CASH_OPEN)), int(np.searchsorted(smin, CASH_OPEN + or_min))
    if b - a < max(5, or_min // 2) or end - b < 30:
        return 0.0, 0.0, 0
    hi, lo = h[a:b].max(), l[a:b].min()
    bu = buf * atr
    iu = _first_true(c[b:end - 1] > hi + bu)
    idn = _first_true(c[b:end - 1] < lo - bu)
    if allow > 0:
        idn = -1
    elif allow < 0:
        iu = -1
    if iu < 0 and idn < 0:
        return 0.0, 0.0, 0
    if idn < 0 or (iu >= 0 and iu < idn):
        direction, sig = 1, b + iu
    else:
        direction, sig = -1, b + idn
    e = sig + 1
    price = o[e]
    stop_d = s * atr
    if direction > 0:
        js = _first_true(l[e:end] <= price - stop_d)
    else:
        js = _first_true(h[e:end] >= price + stop_d)
    if js >= 0:
        j = e + js
        px = min(price - stop_d, o[j]) if direction > 0 else max(price + stop_d, o[j])
        last = j
    else:
        px, last = c[end - 1], end - 1
    _trace(smin, e, last)
    if direction > 0:
        worst, pnl_pts = l[e:last + 1].min() - price, px - price
    else:
        worst, pnl_pts = price - h[e:last + 1].max(), price - px
    net = pnl_pts - cost_pts
    return net, min(worst - cost_pts, net, 0.0), 1


def random_session(o, h, l, c, v, smin, direction, cost_pts):
    """No-edge control: enter at the 09:30 ET open in a coin-flip direction, flat at 15:55 ET."""
    a, end = int(np.searchsorted(smin, CASH_OPEN)), int(np.searchsorted(smin, FLAT_BY))
    if end - a < 100:
        return 0.0, 0.0, 0
    price, px = o[a], c[end - 1]
    if direction > 0:
        worst, pnl_pts = l[a:end].min() - price, px - price
    else:
        worst, pnl_pts = price - h[a:end].max(), price - px
    net = pnl_pts - cost_pts
    return net, min(worst - cost_pts, net, 0.0), 1


def gap_session(o, h, l, c, v, smin, atr, thr, s, mode, cost_pts, allow=0):
    """Overnight-move momentum (mode=+1) or fade (mode=-1).

    The overnight move is close(09:29 ET) - open(18:00 ET). If |move| >= thr*ATR, trade at the 09:30 ET
    open in the direction of the move (momentum) or against it (fade), stop s*ATR, flat by 15:55 ET.
    """
    a, end = int(np.searchsorted(smin, CASH_OPEN)), int(np.searchsorted(smin, FLAT_BY))
    if a < 100 or end - a < 100:
        return 0.0, 0.0, 0
    move = c[a - 1] - o[0]
    if abs(move) < thr * atr:
        return 0.0, 0.0, 0
    direction = int(np.sign(move)) * mode
    if allow != 0 and direction != allow:
        return 0.0, 0.0, 0
    price, stop_d = o[a], s * atr
    if direction > 0:
        js = _first_true(l[a:end] <= price - stop_d)
    else:
        js = _first_true(h[a:end] >= price + stop_d)
    if js >= 0:
        j = a + js
        px = min(price - stop_d, o[j]) if direction > 0 else max(price + stop_d, o[j])
        last = j
    else:
        px, last = c[end - 1], end - 1
    _trace(smin, a, last)
    if direction > 0:
        worst, pnl_pts = l[a:last + 1].min() - price, px - price
    else:
        worst, pnl_pts = price - h[a:last + 1].max(), price - px
    net = pnl_pts - cost_pts
    return net, min(worst - cost_pts, net, 0.0), 1


def coin_session(o, h, l, c, v, smin, atr, direction, s, cost_pts):
    """No-edge lottery: coin-flip direction at the 09:30 ET open, optional stop s*ATR (None = no stop),
    flat by 15:55 ET. A stop truncates losses without creating edge; it only reshapes the distribution."""
    a, end = int(np.searchsorted(smin, CASH_OPEN)), int(np.searchsorted(smin, FLAT_BY))
    if a < 100 or end - a < 100:
        return 0.0, 0.0, 0
    price = o[a]
    js = -1
    if s is not None:
        stop_d = s * atr
        js = _first_true(l[a:end] <= price - stop_d) if direction > 0 else _first_true(h[a:end] >= price + stop_d)
    if js >= 0:
        j = a + js
        px = min(price - stop_d, o[j]) if direction > 0 else max(price + stop_d, o[j])
        last = j
    else:
        px, last = c[end - 1], end - 1
    if direction > 0:
        worst, pnl_pts = l[a:last + 1].min() - price, px - price
    else:
        worst, pnl_pts = price - h[a:last + 1].max(), price - px
    net = pnl_pts - cost_pts
    return net, min(worst - cost_pts, net, 0.0), 1


def run(sess: Sessions, name: str, costs: Costs = Costs(), seed: int = 0, **params) -> DailySeries:
    """Run one strategy over every session; returns $ per 1 mini."""
    n = len(sess)
    cp = costs.points(sess)
    ranges = session_ranges(sess)
    atr = trailing_atr(ranges, sess.roll)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1, 1], size=n)
    pnl, mn, tr = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int32)
    ok = ~sess.roll & ~np.isnan(atr)
    for i in np.flatnonzero(ok):
        o, h, l, c, v, m = sess.day(i)
        if name == "random":
            p, w, t = random_session(o, h, l, c, v, m, flips[i], cp)
        elif name == "vwap_atr":
            p, w, t = vwap_atr_session(o, h, l, c, v, m, atr[i], params["k"], params["s"], cp,
                                       start_min=params.get("start_min", 60))
        elif name == "orb":
            p, w, t = orb_session(o, h, l, c, v, m, atr[i], params["buf"], params["s"], cp,
                                  or_min=params.get("or_min", 30))
        elif name == "coin":
            p, w, t = coin_session(o, h, l, c, v, m, atr[i], flips[i], params.get("s"), cp)
        elif name == "gap":
            p, w, t = gap_session(o, h, l, c, v, m, atr[i], params["thr"], params["s"], params["mode"], cp)
        else:
            raise ValueError(name)
        pnl[i], mn[i], tr[i] = p * sess.mult, w * sess.mult, t
    return DailySeries(sess.dates, pnl, mn, tr, ok, atr, sess.mult)
