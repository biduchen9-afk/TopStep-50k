"""Helpers to run the bar-by-bar sleeves over stored sessions (market_data.Sessions) and compare them with the vectorised back-test."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np

from paper.history import History
from paper.sleeves import make_sleeves


def ctx_for(hist: History, roll: bool) -> dict:
    c = hist.context()
    c["ok"] = (not roll) and math.isfinite(c["atr"]) and math.isfinite(c["prev_close"])
    c["roll"] = roll
    return c


def run_session(sess, i: int, hist: History, sleeves: Optional[dict] = None, cost_pts: float = 0.0) -> Dict[str, tuple]:
    """Feed session i bar by bar to fresh sleeves with the context of the sessions before it. Returns {name: (pnl_pts, low_pts, trades)}; does NOT add the session to `hist`."""
    sleeves = sleeves or make_sleeves(cost_pts)
    ctx = ctx_for(hist, bool(sess.roll[i]))
    for s in sleeves.values():
        s.reset(ctx)
    o, h, l, c, v, m = sess.day(i)
    for j in range(len(c)):
        for s in sleeves.values():
            s.on_bar(j, int(m[j]), o[j], h[j], l[j], c[j], v[j])
    for s in sleeves.values():
        s.finish()
    return {k: s.result() for k, s in sleeves.items()}
