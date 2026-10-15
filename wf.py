"""Walk-forward strategy streams: each calendar year trades the variant that scored best on the
previous `lookback` years (ATR-normalised Sharpe, using only data available before that year)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import Sessions


def _sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.size > 20 and x.std() > 0 else 0.0


def walk_forward_series(sess: Sessions, family: str, grid: list, first_year: int = 2015,
                        lookback: int = 5, cash_only: bool = True):
    """Return (DailySeries covering first_year onward, {year: chosen params})."""
    yrs = pd.DatetimeIndex(sess.dates).year.to_numpy()
    cand = [p for p in grid if not (cash_only and family == "vwap_atr" and p.get("start_min", 0) < st.CASH_OPEN)]
    runs = []
    for p in cand:
        d = st.run(sess, family, **p)
        r = np.where(d.traded_ok, d.pnl / (d.atr * d.mult), 0.0)
        runs.append((p, d, r))
    n = len(sess)
    pnl, low, trades = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int32)
    ok = np.zeros(n, dtype=bool)
    atr = np.full(n, np.nan)
    chosen = {}
    for Y in range(first_year, int(yrs.max()) + 1):
        train = (yrs >= Y - lookback) & (yrs <= Y - 1)
        best = max(runs, key=lambda t: _sharpe(t[2][train & t[1].traded_ok]))
        p, d, _ = best
        te = yrs == Y
        pnl[te], low[te], trades[te], ok[te], atr[te] = d.pnl[te], d.minpnl[te], d.trades[te], d.traded_ok[te], d.atr[te]
        chosen[Y] = p
    keep = yrs >= first_year
    out = st.DailySeries(sess.dates[keep], pnl[keep], low[keep], trades[keep], ok[keep], atr[keep], sess.mult)
    return out, chosen
