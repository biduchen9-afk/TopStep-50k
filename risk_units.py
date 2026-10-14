"""Alternative risk units for position sizing. The engine sizes micros as risk$ / stream.atr_usd, so swapping `atr_usd`
changes how risk is scaled without touching the engine.

  atr      trailing 14-session range (current default)
  pnlstd   trailing realised std of the strategy's own daily $ P&L per micro (strictly past data), rescaled so its long-run
           mean equals the ATR's -> same average size, different timing of size changes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from real_engine import Stream


def with_risk_unit(s: Stream, kind: str = "pnlstd", window: int = 40, floor_q: float = 0.1, dev_end=np.datetime64("2022-12-31")) -> Stream:
    if kind == "atr":
        return s
    pnl = pd.Series(np.where(s.ok, s.pnl, np.nan))
    if kind == "pnlstd":
        sd = pnl.rolling(window, min_periods=window // 2).std().shift(1)
    elif kind == "blend":                      # geometric mean of ATR and realised std
        sd = np.sqrt(pnl.rolling(window, min_periods=window // 2).std().shift(1) * pd.Series(s.atr_usd) * 0.3)
    else:
        raise ValueError(kind)
    sd = sd.to_numpy()
    good = np.isfinite(sd) & (sd > 0) & s.ok
    dv = good & (s.dates <= dev_end)              # scale and floor are fitted on DEV data only
    scale = np.nanmean(s.atr_usd[dv]) / np.nanmean(sd[dv])
    unit = sd * scale
    lo = np.nanquantile(unit[dv], floor_q)
    unit = np.where(good, np.maximum(unit, lo), np.nan)
    ok = s.ok & np.isfinite(unit)
    return Stream(s.name + f"|{kind}{window}", s.dates, s.pnl, s.low, unit, ok)
