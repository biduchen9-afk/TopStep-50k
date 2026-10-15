"""Pre-declared switching rules for a sleeve portfolio. Every rule uses only information strictly BEFORE the day it applies to
(rolling statistics are shifted by one session). Four rules are declared; no others are tried.

  static      the dev-weights portfolio, always on (baseline)
  gate120     a sleeve is ON only when its own trailing 120-session Sharpe (ATR-normalised daily R) > 0
  gate250     same with 250 sessions
  tilt        weight multiplier = 0.5 + 0.5 * clip(trailing-250 Sharpe, 0, 1): half the weight stays on, the rest follows recent performance
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from portfolio import Sleeve, align, basket, trailing_sharpe_mask


def trailing_sharpe(sleeves: List[Sleeve], window: int) -> Dict[str, np.ndarray]:
    idx = align(sleeves)
    out = {}
    for s in sleeves:
        r = np.where(s.ok, s.pnl / np.where(s.ok, s.atr, 1.0), 0.0)
        ser = pd.Series(r, index=pd.DatetimeIndex(s.dates)).reindex(idx).fillna(0.0)
        mu = ser.rolling(window, min_periods=window // 2).mean().shift(1)
        sd = ser.rolling(window, min_periods=window // 2).std().shift(1)
        out[s.name] = (mu / sd * np.sqrt(252)).to_numpy()
    return out


def switched_basket(sleeves: List[Sleeve], rule: str, weights: Dict[str, float], name: str):
    if rule == "static":
        return basket(sleeves, weights=weights, name=name)
    if rule == "gate120":
        return basket(sleeves, masks=trailing_sharpe_mask(sleeves, 120, 0.0), weights=weights, name=name)
    if rule == "gate250":
        return basket(sleeves, masks=trailing_sharpe_mask(sleeves, 250, 0.0), weights=weights, name=name)
    if rule == "tilt":
        sh = trailing_sharpe(sleeves, 250)
        tvw = {k: 0.5 + 0.5 * np.clip(np.nan_to_num(v, nan=0.0), 0.0, 1.0) for k, v in sh.items()}
        return basket(sleeves, weights=weights, tvw=tvw, name=name)
    raise ValueError(rule)
