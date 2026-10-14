"""Portfolio of sleeves as a 'micro basket': 1 micro contract of every active sleeve. All dollar figures are per micro.

Stream.atr_usd for a basket = sum of the active sleeves' dollar ATR per micro, so the engine's sizing (k baskets =
round(R / atr_usd)) keeps integer micros and the 1-micro minimum. The basket's intraday low is the SUM of the sleeves'
lows, a conservative bound (assumes the worst moments coincide)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions
from pricing_config import load_config
from real_engine import MICRO_PER_MINI, Stream, costs_from_config

DEV_END = np.datetime64("2022-12-31")
_CTX: Dict[str, dict] = {}
_EX: Dict[str, dict] = {}
_XS: Dict[str, dict] = {}


@dataclass
class Sleeve:
    name: str
    dates: np.ndarray
    pnl: np.ndarray      # $ per micro
    low: np.ndarray      # worst intraday MTM, $ per micro
    atr: np.ndarray      # $ value of the trailing ATR per micro
    ok: np.ndarray
    volratio: np.ndarray


def make_sleeve(market: str, sleeve: str, params: dict, gate: Optional[dict] = None, costs: Optional[st.Costs] = None, name: str = "") -> Sleeve:
    sess = build_sessions(market)
    if market not in _CTX:
        _CTX[market] = ac.context(sess)
    ctx = _CTX[market]
    if sleeve.startswith("lt_"):
        import lit_sleeves as lit
        d = lit.run_lit(sess, sleeve, costs or costs_from_config(load_config()), ctx, **params)
    elif sleeve.startswith("xs_"):
        import xs_sleeves as xsm
        if market not in _XS:
            _XS[market] = xsm.xs_context(sess)
        d = xsm.run_xs(sess, sleeve, costs or costs_from_config(load_config()), ctx, _XS[market], **params, **(gate or {}))
    elif sleeve.startswith("vv_"):
        import vwap_variants as vv
        if market not in _EX:
            _EX[market] = vv.extra_context(sess)
        d = vv.run_variant(sess, sleeve, costs or costs_from_config(load_config()), ctx, _EX[market], **params, **(gate or {}))
    else:
        d = ac.run_sleeve(sess, sleeve, costs=costs or costs_from_config(load_config()), ctx=ctx, **params, **(gate or {}))
    k = 1.0 / MICRO_PER_MINI
    atr_usd = np.where(d.traded_ok, d.atr * d.mult * k, np.nan)
    return Sleeve(name or f"{market}_{sleeve}", d.dates, np.where(d.traded_ok, d.pnl * k, 0.0), np.where(d.traded_ok, d.minpnl * k, 0.0),
                  atr_usd, d.traded_ok, ctx["volratio"])


def align(sleeves: List[Sleeve]) -> pd.DatetimeIndex:
    idx = sleeves[0].dates
    for s in sleeves[1:]:
        idx = np.union1d(idx, s.dates)
    return pd.DatetimeIndex(idx)


def _on(idx, s: Sleeve, arr, fill):
    out = pd.Series(arr, index=pd.DatetimeIndex(s.dates)).reindex(idx)
    return out.fillna(fill).to_numpy()


def basket(sleeves: List[Sleeve], masks: Optional[Dict[str, np.ndarray]] = None, name: str = "basket",
           weights: Optional[Dict[str, float]] = None, tvw: Optional[Dict[str, np.ndarray]] = None) -> Stream:
    """masks[name] is a bool array on the aligned date index: sleeve active on that day (default all).
    tvw[name] is an optional time-varying weight multiplier (float array on the aligned index, known BEFORE each day)."""
    idx = align(sleeves)
    n = len(idx)
    pnl, low, atr, cnt = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for s in sleeves:
        ok = _on(idx, s, s.ok.astype(float), 0.0) > 0
        act = ok if masks is None or s.name not in masks else (ok & masks[s.name])
        w = 1.0 if weights is None else weights.get(s.name, 1.0)
        if tvw is not None and s.name in tvw:
            w = w * np.asarray(tvw[s.name], dtype=float)
            act = act & (w > 0)
        pnl += w * np.where(act, _on(idx, s, s.pnl, 0.0), 0.0)
        low += w * np.where(act, _on(idx, s, s.low, 0.0), 0.0)
        atr += w * np.where(act, np.nan_to_num(_on(idx, s, s.atr, np.nan)), 0.0)
        cnt += act
    okb = (cnt > 0) & (atr > 0)
    return Stream(name, idx.to_numpy().astype("datetime64[D]"), pnl, low, np.where(okb, atr, np.nan), okb)


def r_series(st_: Stream) -> np.ndarray:
    return np.where(st_.ok, st_.pnl / np.where(st_.ok, st_.atr_usd, 1.0), 0.0)


def sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.size > 20 and x.std() > 0 else 0.0


def split_sharpes(st_: Stream) -> Dict[str, float]:
    r = r_series(st_); ok = st_.ok; dev = st_.dates <= DEV_END
    return dict(full=sharpe(r[ok]), dev=sharpe(r[ok & dev]), hold=sharpe(r[ok & ~dev]), active_days=int(ok.sum()))


def trailing_sharpe_mask(sleeves: List[Sleeve], window: int = 120, thr: float = 0.0) -> Dict[str, np.ndarray]:
    """Switch a sleeve ON only when its own trailing `window`-day (active days) Sharpe, computed strictly from the past, exceeds thr."""
    idx = align(sleeves)
    masks = {}
    for s in sleeves:
        r = np.where(s.ok, s.pnl / np.where(s.ok, s.atr, 1.0), 0.0)
        ser = pd.Series(r, index=pd.DatetimeIndex(s.dates)).reindex(idx).fillna(0.0)
        mu = ser.rolling(window, min_periods=window // 2).mean().shift(1)
        sd = ser.rolling(window, min_periods=window // 2).std().shift(1)
        masks[s.name] = ((mu / sd * np.sqrt(252)) > thr).fillna(False).to_numpy()
    return masks


def volratio_mask(s: Sleeve, idx, hi: bool, thr: float = 1.0) -> np.ndarray:
    vr = _on(idx, s, s.volratio, np.nan)
    return (vr >= thr) if hi else (vr < thr)


def dev_weights(sleeves: List[Sleeve], shrink: float = 0.5, iters: int = 4000) -> Dict[str, float]:
    """Long-only mean-variance weights (tangency) on DEV data only, shrunk toward equal weights. Weights are on ATR-normalised
    daily P&L, so they are comparable across markets; returned as multipliers of one micro-sleeve (mean 1)."""
    idx = align(sleeves)
    R = pd.DataFrame({s.name: pd.Series(np.where(s.ok, s.pnl / np.where(s.ok, s.atr, 1.0), 0.0), index=pd.DatetimeIndex(s.dates)).reindex(idx).fillna(0.0)
                      for s in sleeves})
    dev = R[R.index <= pd.Timestamp(DEV_END)]
    mu, cov = dev.mean().to_numpy(), dev.cov().to_numpy()
    n = len(mu)
    w = np.ones(n) / n
    for _ in range(iters):                       # projected gradient ascent on the Sharpe ratio, long-only
        var = w @ cov @ w
        g = mu / np.sqrt(var) - (w @ mu) * (cov @ w) / var ** 1.5
        w = np.maximum(w + 0.05 * g / (np.abs(g).max() + 1e-12) * 0.05, 0)
        w = w / w.sum()
    w = shrink * w + (1 - shrink) * np.ones(n) / n
    w = w / w.mean()
    return {s.name: float(x) for s, x in zip(sleeves, w)}
