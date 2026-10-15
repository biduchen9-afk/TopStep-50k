"""Pre-registration 17: cost re-basing. The sleeves are charged a fixed $1.72 per micro round trip ($1.22 all-in commission + 1 tick). With position sizes set from ATR, that cost
per unit of risk was ~2.5% of ATR14 in 2013 and is ~0.2% now because NQ rose ~15-fold. `rebase` re-charges it as if the same percentage moves had happened at today's price level:
    cost_t(new) = cost_t(modelled) * m_k * f_t,    f_t = min(1, P_t / P_ref),   m_k = (1.22 + 0.50 k) / 1.72   (k = ticks of slippage per round trip; 1 = as modelled)
P_t = unadjusted NQ session close (stored raw data, not the back-adjusted series), P_ref = mean P_t over 2024-01-01 .. 2026-07-02.
The sleeves' entries and exits do not depend on cost, so P&L is linear in cost: pnl_new = pnl + cost_modelled * (1 - m_k f_t); the worst intraday mark is shifted the same way (capped at the day's P&L and 0).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import numpy as np
import pandas as pd

import intraday_strategies as st
import market_data as md
from portfolio import make_sleeve
from pricing_config import load_config
from real_engine import Stream, costs_from_config

SLEEVES = {"noise": ("nq_noise", "noise_area", {}), "gap": ("nq_gap30", "vv_filt", dict(filt="gap30", k=0.3, s=0.2))}
COMM, TICK_USD, MODELLED_TICKS = 1.22, 0.50, 1.0
CACHE = Path("results/retest_costR.pkl")
RAW_UNADJ = Path(os.environ.get("TOPSTEP_UNADJ_DIR", "/home/user/data_raw"))


def modelled_cost_usd() -> float:
    return COMM + TICK_USD * MODELLED_TICKS


def unadjusted_close(dates: np.ndarray) -> np.ndarray:
    """Unadjusted NQ session close on the given dates (NaN where there is no NQ session); read from the stored raw data."""
    sess = md.build_sessions("nq", raw_dir=RAW_UNADJ)
    last = np.array([sess.c[sess.offsets[i + 1] - 1] for i in range(len(sess))])
    pos = pd.Index(sess.dates).get_indexer(dates)
    return np.where(pos >= 0, last[np.maximum(pos, 0)], np.nan)


def reference_price() -> Tuple[float, pd.Series]:
    sess = md.build_sessions("nq", raw_dir=RAW_UNADJ)
    last = pd.Series([sess.c[sess.offsets[i + 1] - 1] for i in range(len(sess))], index=pd.DatetimeIndex(sess.dates))
    return float(last.loc["2024-01-01":"2026-07-02"].mean()), last


def cost_R() -> Dict[str, pd.Series]:
    """Modelled cost per micro-day in R units (R = P&L / ATR in dollars): gross minus net, from the sleeves run with zero costs and with the modelled costs."""
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    cfg = load_config(); net_costs = costs_from_config(cfg); zero = st.Costs(commission_rt=0.0, slippage_ticks_rt=0.0)
    out = {}
    for g, (name, sleeve, params) in SLEEVES.items():
        rr = {}
        for lab, cs in (("net", net_costs), ("gross", zero)):
            s = make_sleeve("nq", sleeve, params, costs=cs, name=name)
            rr[lab] = pd.Series(np.where(s.ok, s.pnl / np.where(s.ok, s.atr, 1.0), 0.0), index=pd.DatetimeIndex(s.dates))
        out[g] = rr["gross"] - rr["net"]
    pickle.dump(out, open(CACHE, "wb"))
    return out


def price_factor(dates: np.ndarray) -> np.ndarray:
    pref, _ = reference_price()
    p = pd.Series(unadjusted_close(dates), index=pd.DatetimeIndex(dates)).ffill().bfill()
    return np.minimum(1.0, p.values / pref)


def rebase(ST: Dict[str, Stream], ticks: float = 1.0, to_today: bool = True) -> Dict[str, Stream]:
    """Streams re-charged at `ticks` of slippage per round trip and, if to_today, at today's cost-to-ATR (see module doc). fx is set so tick-based liquidation overshoot scales the same way."""
    cR = cost_R(); dates = next(iter(ST.values())).dates
    f = price_factor(dates) if to_today else np.ones(len(dates))
    m = (COMM + TICK_USD * ticks) / modelled_cost_usd()
    out = {}
    for g, s in ST.items():
        cost_usd = cR[g].reindex(pd.DatetimeIndex(s.dates)).fillna(0.0).values * np.where(s.ok, s.atr_usd, 0.0)
        delta = cost_usd * (1.0 - m * f)
        pnl = s.pnl + delta
        low = np.minimum(np.minimum(s.low + delta, pnl), 0.0)
        out[g] = Stream(s.name + f"|rebase{ticks:g}t{'' if to_today else 'dev'}", s.dates, pnl, low, s.atr_usd, s.ok, fx=f if to_today else None)
    return out
