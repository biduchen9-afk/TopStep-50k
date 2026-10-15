"""Pre-registration 17 (with Amendment B): re-basing the cost model to today's price level.

The sleeves are charged a fixed $1.72 per micro round trip ($1.22 all-in commission + 1 tick, i.e. 1.22 + 0.50 k with k ticks of slippage). Position sizes are set from ATR, so that cost per unit of risk
was ~2.5% of ATR14 in 2013 and is ~0.2% now, because NQ rose ~15-fold. Three bases (`rebase(ST, ticks, mode)`), all built straight from sleeve runs with zero cost (gross) and with the modelled cost (net):
  dev        costs as modelled except `ticks` of slippage (no price change)                        pnl = gross - m*cost                       (ATR unchanged)
  cost_only  (PR17 as first run) cost scaled by f_t only; contract counts stay at dev-price levels   pnl = gross - m*f*cost                     (ATR unchanged, fx = f)
  exact      (Amendment B) the same percentage moves at today's price level: P&L, the intraday low and ATR scale by 1/f_t, per-contract dollar costs stay      pnl = gross/f - m*cost   (ATR/f, fx None)
with f_t = min(1, P_t / P_ref), P_t = unadjusted NQ session close from the stored raw data, P_ref = mean P_t over 2024-01-01 .. 2026-07-02, m = (1.22 + 0.50 ticks) / 1.72.
'exact' is the faithful one: with fewer contracts at a higher price the 150-micro Combine cap and the XFA micro ladder bind far less often than at dev-price contract counts. The R-unit statistic (P&L / ATR) is identical in 'cost_only' and 'exact'.
The sleeves' entries and exits do not depend on cost, so P&L is linear in cost. The sleeve's intraday low excludes the cost of the trade still open: it is shifted only by the cost realised inside it (gross low - net low).
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
CACHE = Path("results/retest_grossnet.pkl")
RAW_UNADJ = Path(os.environ.get("TOPSTEP_UNADJ_DIR", "/home/user/data_raw"))


def modelled_cost_usd() -> float:
    return COMM + TICK_USD * MODELLED_TICKS


def _require_live() -> None:
    if not str(md.RAW_DIR).rstrip("/").endswith("data_raw_live"):
        raise RuntimeError(f"retest_cost needs the repaired data (TOPSTEP_RAW_DIR=/home/user/data_raw_live set BEFORE anything imports market_data); market_data.RAW_DIR is {md.RAW_DIR}")


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


def gross_net() -> Dict[str, dict]:
    """Per sleeve, aligned on the sleeve's own dates: gross P&L and low (zero cost), the P&L cost (gross - net) and the cost realised inside the low (gross low - net low), all $ per micro."""
    _require_live()
    if CACHE.exists():
        c = pickle.load(open(CACHE, "rb"))
        if c.get("raw_dir") == str(md.RAW_DIR):
            return c["sleeves"]
    cfg = load_config(); net_costs = costs_from_config(cfg); zero = st.Costs(commission_rt=0.0, slippage_ticks_rt=0.0)
    out = {}
    for g, (name, sleeve, params) in SLEEVES.items():
        gn = make_sleeve("nq", sleeve, params, costs=zero, name=name); nn = make_sleeve("nq", sleeve, params, costs=net_costs, name=name)
        assert np.array_equal(gn.dates, nn.dates) and np.array_equal(gn.ok, nn.ok)
        out[g] = dict(dates=nn.dates, gpnl=gn.pnl.copy(), glow=gn.low.copy(), cost=(gn.pnl - nn.pnl), cin=(gn.low - nn.low), ok=nn.ok.copy())
    pickle.dump(dict(raw_dir=str(md.RAW_DIR), sleeves=out), open(CACHE, "wb"))
    return out


def price_factor(dates: np.ndarray) -> np.ndarray:
    pref, _ = reference_price()
    p = pd.Series(unadjusted_close(dates), index=pd.DatetimeIndex(dates)).ffill().bfill()
    return np.minimum(1.0, p.values / pref)


def rebase(ST: Dict[str, Stream], ticks: float = 1.0, mode: str = "exact") -> Dict[str, Stream]:
    if mode not in ("dev", "cost_only", "exact"):
        raise ValueError(mode)
    gnn = gross_net(); dates = next(iter(ST.values())).dates
    f = np.ones(len(dates)) if mode == "dev" else price_factor(dates)
    m = (COMM + TICK_USD * ticks) / modelled_cost_usd()
    out = {}
    for g, s in ST.items():
        d = gnn[g]; pos = pd.Index(pd.DatetimeIndex(d["dates"])).get_indexer(pd.DatetimeIndex(s.dates)); has = pos >= 0; pos = np.maximum(pos, 0)
        take = lambda a: np.where(has, a[pos], 0.0) * s.ok
        gpnl, glow, cost, cin = take(d["gpnl"]), take(d["glow"]), take(d["cost"]), take(d["cin"])
        if mode == "exact":
            pnl = gpnl / f - m * cost
            low = np.minimum(np.minimum(glow / f - m * cin, pnl), 0.0)
            atr = s.atr_usd / f
        else:
            sc = m * f
            pnl = gpnl - sc * cost
            low = np.minimum(np.minimum(glow - sc * cin, pnl), 0.0)
            atr = s.atr_usd
        out[g] = Stream(s.name + f"|{mode}{ticks:g}t", s.dates, pnl, low, atr, s.ok, fx=f if mode == "cost_only" else None)
    return out
