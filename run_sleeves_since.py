"""Descriptive: the three NQ sleeves (nq_noise, nq_vwap, nq_gap30) net of the modelled cost, dev against the hold-out and against 'since mid-2024' (2024-07-01 to 2026-07-02).
Units: $ per micro per day (net), and R = $ per micro / ATR in $ per micro (the unit the farm sizes on). Nothing is selected. Run with TOPSTEP_RAW_DIR=/home/user/data_raw_live."""
from __future__ import annotations

import os

os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import numpy as np
import pandas as pd

import intraday_strategies as st
import market_data as md
from portfolio import make_sleeve
from pricing_config import load_config
from real_engine import costs_from_config

assert str(md.RAW_DIR).rstrip("/").endswith("data_raw_live"), md.RAW_DIR
SL = {"nq_noise": ("noise_area", {}), "nq_vwap": ("vwap_atr", dict(k=0.3, s=0.2, start_min=930)), "nq_gap30": ("vv_filt", dict(filt="gap30", k=0.3, s=0.2))}
PER = {"dev 2011-2022": ("2011-01-01", "2022-12-31"), "hold-out 2023-01..2024-06": ("2023-01-01", "2024-06-30"), "since mid-2024": ("2024-07-01", "2026-07-02"),
       "2024 H2": ("2024-07-01", "2024-12-31"), "2025 H1": ("2025-01-01", "2025-06-30"), "2025 H2": ("2025-07-01", "2025-12-31"), "2026 H1 (to 07-02)": ("2026-01-01", "2026-07-02")}


def stats(d: pd.DataFrame) -> dict:
    t = d[d.ok]
    n = len(t)
    if n < 5:
        return dict(sessions=n)
    r = t.r; u = t.usd
    cum = r.cumsum()
    return dict(sessions=n, traded_days=int((t.usd != 0).sum()), usd_total=u.sum(), usd_mean=u.mean(), R_total=r.sum(), R_mean=r.mean(), R_sharpe=r.mean() / r.std() * np.sqrt(252), R_t=r.mean() / r.std() * np.sqrt(n),
                hit_rate=(t.usd[t.usd != 0] > 0).mean(), R_maxdd=float((cum - cum.cummax()).min()), best_day_share=float(r.max() / r[r > 0].sum()) if (r > 0).any() else np.nan)


def main() -> None:
    costs = costs_from_config(load_config())
    net, gross = {}, {}
    zero = st.Costs(commission_rt=0.0, slippage_ticks_rt=0.0)
    for n, (sl, p) in SL.items():
        s = make_sleeve("nq", sl, p, costs=costs, name=n); g = make_sleeve("nq", sl, p, costs=zero, name=n)
        net[n] = pd.DataFrame(dict(usd=s.pnl, ok=s.ok, atr=s.atr), index=pd.DatetimeIndex(s.dates))
        gross[n] = pd.Series(g.pnl, index=pd.DatetimeIndex(g.dates))
        net[n]["r"] = np.where(net[n].atr > 0, net[n].usd / net[n].atr, 0.0)
        net[n]["gross_usd"] = gross[n].values
    rows = []
    for n, d in net.items():
        for per, (a, b) in PER.items():
            x = d.loc[a:b]
            r = stats(x); r.update(sleeve=n, period=per, gross_usd_total=x[x.ok].gross_usd.sum()); rows.append(r)
    out = pd.DataFrame(rows).set_index(["period", "sleeve"])
    out.to_csv("results/sleeves_since_2024.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["sessions", "traded_days", "usd_total", "gross_usd_total", "R_total", "R_mean", "R_sharpe", "R_t", "hit_rate", "R_maxdd"]
    for per in PER:
        print("\n==", per); print(out.loc[per][cols].round(3).to_string())
    x = pd.DataFrame({n: d.loc["2024-07-01":"2026-07-02"].query("ok").r for n, d in net.items()})
    print("\ndaily R correlation since mid-2024:"); print(x.corr().round(2).to_string())
    print("\nequal-weight 50/50 noise+gap R since mid-2024: total %.2f, mean/day %.4f, t %.2f" % ((0.5 * x.nq_noise + 0.5 * x.nq_gap30).sum(), (0.5 * x.nq_noise + 0.5 * x.nq_gap30).mean(), (0.5 * x.nq_noise + 0.5 * x.nq_gap30).mean() / (0.5 * x.nq_noise + 0.5 * x.nq_gap30).std() * np.sqrt(len(x))))


if __name__ == "__main__":
    main()
