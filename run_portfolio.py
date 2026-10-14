"""Portfolio construction + switching rules, selected on dev (<=2022), evaluated on hold-out (2023+)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import intraday_strategies as st
from portfolio import (DEV_END, basket, make_sleeve, align, r_series, sharpe, split_sharpes, trailing_sharpe_mask, volratio_mask)

pd.set_option("display.width", 250, "display.max_columns", 30)


def main() -> None:
    S = {
        "nq_noise": make_sleeve("nq", "noise_area", {}, name="nq_noise"),
        "nq_vwap": make_sleeve("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN), name="nq_vwap"),
        "nq_orb": make_sleeve("nq", "orb", dict(or_min=60, buf=0.05, s=0.2), name="nq_orb"),
        "es_noise": make_sleeve("es", "noise_area", {}, name="es_noise"),
        "es_vwap_t250": make_sleeve("es", "vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN), dict(trend=250), name="es_vwap_t250"),
        "gc_noise_t60": make_sleeve("gc", "noise_area", {}, dict(trend=60), name="gc_noise_t60"),
    }
    rows = []
    def add(label, stream):
        rows.append(dict(portfolio=label, **split_sharpes(stream)))
    for k, s in S.items():
        add(f"single: {k}", basket([s], name=k))
    sets = {
        "NQ noise + NQ vwap": ["nq_noise", "nq_vwap"],
        "NQ noise + NQ vwap + ES noise": ["nq_noise", "nq_vwap", "es_noise"],
        "NQ noise + ES noise": ["nq_noise", "es_noise"],
        "NQ noise + NQ vwap + ES noise + ES vwap(t250)": ["nq_noise", "nq_vwap", "es_noise", "es_vwap_t250"],
        "all six (incl. GC)": list(S),
        "NQ noise+vwap+orb": ["nq_noise", "nq_vwap", "nq_orb"],
    }
    for lab, names in sets.items():
        add(lab, basket([S[n] for n in names], name=lab))
    # switching rules
    base = [S[n] for n in sets["NQ noise + NQ vwap + ES noise + ES vwap(t250)"]]
    idx = align(base)
    for thr in (-9.0, 0.0, 0.5):
        m = trailing_sharpe_mask(base, 120, thr)
        add(f"switch: trailing-120d Sharpe > {thr} (4 sleeves)", basket(base, m, name=f"sw{thr}"))
    for hi_noise, thr in ((True, 1.0), (True, 0.9), (True, 1.1)):
        m = {"nq_noise": volratio_mask(S["nq_noise"], idx, True, thr), "es_noise": volratio_mask(S["es_noise"], idx, True, thr),
             "nq_vwap": ~volratio_mask(S["nq_vwap"], idx, True, thr) | volratio_mask(S["nq_vwap"], idx, True, thr),
             "es_vwap_t250": np.ones(len(idx), bool)}
        add(f"switch: noise sleeves only when 5d/60d range ratio >= {thr}", basket(base, m, name="volsw"))
    res = pd.DataFrame(rows)
    res.to_csv("results/portfolio_candidates.csv", index=False)
    print(res.round(2).to_string(index=False))
    # correlations of the sleeves (daily ATR-normalised P&L, active days)
    idx = align(list(S.values()))
    R = pd.DataFrame({k: pd.Series(np.where(s.ok, s.pnl / np.where(s.ok, s.atr, 1.0), np.nan), index=pd.DatetimeIndex(s.dates)).reindex(idx) for k, s in S.items()})
    print("\nDaily correlation of sleeves (ATR-normalised, pairwise active days):"); print(R.corr().round(2).to_string())


if __name__ == "__main__":
    main()
