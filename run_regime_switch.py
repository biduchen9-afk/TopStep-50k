"""Can a causal regime switcher raise the Combine pass rate? (descriptive; all variants declared before running and ALL reported; nothing selected)
Sleeves: nq_noise, nq_vwap, nq_gap30 (R = net P&L / ATR per micro). Each session the switcher looks only at the PAST L sessions (L = 40, 80, 160) and decides:
  V1  trade the sleeve with the highest trailing mean R;
  V2  as V1, but stay flat when that best trailing mean R <= 0 (a regime on/off switch);
  V3  as V1, with size x1.5 when the best trailing mean R > 0 and x0.5 otherwise (size follows the regime).
Compared with always trading each single sleeve and with a zero-edge coin flip, same Combine/XFA engine and baseline cells. Run with TOPSTEP_RAW_DIR=/home/user/data_raw_live."""
from __future__ import annotations

import os

os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import numpy as np
import pandas as pd

import retest as R
import run_zero_edge as Z
from real_engine import Stream

LS = (40, 80, 160)
VARIANTS = ("V1 switch", "V2 switch + flat if <=0", "V3 switch + size x1.5/x0.5")


def build(D) -> dict:
    ref = D["ref"]
    gs = R.K.with_risk_unit(R.K.group_stream(["nq_vwap"]), "atr", 40)
    assert np.array_equal(gs.dates, ref.dates)
    S = dict(noise=D["ST"]["noise"], vwap=gs, gap=D["ST"]["gap"])
    atr = np.where(ref.ok, ref.atr_usd, np.nan)
    r = {k: np.where(s.ok & np.isfinite(s.atr_usd) & (s.atr_usd > 0), s.pnl / s.atr_usd, 0.0) for k, s in S.items()}
    lw = {k: np.where(s.ok & np.isfinite(s.atr_usd) & (s.atr_usd > 0), s.low / s.atr_usd, 0.0) for k, s in S.items()}
    return S, atr, r, lw


def switched(ref: Stream, atr, r, lw, L: int, variant: str) -> Stream:
    names = list(r); n = len(ref.dates)
    M = pd.DataFrame({k: pd.Series(r[k]).rolling(L, min_periods=L).mean().shift(1) for k in names}).to_numpy()      # trailing mean R, past sessions only
    pnl, low = np.zeros(n), np.zeros(n)
    for t in range(n):
        if not ref.ok[t] or not np.isfinite(M[t]).all():
            continue
        j = int(np.argmax(M[t])); best = M[t, j]; k = names[j]
        m = 1.0
        if variant.startswith("V2") and best <= 0:
            continue
        if variant.startswith("V3"):
            m = 1.5 if best > 0 else 0.5
        pnl[t], low[t] = m * r[k][t] * atr[t], m * lw[k][t] * atr[t]
    return Stream(f"switch{L}{variant[:2]}", ref.dates, pnl, low, ref.atr_usd, ref.ok)


def main() -> None:
    D = R.load(); ref = D["ref"]; n, n_dev = D["n"], D["n_dev"]
    ix = lambda s: int(np.searchsorted(ref.dates, np.datetime64(s)))
    first = int(np.flatnonzero(ref.ok)[0]) + 260 + 160
    P = {"dev 2011-2022": ("full", range(first, n_dev - 400, 3), n_dev), "hold-out starts 2023-01..2024-12": ("full", range(ix("2023-01-02"), n - 400, 3), n),
         "starts since 2024-12 (Combine only)": ("comb", range(ix("2024-12-02"), n - 130, 3), n)}
    S, atr, r, lw = build(D)
    streams = {"always noise-area": S["noise"], "always VWAP": S["vwap"], "always gap-VWAP": S["gap"]}
    paths = Z.day_paths(ref)
    streams["ZERO-EDGE coin flip (mean of 6 seeds)"] = None
    for L in LS:
        for v in VARIANTS:
            streams[f"L={L} {v}"] = switched(ref, atr, r, lw, L, v)
    rows = []
    for per, (kind, starts, nn) in P.items():
        for name, s in streams.items():
            f = Z.combine_only if kind == "comb" else Z.full
            if s is None:
                acc = [f(Z.coin_stream(ref, paths, 2000 + sd), nn, 9600.0, starts) for sd in range(6)]
                row = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
            else:
                row = f(s, nn, 9600.0, starts)
            rows.append(dict(period=per, stream=name, **row))
        print(per, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv("results/regime_switch.csv", index=False)
    pd.set_option("display.width", 250)
    for per in P:
        print("\n==", per); print(df[df.period == per].drop(columns="period").round(3).dropna(axis=1, how="all").to_string(index=False))


if __name__ == "__main__":
    main()
