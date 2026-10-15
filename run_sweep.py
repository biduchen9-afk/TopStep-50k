"""Strategy sweep: every variant of every family on ES / NQ / GC, with IS/OOS, rank stability and
walk-forward selection. P&L is ATR-normalised (constant risk through time) so 2010 and 2026 weigh equally.

Usage: python run_sweep.py        -> results/sweep_variants.csv, results/sweep_walkforward.csv
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

import intraday_strategies as st
from market_data import build_sessions

CUT = np.datetime64("2020-01-01")
GRIDS = {
    "vwap_atr": [dict(k=k, s=s, start_min=sm) for k in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
                 for s in (0.1, 0.2, 0.3, 0.5) for sm in (60, st.CASH_OPEN)],
    "orb": [dict(or_min=o, buf=b, s=s) for o in (15, 30, 60) for b in (0.0, 0.05, 0.1)
            for s in (0.1, 0.2, 0.3, 0.5)],
    "gap": [dict(mode=m, thr=t, s=s) for m in (1, -1) for t in (0.0, 0.1, 0.2, 0.4) for s in (0.3, 0.6, 2.0)],
}


def r_series(d: st.DailySeries) -> np.ndarray:
    return np.where(d.traded_ok, d.pnl / (d.atr * d.mult), 0.0)


def sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std(ddof=1) * math.sqrt(252)) if x.size > 20 and x.std() > 0 else 0.0


def label(p: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in p.items())


def main() -> None:
    rows, store = [], {}
    for m in ("es", "nq", "gc"):
        sess = build_sessions(m)
        yrs = pd.DatetimeIndex(sess.dates).year.to_numpy()
        for fam, grid in GRIDS.items():
            for p in grid:
                d = st.run(sess, fam, **p)
                r = r_series(d)
                ok = d.traded_ok
                is_, oos = ok & (sess.dates < CUT), ok & (sess.dates >= CUT)
                store[(m, fam, label(p))] = (r, ok, yrs)
                rows.append(dict(market=m, family=fam, params=label(p),
                                 overnight_entry=bool(fam == "vwap_atr" and p["start_min"] < st.CASH_OPEN),
                                 sharpe_full=sharpe(r[ok]), sharpe_is=sharpe(r[is_]), sharpe_oos=sharpe(r[oos]),
                                 t_full=sharpe(r[ok]) * math.sqrt(ok.sum() / 252),
                                 trades_per_day=float(d.trades[ok].mean()), win_days=float((r[ok & (d.trades > 0)] > 0).mean()),
                                 sharpe_usd=d.sharpe()))
        # no-edge control: 20 coin-flip seeds
        ctl = []
        for sd in range(20):
            d = st.run(sess, "random", seed=sd)
            ctl.append(sharpe(r_series(d)[d.traded_ok]))
        rows.append(dict(market=m, family="random", params="20 seeds mean", overnight_entry=False,
                         sharpe_full=float(np.mean(ctl)), sharpe_is=np.nan, sharpe_oos=np.nan,
                         t_full=np.nan, trades_per_day=1.0, win_days=np.nan, sharpe_usd=np.nan))
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/sweep_variants.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)

    v = df[df.family != "random"]
    print(f"{len(v)} variants ({v.groupby('market').size().to_dict()}), ATR-normalised daily P&L, 2010-06 to 2026-07\n")
    print("=== 1) Does the FAMILY work at all? distribution of Sharpe across all its variants ===")
    g = v.groupby(["market", "family"])
    summ = pd.DataFrame({"n": g.size(), "median": g.sharpe_full.median(), "best": g.sharpe_full.max(),
                         "worst": g.sharpe_full.min(), "%>0": g.sharpe_full.apply(lambda x: (x > 0).mean() * 100),
                         "%>0.5": g.sharpe_full.apply(lambda x: (x > 0.5).mean() * 100),
                         "median_IS": g.sharpe_is.median(), "median_OOS": g.sharpe_oos.median(),
                         "IS-OOS rank corr": g.apply(lambda x: x.sharpe_is.rank().corr(x.sharpe_oos.rank()))})
    print(summ.round(2).to_string())
    ctl = df[df.family == "random"].set_index("market").sharpe_full
    print("\nNo-edge control (coin-flip direction, 20 seeds), Sharpe:", ctl.round(2).to_dict())

    print("\n=== 2) Top 12 variants by full-sample Sharpe ===")
    cols = ["market", "family", "params", "sharpe_full", "t_full", "sharpe_is", "sharpe_oos", "trades_per_day", "overnight_entry"]
    print(v.sort_values("sharpe_full", ascending=False).head(12)[cols].round(2).to_string(index=False))

    print("\n=== 3) Luck check: best Sharpe expected from pure noise among N variants (sd of the variants' own Sharpes x E[max of N]) ===")
    for (m, fam), x in v.groupby(["market", "family"]):
        n = len(x)
        emax = (1 - 0.5772) * _ninv(1 - 1 / n) + 0.5772 * _ninv(1 - 1 / (n * math.e))
        print(f"  {m} {fam:<9} N={n:<3} observed best {x.sharpe_full.max():.2f} | noise-expected best ~ {emax * x.sharpe_full.std():.2f} + mean {x.sharpe_full.mean():.2f}")

    print("\n=== 4) Walk-forward: each year pick the best variant of the family on the PRIOR 5 YEARS, trade next year ===")
    wf_rows = []
    for (m, fam), _ in v.groupby(["market", "family"]):
        keys = [k for k in store if k[0] == m and k[1] == fam]
        yrs = store[keys[0]][2]
        test_r, base = [], []
        per_year = {}
        for Y in range(2015, 2027):
            tr = (yrs >= Y - 5) & (yrs <= Y - 1)
            te = yrs == Y
            if te.sum() < 50:
                continue
            best = max(keys, key=lambda k: sharpe(store[k][0][tr & store[k][1]]))
            r, ok, _y = store[best]
            seg = r[te & ok]
            test_r.append(seg)
            per_year[Y] = sharpe(seg)
            base.append([store[k][0][te & store[k][1]] for k in keys])
        wf = np.concatenate(test_r)
        nv = len(keys)
        pooled = [np.concatenate([b[i] for b in base]) for i in range(nv)]
        med = float(np.median([sharpe(p) for p in pooled]))
        wf_rows.append(dict(market=m, family=fam, wf_sharpe=sharpe(wf), wf_t=sharpe(wf) * math.sqrt(len(wf) / 252),
                            median_variant_same_years=med, years_positive=f"{sum(s > 0 for s in per_year.values())}/{len(per_year)}"))
    wf = pd.DataFrame(wf_rows)
    wf.to_csv("results/sweep_walkforward.csv", index=False)
    print(wf.round(2).to_string(index=False))

    print("\n=== 5) Cost stress on each family's single best static variant (slippage ticks per round trip) ===")
    for (m, fam), x in v.groupby(["market", "family"]):
        best = x.sort_values("sharpe_full", ascending=False).iloc[0]
        p = dict(kv.split("=") for kv in best.params.split())
        p = {k: (float(val) if "." in val else int(val)) for k, val in p.items()}
        sess = build_sessions(m)
        out = []
        for slip in (0.0, 1.0, 2.0, 3.0):
            d = st.run(sess, fam, costs=st.Costs(slippage_ticks_rt=slip), **p)
            out.append(f"{slip:.0f}t: {sharpe(r_series(d)[d.traded_ok]):.2f}")
        print(f"  {m} {fam:<9} [{best.params}]  " + "  ".join(out))


def _ninv(p: float) -> float:
    """Inverse normal CDF (Acklam-style via erfinv bisection; accurate enough for a luck benchmark)."""
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    main()
