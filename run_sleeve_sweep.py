"""Academic sleeves x gates x ES/NQ/GC, ATR-normalised, dev (<=2022) vs hold-out (2023+). Saves daily series for portfolios."""
from __future__ import annotations

import math
import pickle
from itertools import product

import numpy as np
import pandas as pd

import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions

DEV_END = np.datetime64("2022-12-31")
GRIDS = {
    "intraday_mom": [dict(use_r12=r, thr=t) for r in (True, False) for t in (0.0, 0.1, 0.25)],
    "noise_area": [dict()],
    "vwap_atr": [dict(k=k, s=s, start_min=st.CASH_OPEN) for k in (0.2, 0.3) for s in (0.2, 0.5)],
    "orb": [dict(or_min=o, buf=0.05, s=0.2) for o in (15, 30, 60)],
    "vwap_fade": [dict(k=k, s=s) for k in (0.3, 0.5) for s in (0.3, 0.6)],
    "gap": [dict(mode=m, thr=0.2, s=0.6) for m in (1, -1)],
}
GATES = [dict(), dict(trend=60), dict(trend=250), dict(vol="hi"), dict(vol="lo")]


def sharpe(x):
    return float(x.mean() / x.std(ddof=1) * math.sqrt(252)) if x.size > 20 and x.std() > 0 else 0.0


def label(sleeve, p, g):
    return f"{sleeve}|" + ",".join(f"{k}={v}" for k, v in p.items()) + "|" + (",".join(f"{k}={v}" for k, v in g.items()) or "nogate")


def main() -> None:
    rows, series = [], {}
    for m in ("es", "nq", "gc"):
        sess = build_sessions(m)
        ctx = ac.context(sess)
        dev = sess.dates <= DEV_END
        for sleeve, grid in GRIDS.items():
            for p, g in product(grid, GATES):
                d = ac.run_sleeve(sess, sleeve, ctx=ctx, **p, **g)
                r = np.where(d.traded_ok, d.pnl / (d.atr * d.mult), 0.0)
                ok = d.traded_ok
                series[(m, label(sleeve, p, g))] = (r, d.minpnl / (d.atr * d.mult), ok)
                rows.append(dict(market=m, sleeve=sleeve, params=",".join(f"{k}={v}" for k, v in p.items()),
                                 gate=",".join(f"{k}={v}" for k, v in g.items()) or "none",
                                 sharpe_full=sharpe(r[ok]), sharpe_dev=sharpe(r[ok & dev]), sharpe_hold=sharpe(r[ok & ~dev]),
                                 t_full=sharpe(r[ok]) * math.sqrt(ok.sum() / 252), active_days=int((d.trades[ok] > 0).sum()),
                                 trades_per_day=float(d.trades[ok].mean())))
        print(m, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/sleeve_sweep.csv", index=False)
    pickle.dump({"series": series, "dates": {m: build_sessions(m).dates for m in ("es", "nq", "gc")}}, open("/home/user/data_raw/sleeve_series.pkl", "wb"))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
    print(f"{len(df)} variants\n")
    print("=== Paper-faithful, ungated: does the published effect show up in futures? (ATR-normalised Sharpe) ===")
    pf = df[(df.gate == "none") & (((df.sleeve == "intraday_mom") & (df.params == "use_r12=True,thr=0.0")) | (df.sleeve == "noise_area") |
            ((df.sleeve == "orb") & (df.params == "or_min=30,buf=0.05,s=0.2")))]
    print(pf[["market", "sleeve", "params", "sharpe_full", "t_full", "sharpe_dev", "sharpe_hold", "trades_per_day"]].round(2).to_string(index=False))
    print("\n=== Median Sharpe by market x sleeve (all params/gates) ===")
    print(df.pivot_table(index="sleeve", columns="market", values="sharpe_full", aggfunc="median").round(2).to_string())
    print("\n=== Effect of the gates: mean Sharpe by gate ===")
    print(df.pivot_table(index="gate", columns="market", values="sharpe_full", aggfunc="mean").round(2).to_string())
    print("\n=== Top 12 by DEV Sharpe, with their hold-out Sharpe ===")
    print(df.sort_values("sharpe_dev", ascending=False).head(12)[["market", "sleeve", "params", "gate", "sharpe_dev", "sharpe_hold", "t_full", "trades_per_day"]].round(2).to_string(index=False))
    print("\n=== Best DEV variant per market x sleeve and its hold-out ===")
    best = df.sort_values("sharpe_dev", ascending=False).groupby(["market", "sleeve"]).head(1).sort_values(["market", "sleeve"])
    print(best[["market", "sleeve", "params", "gate", "sharpe_dev", "sharpe_hold"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
