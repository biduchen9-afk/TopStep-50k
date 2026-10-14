"""VWAP-combination sleeves on NQ / ES / GC with realistic micro costs. Selection is by DEV (2010-2022) Sharpe only; every trial is logged.
Output: results/vwap_variant_trials.csv (all trials, so the number of looks is visible) and results/vwap_variant_yearly.csv."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import intraday_strategies as st
from portfolio import basket, make_sleeve, r_series, sharpe, DEV_END
from run_corrected import NEW

pd.set_option("display.width", 250, "display.max_columns", 40)
GRID = []
for f, k, s in itertools.product(("gap", "gap30", "first30", "volhi"), (0.2, 0.3, 0.5), (0.2, 0.4)):
    GRID.append(("vv_filt", dict(filt=f, k=k, s=s)))
for a, k, s in itertools.product(("c", "cw", "cd", "cdw"), (0.2, 0.3, 0.5), (0.2, 0.4)):
    GRID.append(("vv_multi", dict(anchors=a, k=k, s=s)))
for kf in (0.4, 0.6, 0.8):
    GRID.append(("vv_blend", dict(k_f=kf, s_f=0.4)))
GRID.append(("vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN)))        # the existing sleeve, for reference


def main() -> None:
    rows, yearly = [], []
    for mkt in ("nq", "es", "gc"):
        for name, p in GRID:
            sl = make_sleeve(mkt, name, p, costs=NEW, name=f"{mkt}_{name}")
            b = basket([sl]); r = r_series(b); ok = b.ok; dev = b.dates <= DEV_END
            rows.append(dict(market=mkt, sleeve=name, params=str(p), full=sharpe(r[ok]), dev=sharpe(r[ok & dev]), hold=sharpe(r[ok & ~dev]),
                             n_trades=int((sl.pnl != 0).sum())))
            y = pd.Series(r, index=pd.DatetimeIndex(b.dates)); y = y[ok]
            for yr, g in y.groupby(y.index.year):
                yearly.append(dict(market=mkt, sleeve=name, params=str(p), year=yr, sharpe=sharpe(g.to_numpy()) if len(g) > 20 else np.nan))
        print(mkt, "done", flush=True)
    t = pd.DataFrame(rows); t.to_csv("results/vwap_variant_trials.csv", index=False)
    pd.DataFrame(yearly).to_csv("results/vwap_variant_yearly.csv", index=False)
    print(f"\n{len(t)} trials logged.  Best by DEV Sharpe per market x sleeve family (hold-out shown, not used to pick):")
    best = t.sort_values("dev", ascending=False).groupby(["market", "sleeve"]).head(1).sort_values(["market", "dev"], ascending=[True, False])
    print(best.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
