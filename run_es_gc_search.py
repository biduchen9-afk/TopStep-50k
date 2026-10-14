"""Pre-registration 6, Part A: the 60 pre-declared ES / GC trials. See docs/PREREGISTRATION_6.md."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import intraday_strategies as st
from portfolio import DEV_END, basket, make_sleeve, r_series, sharpe
from run_corrected import NEW
from xs_sleeves import COMEX_OPEN

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 70)
C = st.CASH_OPEN


def trials():
    T = []
    for m in ("es", "gc"):
        for tr in (None, 60, 250):
            T.append((m, "P1 noise-area", "noise_area", {}, {} if tr is None else dict(trend=tr)))
        for k, s in itertools.product((0.2, 0.3, 0.5), (0.2, 0.4)):
            T.append((m, "P2 VWAP+ATR", "vwap_atr", dict(k=k, s=s, start_min=C), {}))
        for k, s in itertools.product((0.2, 0.3, 0.5), (0.2, 0.4)):
            T.append((m, "P3 gap30 VWAP", "vv_filt", dict(filt="gap30", k=k, s=s), {}))
        for buf, s in itertools.product((0.0, 0.1), (0.3, 0.6)):
            T.append((m, "N1 prior-day breakout", "xs_prevday", dict(buf=buf, s=s), {}))
        anchor = C if m == "es" else COMEX_OPEN
        for buf, s in itertools.product((0.0, 0.1), (0.4, 0.8)):
            T.append((m, "N2 overnight-range breakout", "xs_onrange", dict(buf=buf, s=s, anchor=anchor), {}))
        for thr, s in itertools.product((0.5, 1.0), (0.5, 1.0)):
            T.append((m, "N4 gap fade", "gap", dict(thr=thr, s=s, mode=-1), {}))
    for om, s in itertools.product((15, 30, 60), (0.5, 1.0)):
        T.append(("gc", "N3 COMEX-open ORB", "xs_orb", dict(or_min=om, s=s, anchor=COMEX_OPEN), {}))
    return T


def main() -> None:
    T = trials(); assert len(T) == 60, len(T)
    rows, ser = [], {}
    for k, (m, idea, sl, params, gate) in enumerate(T):
        sle = make_sleeve(m, sl, params, gate, costs=NEW, name=f"{m}_{sl}_{k}")
        b = basket([sle]); r = r_series(b); ok = b.ok; dev = b.dates <= DEV_END
        y = pd.Series(r, index=pd.DatetimeIndex(b.dates))[ok & dev]
        yr = y.groupby(y.index.year).apply(lambda g: sharpe(g.to_numpy()) if len(g) > 20 else np.nan).dropna()
        rows.append(dict(trial=k, market=m, idea=idea, params=str({**params, **gate}), full=sharpe(r[ok]), dev=sharpe(r[ok & dev]), hold=sharpe(r[ok & ~dev]), pos_years=float((yr > 0).mean()), n_days=int((sle.pnl != 0).sum())))
        ser[k] = (b.dates, r)
        if k % 10 == 9:
            print(k + 1, "of 60 done", flush=True)
    t = pd.DataFrame(rows); t.to_csv("results/es_gc_trials.csv", index=False)
    best = t.sort_values("dev", ascending=False).groupby(["market", "idea"]).head(1).sort_values("dev", ascending=False)
    best["accepted"] = (best.dev >= 0.5) & (best.pos_years >= 0.6) & (best.hold > 0)
    best["weak"] = best.accepted & (best.dev < 0.7)
    print(f"\n60 trials logged. Best-dev configuration per (market, idea); acceptance = dev>=0.5 & >=60% positive dev years & hold>0:")
    print(best[["market", "idea", "params", "full", "dev", "hold", "pos_years", "accepted", "weak"]].round(2).to_string(index=False))
    best.to_csv("results/es_gc_best.csv", index=False)
    print("\nbest dev Sharpe among all 60:", round(t.dev.max(), 2), "| median dev Sharpe:", round(t.dev.median(), 2), "| trials with dev >= 0.5:", int((t.dev >= 0.5).sum()))


if __name__ == "__main__":
    main()
