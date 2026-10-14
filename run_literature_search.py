"""Pre-registration 7: the 44 pre-declared ES / GC trials from the SSRN / arXiv search, regular trading hours only. See docs/PREREGISTRATION_7.md."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import intraday_strategies as st
from lit_sleeves import M
from portfolio import DEV_END, basket, make_sleeve, r_series, sharpe
from run_corrected import NEW

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 70)


def trials():
    T = []
    for m in ("es", "gc"):
        for sg, fl in itertools.product(("gao", "rod"), ("none", "hivol")):
            T.append((m, "L1 last-half-hour momentum", "lt_mom", dict(signal=sg, filter=fl, window="close")))
    for sg, fl in itertools.product(("gao", "rod"), ("none", "hivol")):
        T.append(("gc", "L1 last-half-hour momentum", "lt_mom", dict(signal=sg, filter=fl, window="comex")))
    for m in ("es", "gc"):
        for k, ex in itertools.product((0.3, 0.6), ("noon", "close")):
            T.append((m, "L2 opening-move reversal", "lt_reversal", dict(k=k, exit=ex)))
    T.append(("es", "L3 lunch drift", "lt_lunch", dict(cond="all"))); T.append(("es", "L3 lunch drift", "lt_lunch", dict(cond="down")))
    for m in ("es", "gc"):
        for tg in ("10R", "none"):
            T.append((m, "L4 5-minute candle breakout", "lt_candle5", dict(target=tg)))
        for om in (15, 30):
            T.append((m, "L5 volatility-gated ORB", "lt_orbvol", dict(or_min=om)))
        for s0, s1 in ((M(10, 0), M(10, 30)), (M(14, 0), M(14, 30)), (M(15, 0), M(15, 30))):
            T.append((m, "L6 same-slot continuation", "lt_slot", dict(s0=s0, s1=s1)))
        for N in (4, 7):
            T.append((m, "L8 narrow-range breakout", "lt_nr", dict(N=N)))
    for src, sg in itertools.product(("es", "nq"), ("same", "opposite")):
        T.append(("gc", "L7 cross-asset ES/NQ to GC", "lt_xasset", dict(src=src, sign=sg)))
    return T


def main() -> None:
    T = trials(); assert len(T) == 44, len(T)
    rows = []
    for k, (m, idea, sl, params) in enumerate(T):
        st.TRACE = []
        sle = make_sleeve(m, sl, params, None, costs=NEW, name=f"{m}_{sl}_{k}")
        tr = np.array(st.TRACE) if st.TRACE else np.zeros((0, 2)); st.TRACE = None
        b = basket([sle]); r = r_series(b); ok = b.ok; dev = b.dates <= DEV_END
        y = pd.Series(r, index=pd.DatetimeIndex(b.dates))[ok & dev]
        yr = y.groupby(y.index.year).apply(lambda g: sharpe(g.to_numpy()) if len(g) > 20 else np.nan).dropna()
        rth = bool(len(tr) == 0 or (tr[:, 0].min() >= st.CASH_OPEN and tr[:, 1].max() <= 1315))
        rows.append(dict(trial=k, market=m, idea=idea, params=str(params), full=sharpe(r[ok]), dev=sharpe(r[ok & dev]), hold=sharpe(r[ok & ~dev]), pos_years=float((yr > 0).mean()), n_trades=len(tr), rth_only=rth))
        if k % 11 == 10:
            print(k + 1, "of 44", flush=True)
    t = pd.DataFrame(rows); t.to_csv("results/literature_trials.csv", index=False)
    assert t.rth_only.all(), "a sleeve traded outside regular hours"
    best = t.sort_values("dev", ascending=False).groupby(["market", "idea"]).head(1).sort_values("dev", ascending=False)
    best["accepted"] = (best.dev >= 0.5) & (best.pos_years >= 0.6) & (best.hold > 0); best["weak"] = best.accepted & (best.dev < 0.7)
    best.to_csv("results/literature_best.csv", index=False)
    print("\nAll 44 trials traded only 09:30-15:55 ET. Best-dev configuration per (market, idea):")
    print(best[["market", "idea", "params", "full", "dev", "hold", "pos_years", "n_trades", "accepted", "weak"]].round(2).to_string(index=False))
    print("\nbest dev Sharpe of all 44:", round(t.dev.max(), 2), "| median:", round(t.dev.median(), 2), "| dev >= 0.5:", int((t.dev >= 0.5).sum()), "| hold-out Sharpe > 0:", int((t.hold > 0).sum()))


if __name__ == "__main__":
    main()
