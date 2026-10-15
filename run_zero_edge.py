"""Does a zero-edge stream with aggressive sizing pass the Combine more often than the real strategies? (descriptive; nothing here selects or changes anything)
Zero-edge stream: every session a fair coin picks long or short at the 09:30 ET open, flat at 15:55 ET, real intraday paths, the modelled cost ($1.72 per micro round trip), ATR-based sizing exactly like the sleeves.
Same Topstep 150K Combine and XFA engine and the same baseline cells; Combine risk R is scaled 1x / 2x / 4x of 9,600 to vary the aggression (the 150-micro cap still applies).
Run with TOPSTEP_RAW_DIR=/home/user/data_raw_live."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import numpy as np
import pandas as pd

import intraday_strategies as st
import market_data as md
import retest as R
from real_engine import Stream

SEEDS = int(os.environ.get("SEEDS", "12"))
MULTS = (1.0, 2.0, 4.0)
COST_PTS = 0.86                    # $1.22 + one 0.25 tick = $1.72 per micro round trip at $2 per point (same as costs_from_config)
POINT = 2.0


def day_paths(ref: Stream):
    sess = md.build_sessions("nq")
    pos = pd.Index(sess.dates).get_indexer(ref.dates)
    n = len(ref.dates)
    fin, lo, hi, ok = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, bool)
    for k in range(n):
        if pos[k] < 0 or not ref.ok[k]:
            continue
        o, h, l, c, v, m = sess.day(pos[k])
        sel = np.flatnonzero((m >= st.CASH_OPEN) & (m < st.FLAT_BY))
        if len(sel) < 300:
            continue
        a, b = sel[0], sel[-1]
        e = o[a]
        fin[k], lo[k], hi[k], ok[k] = c[b] - e, l[a:b + 1].min() - e, h[a:b + 1].max() - e, True
    return fin, lo, hi, ok


def coin_stream(ref: Stream, paths, seed: int) -> Stream:
    fin, lo, hi, ok = paths
    sgn = np.random.default_rng(seed).choice([-1.0, 1.0], size=len(fin))
    pnl_pts = sgn * fin - COST_PTS
    low_pts = np.where(sgn > 0, lo, -hi)
    low_pts = np.minimum(low_pts, pnl_pts)
    pnl, low = np.where(ok, pnl_pts * POINT, 0.0), np.where(ok, np.minimum(low_pts * POINT, 0.0), 0.0)
    return Stream(f"coin{seed}", ref.dates, pnl, low, ref.atr_usd, ok & ref.ok)


def combine_only(s: Stream, n: int, Rv: float, starts) -> dict:
    ev = R.AcctEval(s, n, Rv, starts=list(starts))
    st_ = np.array([e[1] for e in ev.ev]); tau = np.array([e[2] for e in ev.ev])
    return dict(starts=len(st_), pass_=float((st_ == 1).mean()), breach=float((st_ == 0).mean()), tau=float(np.median(tau)))


def full(s: Stream, n: int, Rv: float, starts) -> dict:
    ev = R.AcctEval(s, n, Rv, starts=list(starts))
    df = ev.run(R.BASE["a1"], R.BASE["t1"], R.BASE["a2"]); m = R.summ(df)
    return dict(pass_=m["pass_rate"], paid=m["paid_share"], ev_start=m["ev"], rate=m["rate"], fees_start=m["fees_per_start"])


def main() -> None:
    D = R.load(); ref = D["ref"]; n, n_dev = D["n"], D["n_dev"]
    ix = lambda s: int(np.searchsorted(ref.dates, np.datetime64(s)))
    first = int(np.flatnonzero(ref.ok)[0]) + 260
    P = {"dev 2011-2022": ("dev", range(first, n_dev - 400, 3), n_dev), "hold-out starts 2023-01..2024-12": ("full", range(ix("2023-01-02"), n - 400, 3), n),
         "starts since 2024-07 (Combine only)": ("comb", range(ix("2024-07-01"), n - 130, 3), n), "starts since 2024-12 (Combine only)": ("comb", range(ix("2024-12-02"), n - 130, 3), n)}
    paths = day_paths(ref)
    real = {"real noise-area": D["ST"]["noise"], "real gap-VWAP": D["ST"]["gap"]}
    rows = []
    for per, (kind, starts, nn) in P.items():
        for mult in MULTS:
            Rv = 9600.0 * mult
            for name, s in real.items():
                r = (full if kind != "comb" else combine_only)(s, nn, Rv, starts); rows.append(dict(period=per, stream=name, R_mult=mult, **r))
            acc = []
            for sd in range(SEEDS):
                acc.append((full if kind != "comb" else combine_only)(coin_stream(ref, paths, 1000 + sd), nn, Rv, starts))
            rows.append(dict(period=per, stream="ZERO-EDGE coin flip (mean of %d seeds)" % SEEDS, R_mult=mult, **{k: float(np.mean([a[k] for a in acc])) for k in acc[0]},
                             **({"pass_sd": float(np.std([a["pass_"] for a in acc]))})))
            print(per, mult, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv("results/zero_edge.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for per in P:
        print("\n==", per); print(df[df.period == per].drop(columns="period").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
