"""Download one-minute bars from the London Strategic Edge vault to a pickle for the forward test.
   LSE_API_KEY=... python -m paper.lse_fetch --symbol <SYMBOL> --start 2026-06-01 [--end 2026-10-15] [--dataset <name>] --out data_new/nq_lse_1m.pkl
Prints a data-quality report (bars per session, RTH completeness, volume present, price level against the stored Databento data where they overlap)."""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from paper.feed import Bar, in_session, lse_fetch


def quality(bars):
    df = pd.DataFrame([dict(ts=b.ts, o=b.o, h=b.h, l=b.l, c=b.c, v=b.v, smin=b.smin, label=b.label, ins=in_session(b)) for b in bars])
    out = {"bars": len(df), "first": str(df.ts.min()), "last": str(df.ts.max()), "duplicates": int(df.ts.duplicated().sum()), "zero_volume_share": float((df.v <= 0).mean())}
    d = df[df.ins]
    g = d.groupby("label")
    rth = d[(d.smin >= 930) & (d.smin < 1315)].groupby("label").size()
    out.update(sessions=int(g.ngroups), rth_bars_median=float(rth.median()) if len(rth) else None, sessions_rth_under_370=int((rth < 370).sum()), sessions_no_rth=int(g.ngroups - len(rth)))
    return out, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="data_new/nq_lse_1m.pkl")
    a = ap.parse_args()
    bars = lse_fetch(a.symbol, a.start, a.end, "1m", dataset=a.dataset)
    if not bars:
        raise SystemExit("no bars returned: check the symbol with python -m paper.lse_probe")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(bars, f)
    q, df = quality(bars)
    print("saved", a.out, q)
    # overlap with the stored Databento sessions: compare closes minute by minute
    try:
        from market_data import load_minute
        old = load_minute("nq")
        j = df.set_index("ts").c.rename("lse").to_frame().join(old.c.rename("db"), how="inner")
        if len(j):
            diff = (j.lse - j.db)
            print(f"overlap with Databento: {len(j)} common minutes; close difference median {diff.median():.2f} pts, p5 {diff.quantile(.05):.2f}, p95 {diff.quantile(.95):.2f} "
                  f"(a constant offset means a different roll / adjustment; a noisy difference means a different contract or feed)")
        else:
            print("no minutes in common with the stored Databento data (their data ends 2026-07-02); start the fetch earlier (e.g. 2026-06-01) to compare price levels")
    except Exception as e:
        print("overlap check skipped:", e)


if __name__ == "__main__":
    main()
