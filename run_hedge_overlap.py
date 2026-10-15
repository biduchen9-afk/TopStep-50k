"""Rules check (Topstep 'cross-account hedging': opposite positions across the user's accounts at the same time are prohibited, even briefly):
how often do the noise-area and gap-VWAP sleeves hold OPPOSITE positions at the same time? Bar-by-bar sleeves (parity-tested against the back-test), repaired data."""
from __future__ import annotations

import os
os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import sys
import numpy as np, pandas as pd
import market_data as md
from paper.history import History
from paper.replay import ctx_for
from paper.sleeves import make_sleeves

sess = md.build_sessions("nq"); n = len(sess)
hist = History(); rows = []
for i in range(n):
    o, h, l, c, v, m = sess.day(i)
    ctx = ctx_for(hist, bool(sess.roll[i]))
    sl = {k: s for k, s in make_sleeves(0.0).items() if k in ("nq_noise", "nq_gap30")}
    for s in sl.values():
        s.reset(ctx)
    a, b = sl["nq_noise"], sl["nq_gap30"]
    opp = same = 0; na = nb = 0
    for j in range(len(c)):
        a.on_bar(j, int(m[j]), o[j], h[j], l[j], c[j], v[j]); b.on_bar(j, int(m[j]), o[j], h[j], l[j], c[j], v[j])
        na += a.pos != 0; nb += b.pos != 0
        if a.pos and b.pos:
            if a.pos * b.pos < 0: opp += 1
            else: same += 1
    a.finish(); b.finish()
    rows.append(dict(date=sess.dates[i], ok=ctx["ok"], noise_min=na, gap_min=nb, opp_min=opp, same_min=same, noise_trades=a.trades, gap_trades=b.trades))
    hist.add_bars(o, h, l, c, m, bool(sess.roll[i]))
    if i % 500 == 0:
        print(i, n, flush=True)
D = pd.DataFrame(rows); D.to_csv("results/hedge_overlap.csv", index=False)
d = D[D.ok]
both = d[(d.noise_trades > 0) & (d.gap_trades > 0)]
print(f"tradable sessions {len(d)}; noise-area trades on {int((d.noise_trades > 0).sum())}, gap-VWAP on {int((d.gap_trades > 0).sum())}, both on {len(both)}")
print(f"sessions with at least one minute of OPPOSITE positions: {int((d.opp_min > 0).sum())} = {(d.opp_min > 0).mean():.1%} of tradable sessions, {(both.opp_min > 0).mean():.1%} of sessions where both trade")
print(f"minutes with both in a position: opposite {int(d.opp_min.sum())}, same direction {int(d.same_min.sum())}; opposite share {d.opp_min.sum() / max(d.opp_min.sum() + d.same_min.sum(), 1):.1%}")
print("per year (opposite sessions / tradable):", d.assign(y=pd.DatetimeIndex(d.date).year).groupby("y").apply(lambda x: f"{int((x.opp_min > 0).sum())}/{len(x)}").to_dict())
