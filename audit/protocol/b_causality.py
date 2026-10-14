"""Audit B: does anything computed for sessions <= 2022-12-31 change when 2023+ data is removed? (causality / truncation test)"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions, Sessions

DEV_END = np.datetime64("2022-12-31")


def truncate(s: Sessions, n: int) -> Sessions:
    b = s.offsets[n]
    return Sessions(s.market, s.dates[:n], s.offsets[:n + 1], s.o[:b], s.h[:b], s.l[:b], s.c[:b], s.v[:b], s.smin[:b], s.roll[:n], s.mult, s.tick)


def cmp(a, b, name):
    a = np.asarray(a, float); b = np.asarray(b, float)
    same = np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99))
    print(f"  {name:<28} identical on dev: {same}  max|diff|={np.nanmax(np.abs(a - b)) if a.size else 0:.3g}")
    return same


for m in ("nq", "es", "gc"):
    t0 = time.time()
    full = build_sessions(m)
    n_dev = int(np.searchsorted(full.dates, DEV_END, side="right"))
    cut = truncate(full, n_dev)
    print(m, "sessions", len(full), "dev", n_dev, "last dev date", full.dates[n_dev - 1], "first hold date", full.dates[n_dev])
    cf, cc = ac.context(full), ac.context(cut)
    for k in ("prev_close", "ranges", "volratio", "trend20", "trend60", "trend250"):
        cmp(cf[k][:n_dev], cc[k], "ctx." + k)
    cmp(cf["sigma"][:n_dev], cc["sigma"], "ctx.sigma")
    tests = [("noise_area", {}, {}), ("vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN), {}),
             ("noise_area", {}, dict(trend=60)), ("vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN), dict(trend=250)),
             ("noise_area", {}, dict(vol="hi")), ("intraday_mom", dict(use_r12=True, thr=0.0), {})]
    if m != "nq":
        tests = tests[:4]
    for name, p, g in tests:
        a = ac.run_sleeve(full, name, ctx=cf, **p, **g)
        b = ac.run_sleeve(cut, name, ctx=cc, **p, **g)
        for f in ("pnl", "minpnl", "atr", "traded_ok", "trades"):
            cmp(getattr(a, f)[:n_dev], getattr(b, f), f"{name} {g} {f}")
    print(f"  ({time.time() - t0:.0f}s)")
