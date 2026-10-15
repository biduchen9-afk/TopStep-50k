"""Combine path diagnostic (descriptive): replay each Combine start day by day with the engine's own rules and record how far it got, whether the profit target was reached but the
55% consistency rule blocked the pass, and when/why it breached. Parity-checked against run_eval_firm. Run with TOPSTEP_RAW_DIR=/home/user/data_raw_live."""
from __future__ import annotations

import os

os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import numpy as np
import pandas as pd

import retest as R
from firm_engine import _floor_after


def trace(ph, x, lo, nd, i0):
    bal = peak = best = 0.0
    floor = -ph.dd; days = 0; t = i0; max_bal = 0.0; reached = None; blocked_days = 0; n = len(x)
    while t < min(n, i0 + ph.max_days):
        k = min(nd[t], ph.max_micros); p, w = x[t] * k, lo[t] * k
        lim = floor - bal
        if ph.dd_type == "intraday" or ph.dll is not None:
            first = lim if ph.dll is None else max(lim, ph.dll)
            if ph.dd_type == "intraday" and w <= first:
                if ph.dll is not None and ph.dll > lim and not ph.dll_hard:
                    p = ph.dll - ph.dll_slip - ph.dll_slip_ticks * 0.5 * k
                else:
                    return dict(status=0, tau=t - i0 + 1, max_bal=max_bal, reached=reached, blocked_days=blocked_days, end=t, best=best, bal=bal)
            elif ph.dll is not None and w <= ph.dll:
                if ph.dll_hard:
                    return dict(status=0, tau=t - i0 + 1, max_bal=max_bal, reached=reached, blocked_days=blocked_days, end=t, best=best, bal=bal)
                p = ph.dll - ph.dll_slip - ph.dll_slip_ticks * 0.5 * k
        if ph.day_cap is not None and p > ph.day_cap:
            p = ph.day_cap
        bal += p
        if p != 0.0:
            days += 1
        best = max(best, p)
        if bal > peak:
            peak = bal; floor = _floor_after(bal, peak, ph.trail, ph.dd, ph.lock, ph.lock_level, floor)
        max_bal = max(max_bal, bal)
        t += 1
        if bal >= ph.target and days >= ph.min_days:
            if reached is None:
                reached = t - i0
            if ph.consistency is None or best <= ph.consistency * bal:
                return dict(status=1, tau=t - i0, max_bal=max_bal, reached=reached, blocked_days=blocked_days, end=t - 1, best=best, bal=bal)
            blocked_days += 1
    return dict(status=-1, tau=min(n, t) - i0, max_bal=max_bal, reached=reached, blocked_days=blocked_days, end=t - 1, best=best, bal=bal)


def main() -> None:
    D = R.load(); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for g in ("noise", "gap"):
        s, n = D["ST"][g], D["n"]; d = pd.DatetimeIndex(s.dates)
        i_first = int(np.searchsorted(s.dates, np.datetime64("2023-01-02")))
        ev = R.AcctEval(s, n, R.BASE["R"], starts=list(range(i_first, n - 130, 3)))
        ph = ev.spec.eval_phases[0]
        rows = []
        for i0, status, tau in ev.ev:
            r = trace(ph, ev.x, ev.lo, ev.nd, i0)
            assert (r["status"], r["tau"]) == (status, tau), (g, d[i0], r, status, tau)       # parity with the engine
            r.update(start=d[i0], end_date=d[min(r["end"], n - 1)], tgt_frac=r["max_bal"] / ph.target, micros=int(ev.nd[i0]))
            rows.append(r)
        df = pd.DataFrame(rows); df["half"] = df.start.dt.year.astype(str) + "H" + np.where(df.start.dt.month <= 6, "1", "2")
        df.to_csv(f"results/diag_combine_{g}.csv", index=False)
        print(f"\n===== {g}: parity with the engine OK for {len(df)} Combine starts; target ${ph.target:,.0f}, trailing limit ${ph.dd:,.0f}, consistency {ph.consistency}")
        out = []
        for h, x in df.groupby("half"):
            out.append(dict(half=h, starts=len(x), passed=(x.status == 1).mean(), breached=(x.status == 0).mean(), reached_target=x.reached.notna().mean(), blocked_by_consistency_then_breach=((x.status == 0) & x.reached.notna()).mean(),
                            median_max_progress=x.tgt_frac.median(), p90_max_progress=x.tgt_frac.quantile(0.9), median_micros=x.micros.median(), median_sessions=x.tau.median()))
        print(pd.DataFrame(out).set_index("half").round(3).to_string())
        b = df[(df.start >= "2024-07-01") & (df.status == 0)]
        print(f"breach month of Combines started since 2024-07:"); print(b.end_date.dt.to_period("M").value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
