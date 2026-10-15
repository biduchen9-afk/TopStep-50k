"""Why are farms started since mid-2024 flat? Diagnostic (descriptive, nothing selected): Combine and funded-account outcomes by half-year of the start, per sleeve, as modelled costs
(= today's cost environment in these years), baseline cells. Then the same starts under other splits and one-at-a-time setting changes to see what moves them.
Run with TOPSTEP_RAW_DIR=/home/user/data_raw_live."""
from __future__ import annotations

import math
import os

os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import numpy as np
import pandas as pd

import retest as R
from firm_engine import run_funded_firm
from multi_firm import Policy, run_multifirm

BASE = R.BASE


def account_diag(D: dict, g: str, first_date: str = "2023-01-02", min_follow: int = 130) -> pd.DataFrame:
    s, n = D["ST"][g], D["n"]
    d = pd.DatetimeIndex(s.dates)
    i_first = int(np.searchsorted(s.dates, np.datetime64(first_date)))
    ev = R.AcctEval(s, n, BASE["R"], starts=list(range(i_first, n - min_follow, 3)))
    plan = R.mk_plan(BASE["a1"], BASE["t1"], BASE["a2"])
    rows = []
    for i0, status, tau in ev.ev:
        r = dict(start=d[i0], status=status, tau=tau, group=g)
        if status == 1:
            act = min(i0 + tau + 1, n - 1); tm: list = []
            pays, days, state, _ = run_funded_firm(ev.spec.funded, ev.x, ev.lo, ev.nd, act, plan, ev.atr, tm, None, ev.fx)
            r.update(act=d[act], follow=n - 1 - act, days_alive=days, died=int(state == 1), n_pay=len(pays), pay_total=sum(pays), first_pay_days=(d[min(tm[0], n - 1)] - d[i0]).days if pays else np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["half"] = df.start.dt.year.astype(str) + "H" + np.where(df.start.dt.month <= 6, "1", "2")
    return df


def main() -> None:
    D = R.load()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for g in ("noise", "gap"):
        df = account_diag(D, g)
        print(f"\n===== {g}: Combine and funded outcomes by half-year of the Combine start (baseline cells; follow-up to 2026-07-02)")
        out = []
        for h, x in df.groupby("half"):
            p = x[x.status == 1]
            out.append(dict(half=h, starts=len(x), pass_=(x.status == 1).mean(), breach=(x.status == 0).mean(), timeout=(x.status == -1).mean(), sessions_to_pass=p.tau.median() if len(p) else np.nan,
                            funded=len(p), died_before_end=p.died.mean() if len(p) else np.nan, paid=(p.n_pay > 0).mean() if len(p) else np.nan, alive_median_sessions=p.days_alive.median() if len(p) else np.nan,
                            payouts_per_funded=p.n_pay.mean() if len(p) else np.nan, usd_per_funded=p.pay_total.mean() if len(p) else np.nan, first_pay_days=p.first_pay_days.median() if len(p) else np.nan))
        print(pd.DataFrame(out).set_index("half").round(3).to_string())
        df.to_csv(f"results/diag_since2024_{g}.csv", index=False)


if __name__ == "__main__":
    main()
