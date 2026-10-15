"""Audit follow-up: 'accounts paid' and 'days to payout' for the funded accounts the farm ACTUALLY produces (not the 3:2 replicate pooling of sections 22-26).
Farms are started every STEP sessions through the development period and run to the end of 2022 (accounts live out their life); accounts are de-duplicated by (group, Combine purchase
session, activation session). Also one continuous farm from the first valid session. Uses paper.farm.PaperFarm (parity-tested against multi_firm.run_multifirm)."""
from __future__ import annotations
import sys
from dataclasses import replace
import numpy as np, pandas as pd
from paper.farm import BALANCED_PLAN, PaperFarm
from portfolio import DEV_END
from real_engine import FundedPlan
import run_kelly as K

STEP = 14
pd.set_option("display.width", 250, "display.max_columns", 40)


def plan(a1, t1, a2, scale=None, scale2=None, w1=0.5):
    return FundedPlan("p", "x", t1, w1, 1.0, 0.0, 0.5, 1.0, 99, a1, a2, scale, scale2)


CONFIGS = {
    "baseline (both groups 1.2 / $2,000, 0.8 after)": {"noise": plan(1.2, 2000.0, 0.8), "gap": plan(1.2, 2000.0, 0.8)},
    "option 2 (after payout 0.4, 1.2 at $3,500)": {g: plan(1.2, 2000.0, 0.4, None, ("ladder", ((3500.0, 1.2),))) for g in ("noise", "gap")},
    "option 3 (before payout 0.8, 2.0 at +$500)": {g: plan(0.8, 2000.0, 0.8, ("ladder", ((500.0, 2.0),))) for g in ("noise", "gap")},
    "A (noise 1.2 / $750, gap 0.4 / $1,000)": {"noise": plan(1.2, 750.0, 0.8), "gap": plan(0.4, 1000.0, 0.8)},
    "B (noise 0.6 / $750, gap 0.8 / $750)": {"noise": plan(0.6, 750.0, 0.8), "gap": plan(0.8, 750.0, 0.8)},
    "two-tier (noise 1.2 / $1,000, gap 0.4 / $2,000)": {"noise": plan(1.2, 1000.0, 0.8), "gap": plan(0.4, 2000.0, 0.8)},
    "old safe on both groups (0.4 / $4,000)": {g: plan(0.4, 4000.0, 0.8) for g in ("noise", "gap")},
}


def farm_accounts(ST, plans, i0, n_end, d):
    ref = next(iter(ST.values()))
    farm = PaperFarm(stop_new_at=n_end - 5, plans=plans)
    nd = {g: s.desired_micros(farm.R) for g, s in ST.items()}
    for t in range(i0, n_end):
        farm.begin_session(t, ref.dates[t], {g: dict(ok=bool(s.ok[t]), atr_usd=float(s.atr_usd[t]), nd=int(nd[g][t])) for g, s in ST.items()})
        farm.end_session({g: dict(x=float(s.pnl[t]), lo=float(s.low[t])) for g, s in ST.items()})
    rows = []
    for fr in farm.all_funded():
        rows.append(dict(group=fr.group, t_buy=fr.t_buy, t_act=fr.t_act, paid=fr.first_pay_t is not None, days=(d[fr.first_pay_t] - d[fr.t_buy]).days if fr.first_pay_t is not None else np.nan,
                         n_pay=len(fr.acct.pays), total=sum(fr.acct.pays), died=fr.acct.state == 1))
    return rows


def summarize(A):
    f = len(A); p = A[A.paid]
    return dict(funded=f, noise_share=(A.group == "noise").mean(), paid=p.shape[0] / f if f else np.nan, paid_noise=A[A.group == "noise"].paid.mean(), paid_gap=A[A.group == "gap"].paid.mean() if (A.group == "gap").any() else np.nan,
                days_med=p.days.median(), days_p25=p.days.quantile(.25), days_p75=p.days.quantile(.75), within_90=(p.days <= 90).sum() / f if f else np.nan, within_180=(p.days <= 180).sum() / f if f else np.nan,
                payouts_per_paid=p.n_pay.mean(), paid_per_paid=p.total.mean())


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}
    ref = ST["noise"]; d = pd.DatetimeIndex(ref.dates); n = len(d); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    first = int(np.flatnonzero(ref.ok)[0]); starts = list(range(first, n_dev - 400, STEP))
    out = []
    for lab, plans in CONFIGS.items():
        allrows = []
        for i0 in starts:
            allrows += [dict(r, farm=i0) for r in farm_accounts(ST, plans, i0, n_dev, d)]
        A = pd.DataFrame(allrows); raw = len(A)
        U = A.drop_duplicates(["group", "t_buy", "t_act"])
        cont = pd.DataFrame(farm_accounts(ST, plans, first, n_dev, d))
        for kind, df in (("many farms, de-duplicated accounts", U), ("one continuous farm 2010-2022", cont)):
            out.append(dict(config=lab, sample=kind, raw_accounts=raw if kind.startswith("many") else len(df), **summarize(df)))
        print(f"done {lab}: farms {len(starts)}, funded {raw} -> {len(U)} distinct", flush=True)
    import os
    tag = os.path.basename(os.environ.get("TOPSTEP_RAW_DIR", "data_raw"))
    R = pd.DataFrame(out); R.to_csv(f"results/audit_account_metrics_{tag}.csv", index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
