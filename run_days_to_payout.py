"""Pre-registration 9, Part D: days to payout for the Topstep default (3 noise-area + 2 gap-VWAP accounts) vs the old same-basket x5 version, balanced preset.
Per account (start -> pass, activation -> first payout, start -> first payout) on sequential starts, and per farm (first Combine bought -> first payout), unconstrained and in the self-funded pipeline."""
from __future__ import annotations
import sys
from dataclasses import replace
import numpy as np, pandas as pd
from firm_engine import run_eval_firm, run_funded_firm
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from risk_units import with_risk_unit
from run_firm_comparison import mkplan

sys.argv = sys.argv[:1]
from run_monthly_diversified import group_stream

pd.set_option("display.width", 250, "display.max_columns", 40)
S = specs()["Topstep 150K"]; R = 9600.0; T1, A1, A2 = 2000.0, 1.2, 0.8


def attempts(s, starts, n_end):
    spec = S["spec"]; d = pd.DatetimeIndex(s.dates); plan = mkplan(S, T1, A1, A2)
    x, lo, atr = s.pnl[:n_end].tolist(), s.low[:n_end].tolist(), s.atr_usd[:n_end].tolist(); nd = s.desired_micros(R)[:n_end]
    rows = []
    for i0 in starts:
        st, tau = run_eval_firm(spec.eval_phases, x, lo, nd, i0)
        r = dict(passed=st == 1, d_pass=np.nan, funded_paid=False, d_act_pay=np.nan, d_total=np.nan, first=np.nan)
        if st == 1:
            act = min(i0 + tau + 1, n_end - 1); r["d_pass"] = (d[act] - d[i0]).days
            tm = []; pays, *_ = run_funded_firm(spec.funded, x, lo, nd, act, plan, atr, tm)
            if pays:
                r.update(funded_paid=True, d_act_pay=(d[min(tm[0], n_end - 1)] - d[act]).days, d_total=(d[min(tm[0], n_end - 1)] - d[i0]).days, first=pays[0])
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df):
    p = df[df.passed]
    return dict(attempts=len(df), pass_rate=df.passed.mean(), p_paid_given_funded=p.funded_paid.mean() if len(p) else np.nan, days_to_pass_med=p.d_pass.median(),
                days_act_to_payout_med=p.d_act_pay.median(), days_start_to_payout_med=p.d_total.median(), days_start_to_payout_p25=p.d_total.quantile(.25), days_start_to_payout_p75=p.d_total.quantile(.75),
                first_payout_mean=p.first.mean(), attempts_per_paid_account=len(df) / max(int(df.funded_paid.sum()), 1))


def farm_days(fps, ref, n, n_dev, pol, years):
    idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt))); d = pd.DatetimeIndex(ref.dates); out = []
    for y in years:
        i0 = idx(f"{y}-01-01"); r = run_multifirm(fps, pol, i0, min(i0 + 756, n_dev))
        fb = [v["first_buy_t"] for v in r["firms"].values() if v["first_buy_t"] is not None]; fp = [v["first_payout_t"] for v in r["firms"].values() if v["first_payout_t"] is not None]
        out.append((d[min(min(fp), n - 1)] - d[min(fb)]).days if fb and fp else np.nan)
    return np.array(out, dtype=float)


def main() -> None:
    gs = lambda names: with_risk_unit(group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"]), "basket": gs(["nq_noise", "nq_vwap", "nq_gap30"])}
    ref = ST["basket"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right")); idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt)))
    first = int(np.flatnonzero(ref.ok)[0]) + 260
    starts_dev = list(range(first, n_dev - 300, 3)); starts_seen = list(range(idx("2023-01-01"), n - 300, 3))
    rows = []
    for per, starts, ne in (("dev", starts_dev, n_dev), ("2023+", starts_seen, n)):
        A = {k: attempts(ST[k], starts, ne) for k in ST}
        pools = {"old: same basket x5": A["basket"], "new default: 3 noise + 2 gap": pd.concat([A["noise"]] * 3 + [A["gap"]] * 2, ignore_index=True), "noise-area sleeve alone": A["noise"], "gap-VWAP sleeve alone": A["gap"]}
        for lab, df in pools.items():
            rows.append(dict(period=per, version=lab, **summarize(df)))
    P = pd.DataFrame(rows); P.to_csv("results/days_to_payout_accounts.csv", index=False); print(P.round(2).to_string(index=False))
    mk = lambda groups: [FirmPlan(replace(S["spec"], name=g, max_funded=k, max_evals=1), [ST[g]], R, mkplan(S, T1, A1, A2), p_est=1.0, refill="all", firm="Topstep") for g, k in groups]
    new, old = mk([("noise", 3), ("gap", 2)]), mk([("basket", 5)])
    out = []
    for lab, fps in (("old: same basket x5", old), ("new default: 3 noise + 2 gap", new)):
        for pl, pol in (("unconstrained (cash not gated, one Combine at a time)", Policy(reserve_cycles=None, global_one_eval=True)), ("self-funded pipeline (one account until the first payout)", Policy(reserve_cycles=3.0, global_one_eval=True, group_aware=True))):
            v = farm_days(fps, ref, n, n_dev, pol, range(2011, 2021))
            out.append(dict(version=lab, rule=pl, farms=len(v), reached_payout=int(np.isfinite(v).sum()), median=np.nanmedian(v), p25=np.nanpercentile(v, 25), p75=np.nanpercentile(v, 75), mean=np.nanmean(v), min=np.nanmin(v), max=np.nanmax(v), **{"<=90d": int((v <= 90).sum())}))
    F = pd.DataFrame(out); F.to_csv("results/days_to_payout_farms.csv", index=False); print("\nFARM level: calendar days from the first Combine bought to the first payout (ten 3-year dev farms)"); print(F.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
