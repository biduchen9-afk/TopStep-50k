"""Days to payout and accounts paid for the baseline and the two forward-test candidates, on identical starts.
  baseline : constant 1.2 before the first payout, constant 0.8 after (the default)
  cand. 2  : after the first payout start at 0.4 and step up to 1.2 once the balance reaches $3,500 (pre-payout unchanged)
  cand. 3  : before the first payout start at 0.8 and step up to 2.0 once the balance reaches +$500 (post-payout unchanged)"""
from __future__ import annotations
import numpy as np, pandas as pd
from firm_engine import run_eval_firm, run_funded_firm
from multi_firm import Policy, run_multifirm
from portfolio import DEV_END
import run_kelly as K
import run_paid_share as P

pd.set_option("display.width", 260, "display.max_columns", 50)
CFGS = {"baseline (today)": dict(a1=1.2, a2=0.8, scale=None, scale2=None),
        "candidate 2: after payout 0.4, 1.2 at $3,500": dict(a1=1.2, a2=0.4, scale=None, scale2=("ladder", ((3500.0, 1.2),))),
        "candidate 3: before payout 0.8, 2.0 at +$500": dict(a1=0.8, a2=0.8, scale=("ladder", ((500.0, 2.0),)), scale2=None)}
T1 = 2000.0


def attempts(s, starts, n_end, cfg):
    spec = P.S["spec"]; d = pd.DatetimeIndex(s.dates); plan = P.mk_plan(cfg["a1"], T1, cfg["a2"], None, cfg["scale"], cfg["scale2"])
    x, lo, atr = s.pnl[:n_end].tolist(), s.low[:n_end].tolist(), s.atr_usd[:n_end].tolist(); nd = s.desired_micros(P.R)[:n_end]
    rows = []
    for i0 in starts:
        st, tau = run_eval_firm(spec.eval_phases, x, lo, nd, i0)
        r = dict(passed=st == 1, paid=False, d_act=np.nan, d_first=np.nan, d_second=np.nan, n_pay=0, total=0.0)
        if st == 1:
            act = min(i0 + tau + 1, n_end - 1); r["d_act"] = (d[act] - d[i0]).days
            tm = []; pays, *_ = run_funded_firm(spec.funded, x, lo, nd, act, plan, atr, tm)
            if pays:
                r.update(paid=True, d_first=(d[min(tm[0], n_end - 1)] - d[i0]).days, n_pay=len(pays), total=sum(pays))
                if len(tm) > 1:
                    r["d_second"] = (d[min(tm[1], n_end - 1)] - d[min(tm[0], n_end - 1)]).days
        rows.append(r)
    return pd.DataFrame(rows)


def summ(df):
    f = df[df.passed]; p = f[f.paid]
    return dict(attempts=len(df), pass_rate=df.passed.mean(), funded_paid=p.shape[0] / max(len(f), 1), days_first_med=p.d_first.median(), days_first_p25=p.d_first.quantile(.25), days_first_p75=p.d_first.quantile(.75),
                days_act_to_first_med=(p.d_first - p.d_act).median(), paid_within_60=(p.d_first <= 60).sum() / max(len(f), 1), paid_within_90=(p.d_first <= 90).sum() / max(len(f), 1), paid_within_180=(p.d_first <= 180).sum() / max(len(f), 1),
                days_between_1st_2nd_med=p.d_second.median(), payouts_per_paid=p.n_pay.mean(), paid_per_paid=p.total.mean())


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}; ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right")); d = pd.DatetimeIndex(ref.dates)
    idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt))); first = int(np.flatnonzero(ref.ok)[0]) + 260
    periods = {"2010-2022 starts": (list(range(first, n_dev - 400, 3)), n_dev), "2023+ starts": (list(range(idx("2023-01-01"), n - 300, 3)), n)}
    SPLIT = [("noise", 3), ("gap", 2)]; pol = Policy(reserve_cycles=None, global_one_eval=True)
    rows = []
    for lab, cfg in CFGS.items():
        for per, (starts, ne) in periods.items():
            A = {g: attempts(ST[g], starts, ne, cfg) for g in ST}
            pooled = pd.concat([A["noise"]] * 3 + [A["gap"]] * 2, ignore_index=True)
            r = dict(config=lab, period=per, **summ(pooled)); r["paid_noise"] = summ(A["noise"])["funded_paid"]; r["paid_gap"] = summ(A["gap"])["funded_paid"]; rows.append(r)
        # farm level: ten 3-year dev farms and the seen starts
        fps = P.plans_g(ST, SPLIT, cfg["a1"], T1, cfg["a2"], None, cfg["scale"], cfg["scale2"])
        def fdays(i0, ne):
            res = run_multifirm(fps, pol, i0, ne)
            fb = [v["first_buy_t"] for v in res["firms"].values() if v["first_buy_t"] is not None]; fp = [v["first_payout_t"] for v in res["firms"].values() if v["first_payout_t"] is not None]
            paid = sum(v["n_paid_accts"] for v in res["firms"].values()); fund = sum(v["n_funded"] for v in res["firms"].values())
            return ((d[min(min(fp), n - 1)] - d[min(fb)]).days if fb and fp else np.nan), paid, fund
        dv = [fdays(idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev)) for y in range(2011, 2021)]
        sn = {y: fdays(idx(dt), None) for y, dt in ((2021, "2021-01-04"), (2023, "2023-01-02"), (2025, "2025-01-02"))}
        days = np.array([v[0] for v in dv], dtype=float)
        rows.append(dict(config=lab, period="whole farm: ten 3-year dev farms", farm_days_med=np.nanmedian(days), farm_days_p25=np.nanpercentile(days, 25), farm_days_p75=np.nanpercentile(days, 75), farms_within_90=int((days <= 90).sum()), farms_within_180=int((days <= 180).sum()),
                         farm_paid_share=sum(v[1] for v in dv) / max(sum(v[2] for v in dv), 1), farm_days_2021=sn[2021][0], farm_days_2023=sn[2023][0], farm_days_2025=sn[2025][0]))
    R = pd.DataFrame(rows); R.to_csv("results/days_candidates.csv", index=False)
    acct = R[R.period.isin(list(periods))]
    print(acct[["config", "period", "attempts", "pass_rate", "funded_paid", "paid_noise", "paid_gap", "days_first_med", "days_first_p25", "days_first_p75", "days_act_to_first_med", "paid_within_60", "paid_within_90", "paid_within_180", "days_between_1st_2nd_med", "payouts_per_paid", "paid_per_paid"]].round(3).to_string(index=False))
    print(); print(R[R.period.str.startswith("whole farm")].drop(columns=["period"]).dropna(axis=1, how="all").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
