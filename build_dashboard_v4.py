"""Dashboard data (v4): adds the Topstep default (3 noise-area + 2 gap-VWAP accounts) as a variant, the per-firm split study, the full scaling pipeline (month by month),
the Kelly sizing study and the days-to-payout comparison, on top of the v3 files. Reads results/dashboard_v3/*.json, writes JSON files to argv[1]."""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from portfolio import DEV_END
from real_engine import Stream
from risk_units import with_risk_unit
from multi_firm import Policy, run_multifirm

OUT = sys.argv[1]; V3 = "results/dashboard_v3"
sys.argv = sys.argv[:1]
from run_monthly_diversified import group_stream
import run_kelly as K
import run_scaling_pipeline as SP
from run_firm_splits import CFG, S as SPECS

gs = lambda names: with_risk_unit(group_stream(names), "atr", 40)
ST = {"noise": gs(["nq_noise"]), "vwap": gs(["nq_vwap"]), "gap": gs(["nq_gap30"]), "basket": gs(["nq_noise", "nq_vwap", "nq_gap30"])}
ref = ST["basket"]; d = pd.DatetimeIndex(ref.dates); n = len(d); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt))); first = int(np.flatnonzero(ref.ok)[0])


def monthly_rows(res, i0, firm_of=lambda name: "all", firms=("all",)):
    ev = pd.DataFrame(res["events"], columns=["t", "name", "kind", "amt"]); ev["firm"] = ev.name.map(firm_of); ev["m"] = d[np.minimum(ev.t.to_numpy(), n - 1)].to_period("M")
    dl = res["daily"]; dm = pd.DataFrame({"t": [x[0] for x in dl], "cash": [x[1] for x in dl]}); dm["m"] = d[np.minimum(dm.t.to_numpy(), n - 1)].to_period("M")
    per = {f: pd.DataFrame({"t": dm.t, "m": dm.m, "evals": [sum(e for (e, fu), nm in zip(x[2], res["order"]) if firm_of(nm) == f) for x in dl], "funded": [sum(fu for (e, fu), nm in zip(x[2], res["order"]) if firm_of(nm) == f) for x in dl]}) for f in firms}
    rows = []; cum = 0.0
    for m in pd.period_range(d[min(i0, n - 1)].to_period("M"), d[-1].to_period("M"), freq="M"):
        e = ev[ev.m == m]; r = dict(month=str(m) + "-01")
        tp = tf = 0.0
        for f in firms:
            ef = e[e.firm == f]
            pay = float(ef[ef.kind == "payout"].amt.sum()); fee = float(ef[ef.kind.isin(["eval_fee", "activation"])].amt.sum())
            a = per[f][per[f].m == m]
            r[f"payouts_{f}"] = round(pay, 2); r[f"fees_{f}"] = round(fee, 2); r[f"funded_{f}"] = round(float(a.funded.mean()) if len(a) else 0.0, 2); r[f"evals_{f}"] = round(float(a.evals.mean()) if len(a) else 0.0, 2)
            r[f"payout_n_{f}"] = int((ef.kind == "payout").sum()); tp += pay; tf += fee
        r["payouts"] = round(tp, 2); r["fees"] = round(tf, 2); r["net"] = round(tp - tf, 2); cum += tp - tf; r["cum_net"] = round(cum, 2)
        c = dm[dm.m == m]; r["cash"] = round(float(c.cash.iloc[-1]) if len(c) else 0.0, 2)
        r["payout_count"] = int((e.kind == "payout").sum()); r["evals_passed"] = int((e.kind == "eval_pass").sum()); r["evals_failed"] = int((e.kind == "eval_fail").sum())
        r["activations"] = int((e.kind == "activation").sum()); r["funded_died"] = int((e.kind == "funded_died").sum()); r["funded_closed"] = int((e.kind == "funded_closed").sum())
        r["avg_evals"] = round(sum(r[f"evals_{f}"] for f in firms), 2); r["avg_funded"] = round(sum(r[f"funded_{f}"] for f in firms), 2)
        rows.append(r)
    return rows


def logged(fps, pol, i0, n_end=None):
    res = run_multifirm(fps, pol, i0, n_end, log=True); res["order"] = [f.spec.name for f in fps]; return res


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    L = lambda name: json.load(open(f"{V3}/{name}.json"))
    monthly, variants, yearly, starts = L("monthly"), L("variants"), L("yearly"), L("start_dates")
    pol0 = Policy(reserve_cycles=None, global_one_eval=True)
    # ---------- default variant (and its edge-removed control)
    dev = lambda s: (s.dates <= DEV_END) & s.ok
    PL = {}; PS = {}
    for g in ("noise", "gap"):
        s = ST[g]; mu_R = float(np.mean(s.pnl[dev(s)] / s.atr_usd[dev(s)])); dm_ = np.where(s.ok, s.pnl - mu_R * np.nan_to_num(s.atr_usd), s.pnl)
        PS[g] = Stream(s.name, s.dates, dm_, np.minimum(s.low - (s.pnl - dm_), 0.0), s.atr_usd, s.ok); PL[g] = dm_
    DEF = [("default", "Default: 3 noise-area + 2 gap-VWAP accounts, balanced", "One sleeve per group of accounts, one Combine open at a time, first payout at $2,000, size 1.2 then 0.8.", ST),
           ("default_control", "Edge removed, default settings", "Same path, same split, the strategies' average edge set to zero.", PS)]
    for vid, label, note, stm in DEF:
        res = logged(K.plans(stm, 1.2, 0.8, False), pol0, first)
        rows = monthly_rows(res, first)
        for r in rows:
            monthly.append(dict(variant=vid, month=r["month"], sample="in-sample" if r["month"][:7] <= "2022-12" else "out-of-sample", payouts=r["payouts"], payout_count=float(r["payout_count"]), fees=r["fees"], net=r["net"], cum_net=r["cum_net"],
                                cum_payouts=round(sum(x["payouts"] for x in rows if x["month"] <= r["month"]), 2), cum_fees=round(sum(x["fees"] for x in rows if x["month"] <= r["month"]), 2), evals_passed=float(r["evals_passed"]), evals_failed=float(r["evals_failed"]),
                                activations=float(r["activations"]), funded_died=float(r["funded_died"]), funded_closed=float(r["funded_closed"]), avg_evals=r["avg_evals"], avg_funded=r["avg_funded"]))
        ms = pd.DataFrame(rows); pay, fee = float(ms.payouts.sum()), float(ms.fees.sum()); oos = ms.month >= "2023-01-01"
        variants.append(dict(id=vid, label=label, note=note, net_total=round(pay - fee, 2), net_in_sample=round(float(ms[~oos].net.sum()), 2), net_out_of_sample=round(float(ms[oos].net.sum()), 2), payouts_total=round(pay, 2), fees_total=round(fee, 2),
                             reinvest_pct=round(fee / pay, 4) if pay > 0 else 0.0, peak_outlay=round(float(-min(0.0, ms.cum_net.min())), 2), funded_accounts=int(ms.activations.sum()), payout_count=int(ms.payout_count.sum()), months=int(len(ms))))
        ms["year"] = ms.month.str[:4]
        for y, g in ms.groupby("year"):
            yearly.append(dict(variant=vid, year=y, payouts=round(float(g.payouts.sum()), 2), fees=round(float(g.fees.sum()), 2), net=round(float(g.net.sum()), 2), payout_count=round(float(g.payout_count.sum()), 2), activations=round(float(g.activations.sum()), 2),
                               funded_died=round(float(g.funded_died.sum()), 2), avg_funded=round(float(g.avg_funded.mean()), 2), sample="in-sample" if y <= "2022" else "out-of-sample"))
        if vid == "default":
            fps = K.plans(ST, 1.2, 0.8, False)
            for dt in ["2012-01-03", "2015-01-02", "2018-01-02", "2019-01-02", "2021-01-04", "2023-01-02", "2024-01-02", "2025-01-02", "2025-07-01"]:
                r = run_multifirm(fps, pol0, idx(dt)); starts.append(dict(config=label, start=dt, net=round(r["net"], 2), payouts=round(r["payouts"], 2), fees=round(r["fees"], 2), n_payouts=int(sum(v["n_payouts"] for v in r["firms"].values())),
                                                                       funded=int(sum(v["n_funded"] for v in r["firms"].values()))))
    json.dump(monthly, open(f"{OUT}/monthly.json", "w")); json.dump(variants, open(f"{OUT}/variants.json", "w")); json.dump(yearly, open(f"{OUT}/yearly.json", "w")); json.dump(starts, open(f"{OUT}/start_dates.json", "w"))
    # ---------- Part A: splits per firm
    A = pd.read_csv("results/firm_splits.csv"); pick = pd.read_pickle("results/firm_split_choice.pkl")
    arows = []
    for r in A.itertuples():
        seen = json.loads(r.seen)
        arows.append(dict(firm=r.firm, split=r.split, dev_mean=round(r.dev_mean), dev_p25=round(r.dev_p25), dev_worst=round(r.dev_worst), green=int(r.green), seen_2021=seen[0], seen_2023=seen[1], seen_2025=seen[2], full_net=round(r.full_net), full_fees=round(r.full_fees),
                          payout_months=round(float(r.months_with_payout), 3), chosen="Chosen by the rule" if pick.get(r.firm) == r.split else ("Your choice" if r.firm == "Topstep 150K" and r.split == "3 noise + 2 gap" else "")))
    json.dump(arows, open(f"{OUT}/split_firms.json", "w"))
    # ---------- Part B: pipeline
    R = pd.read_csv("results/scaling_pipeline.csv")
    g = R.groupby(["case", "unlock", "kind"], dropna=False).agg(mean_net=("net", "mean"), worst_net=("net", "min"), green=("net", lambda v: int((v > 0).sum())), runs=("net", "size"), ruined=("ruined", "sum"), oop_max=("oop", "max"), days_first_payout=("days_first_payout", "median")).reset_index()
    g["rule"] = np.where(g.unlock.isna(), "Topstep only", "Topstep + FundedNext" + np.where(g.case.str.contains("FTMO"), " + FTMO", "") + " (unlock at $" + g.unlock.fillna(0).astype(int).astype(str) + ")")
    g["rule"] = np.where(g.case == "Topstep only", "Topstep only", g.rule)
    prows = [dict(rule=r.rule, kind=r.kind, mean_net=round(r.mean_net), worst_net=round(r.worst_net), green=int(r.green), runs=int(r.runs), ruined=int(r.ruined), oop_max=round(r.oop_max), median_days_first_payout=None if pd.isna(r.days_first_payout) else round(r.days_first_payout)) for r in g.itertuples()]
    json.dump(prows, open(f"{OUT}/pipeline_summary.json", "w"))
    runs = [dict(rule=("Topstep only" if r.case == "Topstep only" else r.case + f" (unlock ${int(r.unlock)})"), kind=r.kind, start=int(r.start), net=round(r.net), oop=round(r.oop), ruined="yes" if r.ruined else "no", days_first_payout=None if pd.isna(r.days_first_payout) else int(r.days_first_payout), by_firm=", ".join(f"{k} {v:,.0f}" for k, v in __import__("ast").literal_eval(r.nets).items())) for r in R.itertuples()]
    json.dump(runs, open(f"{OUT}/pipeline_runs.json", "w"))
    dv = R[R.kind.isin(["dev 3y", "dev long"]) & (R.case == "Topstep + FundedNext + FTMO")].groupby("unlock").agg(ruins=("ruined", "sum"), mean=("net", "mean")).reset_index().sort_values(["ruins", "mean"], ascending=[True, False])
    unlock = float(dv.iloc[0].unlock); print("rule-chosen unlock threshold:", unlock); print(dv)
    choice = pd.read_pickle("results/firm_split_choice.pkl")
    fps = SP.build(ST, choice); pol = Policy(k0=1, reserve_cycles=3.0, global_one_eval=True, group_aware=True, unlock_cash=unlock)
    firm_of = lambda nm: nm.split(":")[0]; trows = []
    for lab, y in (("Full history from 2010", None), ("Start 2012", 2012), ("Start 2018", 2018), ("Start 2021", 2021), ("Start 2023", 2023), ("Start 2025", 2025)):
        i0 = first if y is None else idx(f"{y}-01-01")
        res = logged(fps, pol, i0)
        for r in monthly_rows(res, i0, firm_of, ("Topstep", "FundedNext", "FTMO")):
            trows.append(dict(start=lab, **r))
        print(lab, round(res["net"]), "oop", round(res["oop_total"]), "ruined", res["ruined_t"] is not None, {k: (round(v["net"]), v["n_funded"]) for k, v in res["firms"].items()})
    json.dump(trows, open(f"{OUT}/pipeline_months.json", "w"))
    # ---------- Part C, D
    Kd = pd.read_csv("results/kelly_study.csv"); b0 = Kd.iloc[0]

    def verdict(i, r):                                  # the pre-registered 'not greedy' acceptance rule against the current setting
        if i == 0:
            return "Current setting"
        why = []
        if not r.dev_mean > b0.dev_mean: why.append("lower mean")
        if not r.dev_p25 > b0.dev_p25: why.append("worse bad-quartile farm")
        if not r.dev_worst >= (b0.dev_worst * 1.25 if b0.dev_worst < 0 else b0.dev_worst * 0.75): why.append("deeper worst farm")
        if not r.paid_share >= b0.paid_share - 0.05: why.append("fewer accounts paid")
        return "Passes" if not why else "Rejected: " + ", ".join(why)
    krows = [dict(config=r.config, kind="Volatility rule" if r.kind == "atr" else "Kelly multiple", pre_payout=r.f1, post_payout=r.f2, dev_mean=round(r.dev_mean), dev_p25=round(r.dev_p25), dev_worst=round(r.dev_worst), green=int(r.green), paid_share=round(r.paid_share, 3), died_share=round(r.died_share, 3),
                   pay_per_paid=round(r.pay_per_paid), median_days_first_payout=None if pd.isna(r.days_first_payout) else round(r.days_first_payout), seen_2021=json.loads(r.seen)[0], seen_2023=json.loads(r.seen)[1], seen_2025=json.loads(r.seen)[2], full_net=round(r.full_net), verdict=verdict(i, r)) for i, r in enumerate(Kd.itertuples())]
    Hd = pd.read_csv("results/hybrid_study.csv")                    # pre-registration 10: volatility rule before the first payout, Kelly after it
    for r in Hd.iloc[1:].itertuples():
        seen = json.loads(r.seen)
        krows.append(dict(config=r.config, kind="Hybrid", pre_payout=1.2, post_payout=r.f2, dev_mean=round(r.dev_mean), dev_p25=round(r.dev_p25), dev_worst=round(r.dev_worst), green=int(r.green), paid_share=round(r.paid_share, 3), died_share=round(r.died_share, 3),
                          pay_per_paid=round(r.pay_per_paid), median_days_first_payout=round(r.days_first_payout), seen_2021=seen[0], seen_2023=seen[1], seen_2025=seen[2], full_net=round(r.full_net), verdict=r.verdict))
    json.dump(krows, open(f"{OUT}/kelly.json", "w"))
    Pa = pd.read_csv("results/days_to_payout_accounts.csv"); Pf = pd.read_csv("results/days_to_payout_farms.csv")
    drows = [dict(level="One account", version=r.version, period=("2010-22 starts" if r.period == "dev" else "2023+ starts"), sample=int(r.attempts), pass_rate=round(r.pass_rate, 3), paid_if_funded=round(r.p_paid_given_funded, 3), days_to_pass=r.days_to_pass_med, days_activation_to_payout=r.days_act_to_payout_med,
                  days_total_median=r.days_start_to_payout_med, days_total_p25=r.days_start_to_payout_p25, days_total_p75=r.days_start_to_payout_p75, combines_per_paid_account=round(r.attempts_per_paid_account, 1)) for r in Pa.itertuples()]
    drows += [dict(level="Whole farm", version=r.version + " | " + r.rule, period="ten 3-year farms", sample=int(r.farms), pass_rate=None, paid_if_funded=None, days_to_pass=None, days_activation_to_payout=None, days_total_median=r.median, days_total_p25=r.p25, days_total_p75=r.p75, combines_per_paid_account=None) for r in Pf.itertuples()]
    json.dump(drows, open(f"{OUT}/payout_days.json", "w"))
    for f in sorted(os.listdir(OUT)):
        print(f, os.path.getsize(f"{OUT}/{f}") // 1024, "KB")


if __name__ == "__main__":
    main()
