"""Month-by-month cash flows of the 5-funded / <=10-eval farm over the FULL history, realistic micro fees. -> results/dashboard_data.json"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import intraday_strategies as st
from market_data import build_sessions
from real_engine import Stream
from run_corrected import NEW, cfg, farm, plan, streams, idx_of
from portfolio import DEV_END
from real_engine import costs_from_config


def monthly(s, res, label):
    d = pd.DatetimeIndex(s.dates)
    n = len(d)
    ev = pd.DataFrame(res["events"], columns=["t", "kind", "amt"])
    ev["date"] = d[np.minimum(ev.t.to_numpy(), n - 1)]
    ev["m"] = ev.date.dt.to_period("M")
    dl = pd.DataFrame(res["daily"], columns=["t", "evals", "funded"])
    dl["m"] = d[np.minimum(dl.t.to_numpy(), n - 1)].to_period("M")
    months = pd.period_range(d[idx_first(s)].to_period("M"), d[-1].to_period("M"), freq="M")
    rows = []
    for m in months:
        e = ev[ev.m == m]
        k = e.groupby("kind").amt.agg(["sum", "count"])
        g = lambda kind, col: float(k.loc[kind, col]) if kind in k.index else 0.0
        a = dl[dl.m == m]
        rows.append(dict(m=str(m), payouts=g("payout", "sum"), payout_count=int(g("payout", "count")), eval_fees=g("eval_fee", "sum"),
                         activation_fees=g("activation", "sum"), evals_failed=int(g("eval_fail", "count")), evals_passed=int(g("eval_pass", "count")),
                         activations=int(g("activation", "count")), funded_died=int(g("funded_died", "count")), funded_closed=int(g("funded_closed", "count")),
                         avg_evals=float(a.evals.mean()) if len(a) else 0.0, avg_funded=float(a.funded.mean()) if len(a) else 0.0))
    return rows


def idx_first(s):
    return int(np.flatnonzero(s.ok)[0])


def finish(rows):
    cum = cp = cf = 0.0
    for r in rows:
        r["fees"] = r["eval_fees"] + r["activation_fees"]
        r["net"] = r["payouts"] - r["fees"]
        cum += r["net"]; cp += r["payouts"]; cf += r["fees"]
        r["cum_net"], r["cum_payouts"], r["cum_fees"] = cum, cp, cf
    return rows


def main() -> None:
    S = streams(NEW)
    nq, pf = S["NQ noise-area"], S["Portfolio"]
    nq_sess = build_sessions("nq")
    variants = []
    def run(label, vid, note, s, pl):
        r = farm(s, pl, idx_first(s), log=True)
        variants.append(dict(id=vid, label=label, note=note, months=finish(monthly(s, r, label)), totals=dict(net=r["net"], payouts=r["payouts"], fees=r["fees"], funded=r["n_funded"], evals=r["n_evals"], n_payouts=r["n_payouts"])))
    run("NQ noise-area, plan chosen on 2010-22 data (a2=1.6)", "nq_frozen", "Pre-registered style: parameters picked on dev data only.", nq, plan(0.4, 1.6))
    run("NQ noise-area, a2=0.4 (picked after seeing 2023-26)", "nq_posthoc", "Post-hoc: the moderate a2 was chosen after the hold-out was seen.", nq, plan(0.4, 0.4))
    run("Portfolio (NQ x2.3, NQ-VWAP, ES x2, GC), a2=0.4", "portfolio", "Sleeves/weights dev-based; sleeve list chosen with full-sample results visible.", pf, plan(0.4, 0.4))
    # no-edge control: mean of 10 coin-flip seeds, same plan as the post-hoc NQ run
    acc = None
    ctl_tot = []
    for sd in range(10):
        c = Stream.from_series("coin", st.run(nq_sess, "coin", seed=sd, s=None, costs=costs_from_config(cfg)))
        first = int(np.flatnonzero(c.ok)[0])
        r = farm(c, plan(0.4, 0.4), first, log=True)
        rows = monthly(c, r, "coin")
        df = pd.DataFrame(rows).set_index("m")
        acc = df if acc is None else acc.add(df, fill_value=0.0)
        ctl_tot.append(r["net"])
    rows = (acc / 10).reset_index().to_dict("records")
    for r in rows:
        r["payout_count"] = float(r["payout_count"])
    variants.append(dict(id="control", label="No-edge control (coin-flip), same plan, mean of 10 runs", note="Shows what the payout rules alone give with zero edge.",
                         months=finish(rows), totals=dict(net=float(np.mean(ctl_tot)), net_min=float(np.min(ctl_tot)), net_max=float(np.max(ctl_tot)))))
    for v in variants:
        v["peak_outlay"] = float(-min(0.0, min(m["cum_net"] for m in v["months"])))
    out = dict(boundary="2022-12", variants=variants, assumptions=dict(
        farm="up to 5 funded accounts and 10 concurrent evals, all on one market path", path="no_activation_dll ($199/month, no activation fee, -$3,000 daily loss limit)",
        costs="$1.22 all-in per micro round turn + 1 tick slippage", payouts="every eligible payout taken at the 50% maximum, assumed approved instantly",
        sizing="Combine fixed ~$6,400 risk per ATR; funded risk = a x cushion above the loss floor", data="NQ/ES/GC 1-minute bars, 2010-06 to 2026-07"))
    json.dump(out, open("results/dashboard_data.json", "w"))
    for v in variants:
        t = v["totals"]
        print(f"{v['id']:<11} net {t['net']:>9,.0f} | payouts {t.get('payouts', float('nan')):>9,.0f} | fees {t.get('fees', float('nan')):>8,.0f} | peak outlay {v['peak_outlay']:>8,.0f}")
    # yearly net per variant
    ydf = pd.DataFrame({v["id"]: pd.DataFrame(v["months"]).assign(y=lambda d: d.m.str[:4]).groupby("y").net.sum() for v in variants})
    print("\nNet by calendar year (realistic fees):"); print(ydf.round(0).to_string())


if __name__ == "__main__":
    main()
