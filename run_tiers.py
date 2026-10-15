"""Pre-registration 6, Part B: the three lists (conservative / moderate / big risk) with one-at-a-time Combine buying. See docs/PREREGISTRATION_6.md."""
from __future__ import annotations
import sys
from dataclasses import replace
import numpy as np, pandas as pd
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import FundedPlan
from risk_units import with_risk_unit

sys.argv = sys.argv[:1]
from run_monthly_diversified import group_stream

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
SAFE, BAL, FAST = (0.4, 4000.0), (1.2, 2000.0), (2.4, 6000.0)
TS = specs()["Topstep 150K"]["spec"]


def plan(a1t1):
    return FundedPlan("p", "x", a1t1[1], 0.5, 1.0, 0.0, 0.5, 1.0, 99, a1t1[0], 0.8)


def fp(name, stream, a1t1, slots):
    return FirmPlan(replace(TS, name=name, max_funded=slots, max_evals=1), [stream], 9600.0, plan(a1t1), p_est=1.0, refill="all")


def main() -> None:
    gs = lambda names: with_risk_unit(group_stream(names), "atr", 40)
    NQ3 = gs(["nq_noise", "nq_vwap", "nq_gap30"]); GAP = gs(["nq_gap30"]); NQ3_ES = gs(["nq_noise", "nq_vwap", "nq_gap30", "es_vwap_t250"])
    d = pd.DatetimeIndex(NQ3.dates); n = len(d); n_dev = int(np.searchsorted(NQ3.dates, DEV_END, side="right"))
    idx = lambda dt: int(np.searchsorted(NQ3.dates, np.datetime64(dt)))
    cases = {
        "all 5 slots SAFE, NQ basket": [fp("single", NQ3, SAFE, 5)],
        "all 5 slots BALANCED, NQ basket (default)": [fp("single", NQ3, BAL, 5)],
        "all 5 slots FAST, NQ basket": [fp("single", NQ3, FAST, 5)],
        "3 lists 1 big / 2 moderate / 2 conservative": [fp("big", GAP, FAST, 1), fp("moderate", NQ3, BAL, 2), fp("conservative", NQ3, SAFE, 2)],
        "3 lists, balanced+safe only (no big)": [fp("moderate", NQ3, BAL, 3), fp("conservative", NQ3, SAFE, 2)],
        "EXPLORATORY 3 lists, conservative adds ES VWAP (dev 0.39, not accepted)": [fp("big", GAP, FAST, 1), fp("moderate", NQ3, BAL, 2), fp("conservative", NQ3_ES, SAFE, 2)],
    }
    pol0 = Policy(reserve_cycles=None, global_one_eval=True)
    rows, logs = [], {}
    cases["EXTRA (different rule): each list buys its own Combine, 3 open at once"] = [fp("big", GAP, FAST, 1), fp("moderate", NQ3, BAL, 2), fp("conservative", NQ3, SAFE, 2)]
    for lab, fps in cases.items():
        pol = Policy(reserve_cycles=None, global_one_eval=not lab.startswith("EXTRA"))
        nets, fees_l, firsts = [], [], []
        for y in range(2011, 2021):
            i0 = idx(f"{y}-01-01"); ne = min(i0 + 756, n_dev)
            r = run_multifirm(fps, pol, i0, ne)
            nets.append(r["net"]); fees_l.append(r["fees"])
            fpay = [v["first_payout_t"] for v in r["firms"].values() if v["first_payout_t"] is not None]
            fb = min(v["first_buy_t"] for v in r["firms"].values() if v["first_buy_t"] is not None)
            firsts.append((d[min(min(fpay), n - 1)] - d[fb]).days if fpay else np.nan)
        nets = np.array(nets)
        seen = [round(run_multifirm(fps, pol, idx(x))["net"]) for x in ("2021-01-04", "2023-01-02", "2025-01-02", "2025-07-01")]
        i0 = int(np.flatnonzero(NQ3.ok)[0]); full = run_multifirm(fps, pol, i0, log=True)
        ev = pd.DataFrame(full["events"], columns=["t", "tier", "kind", "amt"]); ev["m"] = d[np.minimum(ev.t.to_numpy(), n - 1)].to_period("M")
        ev["sgn"] = np.where(ev.kind == "payout", ev.amt, np.where(ev.kind.isin(["activation", "eval_fee"]), -ev.amt, 0.0))
        mo = ev.pivot_table(index="m", columns="tier", values="sgn", aggfunc="sum").reindex(pd.period_range("2012-01", "2026-06", freq="M")).fillna(0.0)
        pay_m = ev[ev.kind == "payout"].groupby("m").amt.sum().reindex(mo.index).fillna(0.0)
        logs[lab] = mo
        rows.append(dict(case=lab, dev_mean=nets.mean(), dev_p25=np.percentile(nets, 25), dev_worst=nets.min(), green=int((nets > 0).sum()), dev_fees=np.mean(fees_l), first_payout_days_med=np.nanmedian(firsts),
                         never=int(np.isnan(firsts).sum()), seen_2021_23_25_25b=str(seen), full_net=full["net"], full_payouts=full["payouts"], full_fees=full["fees"], months_with_payout=float((pay_m > 0).mean()),
                         peak_outlay=float(-min(0.0, mo.sum(axis=1).cumsum().min()))))
    R = pd.DataFrame(rows); R.to_csv("results/tiers.csv", index=False)
    print(R.round(2).T.to_string())
    lab = "3 lists 1 big / 2 moderate / 2 conservative"
    print("\ncorrelation of monthly net across lists (2012-2026):"); print(logs[lab].corr().round(2).to_string())
    print("\nnet by list, full history:"); print(logs[lab].sum().round(0).to_string())


if __name__ == "__main__":
    main()
