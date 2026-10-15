"""(1) Combine pass rate by sleeve / basket at several risk settings R (dev starts every 3 sessions, 2010-2022, and 2023+ starts, descriptive).
(2) Diversifying the strategies across funded accounts: every account group runs its own sleeve; compared with all accounts running the same basket."""
from __future__ import annotations
import sys
from dataclasses import replace
import numpy as np, pandas as pd
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import FundedPlan, run_combine, rules_from_config
from pricing_config import load_config
from risk_units import with_risk_unit
from final_strategy import PATH

sys.argv = sys.argv[:1]
from run_monthly_diversified import group_stream

pd.set_option("display.width", 250, "display.max_columns", 40)
TS = specs()["Topstep 150K"]["spec"]; rules = rules_from_config(load_config(), PATH)
BAL = FundedPlan("p", "x", 2000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8)


def main() -> None:
    gs = lambda names: with_risk_unit(group_stream(names), "atr", 40)
    S = {"noise-area only": gs(["nq_noise"]), "VWAP+ATR only": gs(["nq_vwap"]), "gap-VWAP only": gs(["nq_gap30"]), "NQ basket (3 sleeves)": gs(["nq_noise", "nq_vwap", "nq_gap30"]),
         "noise-area + gap-VWAP": gs(["nq_noise", "nq_gap30"]), "ES VWAP (trend-gated)": gs(["es_vwap_t250"]), "GC noise (trend-gated)": gs(["gc_noise_t60"])}
    ref = S["NQ basket (3 sleeves)"]; d = pd.DatetimeIndex(ref.dates); n = len(d); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt)))
    first = int(np.flatnonzero(ref.ok)[0]) + 260; i23 = idx("2023-01-01")
    rows = []
    for name, s in S.items():
        x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist()
        for R in (3200., 4800., 6400., 9600., 12800.):
            nd = s.desired_micros(R)
            for per, rng in (("2010-22", range(first, n_dev - 300, 3)), ("2023+", range(i23, n - 150, 3))):
                res = [run_combine(x, lo, nd, i, rules) for i in rng]
                ps = [t for st_, t in res if st_ == 1]
                rows.append(dict(sleeve=name, R=R, period=per, pass_rate=len(ps) / len(res), breach=np.mean([st_ == 0 for st_, _ in res]), days_to_pass=float(np.median(ps)) * 365 / 252 if ps else np.nan, n=len(res)))
    P = pd.DataFrame(rows); P.to_csv("results/pass_rate_by_sleeve.csv", index=False)
    piv = P.pivot_table(index=["sleeve", "R"], columns="period", values=["pass_rate", "days_to_pass"]).round(2)
    print("COMBINE PASS RATE (and median calendar days to pass) by sleeve and risk R:"); print(piv.to_string())
    best = P[P.period == "2010-22"].sort_values("pass_rate", ascending=False).groupby("sleeve").head(1).sort_values("pass_rate", ascending=False)
    print("\nbest dev pass rate per sleeve (R, pass, days):"); print(best[["sleeve", "R", "pass_rate", "days_to_pass"]].round(2).to_string(index=False))
    # (2) account diversification
    def fp(name, s, slots, R=9600.0):
        return FirmPlan(replace(TS, name=name, max_funded=slots, max_evals=1), [s], R, BAL, p_est=1.0, refill="all")
    cases = {
        "5 accounts, all the same NQ basket": [fp("basket", S["NQ basket (3 sleeves)"], 5)],
        "accounts split by sleeve: 2 noise-area / 1 VWAP / 2 gap-VWAP": [fp("noise", S["noise-area only"], 2), fp("vwap", S["VWAP+ATR only"], 1), fp("gap", S["gap-VWAP only"], 2)],
        "accounts split: 3 noise-area / 2 gap-VWAP": [fp("noise", S["noise-area only"], 3), fp("gap", S["gap-VWAP only"], 2)],
    }
    out = []
    for rule in ("one Combine at a time (your rule)", "each group buys its own Combine"):
        for lab, fps in cases.items():
            pol = Policy(reserve_cycles=None, global_one_eval=rule.startswith("one"))
            nets, firsts = [], []
            for y in range(2011, 2021):
                i0 = idx(f"{y}-01-01"); r = run_multifirm(fps, pol, i0, min(i0 + 756, n_dev)); nets.append(r["net"])
            nets = np.array(nets)
            seen = [round(run_multifirm(fps, pol, idx(x))["net"]) for x in ("2021-01-04", "2023-01-02", "2025-01-02", "2025-07-01")]
            full = run_multifirm(fps, pol, int(np.flatnonzero(ref.ok)[0]), log=True)
            ev = pd.DataFrame(full["events"], columns=["t", "g", "kind", "amt"]); ev["m"] = d[np.minimum(ev.t.to_numpy(), n - 1)].to_period("M")
            ev["sgn"] = np.where(ev.kind == "payout", ev.amt, np.where(ev.kind.isin(["activation", "eval_fee"]), -ev.amt, 0.0))
            mo = ev.pivot_table(index="m", columns="g", values="sgn", aggfunc="sum").reindex(pd.period_range("2012-01", "2026-06", freq="M")).fillna(0.0)
            pay_m = ev[ev.kind == "payout"].groupby("m").amt.sum().reindex(mo.index).fillna(0.0)
            cm = mo.corr().values[np.triu_indices(mo.shape[1], 1)] if mo.shape[1] > 1 else [np.nan]
            out.append(dict(rule=rule, case=lab, dev_mean=nets.mean(), dev_p25=np.percentile(nets, 25), dev_worst=nets.min(), green=int((nets > 0).sum()), seen=str(seen), full_net=full["net"], fees=full["fees"],
                            months_with_payout=float((pay_m > 0).mean()), group_corr=float(np.nanmean(cm))))
    O = pd.DataFrame(out); O.to_csv("results/account_diversification.csv", index=False)
    print("\nDIVERSIFYING STRATEGIES ACROSS FUNDED ACCOUNTS (balanced sizing, same rules and fees):"); print(O.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
