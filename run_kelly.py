"""Pre-registration 9, Part C: Kelly-style funded sizing vs the current volatility rule, on the Topstep default split (3 noise-area + 2 gap-VWAP accounts, one Combine open at a time)."""
from __future__ import annotations
import sys
from dataclasses import replace
import numpy as np, pandas as pd
from firm_specs import specs
from kelly_sizing import kelly_unit
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import Stream
from risk_units import with_risk_unit
from run_firm_comparison import mkplan

sys.argv = sys.argv[:1]
from run_monthly_diversified import group_stream

pd.set_option("display.width", 250, "display.max_columns", 40)
S = specs()["Topstep 150K"]; R = 9600.0; T1 = 2000.0
ATR_CFG = {"ATR 1.2/0.8 (current)": (1.2, 0.8), "ATR 1.8/0.8": (1.8, 0.8), "ATR 1.2/1.2": (1.2, 1.2)}
KELLY_CFG = {f"Kelly {f1:g}x/{f2:g}x": (f1, f2) for f1 in (2, 3, 4) for f2 in (1, 2, 3)}


def plans(ST, a1, a2, kelly, placebo=None):
    fps = []
    for g, k in (("noise", 3), ("gap", 2)):
        s = ST[g]
        fa = [kelly_unit(s.pnl if placebo is None else placebo[g])] if kelly else None
        sp = replace(S["spec"], name=g, max_funded=k, max_evals=1)
        fps.append(FirmPlan(sp, [s], R, mkplan(S, T1, a1, a2), p_est=1.0, refill="all", funded_atr=fa))
    return fps


def evaluate(fps, ref, n_dev, n):
    idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt)))
    pol = Policy(reserve_cycles=None, global_one_eval=True)
    nets, nf, npaid, ndied, pay = [], 0, 0, 0, 0.0
    d1 = []
    for y in range(2011, 2021):
        i0 = idx(f"{y}-01-01"); r = run_multifirm(fps, pol, i0, min(i0 + 756, n_dev)); nets.append(r["net"])
        for v in r["firms"].values():
            nf += v["n_funded"]; npaid += v["n_paid_accts"]; ndied += v["died"]; pay += v["payouts"]
        fb = [v["first_buy_t"] for v in r["firms"].values() if v["first_buy_t"] is not None]; fp_ = [v["first_payout_t"] for v in r["firms"].values() if v["first_payout_t"] is not None]
        if fb and fp_:
            d1.append((pd.DatetimeIndex(ref.dates)[min(min(fp_), n - 1)] - pd.DatetimeIndex(ref.dates)[min(fb)]).days)
    nets = np.array(nets)
    seen = [round(run_multifirm(fps, pol, idx(x))["net"]) for x in ("2021-01-04", "2023-01-02", "2025-01-02", "2025-07-01")]
    full = run_multifirm(fps, pol, int(np.flatnonzero(ref.ok)[0]))
    return dict(dev_mean=nets.mean(), dev_p25=np.percentile(nets, 25), dev_worst=nets.min(), green=int((nets > 0).sum()), paid_share=npaid / max(nf, 1), died_share=ndied / max(nf, 1),
                pay_per_paid=pay / max(npaid, 1), days_first_payout=float(np.median(d1)) if d1 else np.nan, seen=str(seen), full_net=full["net"])


def main() -> None:
    gs = lambda names: with_risk_unit(group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}
    ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    rows = []
    for lab, (a1, a2) in ATR_CFG.items():
        rows.append(dict(config=lab, kind="atr", f1=a1, f2=a2, **evaluate(plans(ST, a1, a2, False), ref, n_dev, n)))
    for lab, (f1, f2) in KELLY_CFG.items():
        rows.append(dict(config=lab, kind="kelly", f1=f1, f2=f2, **evaluate(plans(ST, f1, f2, True), ref, n_dev, n)))
    K = pd.DataFrame(rows); K.to_csv("results/kelly_study.csv", index=False)
    print(K.round(2).to_string(index=False))
    base = K.iloc[0]
    ok = K[(K.dev_mean > base.dev_mean) & (K.dev_p25 > base.dev_p25) & (K.dev_worst >= base.dev_worst * 1.25 if base.dev_worst < 0 else K.dev_worst >= base.dev_worst * 0.75) & (K.paid_share >= base.paid_share - 0.05)]
    print("\nPASS the 'not greedy' acceptance vs the current rule:", list(ok.config) if len(ok) else "none")
    if len(ok):
        best = ok.sort_values("dev_p25", ascending=False).iloc[0]; print("chosen:", best.config)
        # placebo: edge removed (dev-fitted mean per ATR unit removed), same sizing rule
        dev = lambda s: (s.dates <= DEV_END) & s.ok
        PL = {}; PS = {}
        for g in ST:
            s = ST[g]; mu_R = float(np.mean(s.pnl[dev(s)] / s.atr_usd[dev(s)])); dm = np.where(s.ok, s.pnl - mu_R * np.nan_to_num(s.atr_usd), s.pnl)
            PS[g] = Stream(s.name, s.dates, dm, np.minimum(s.low - (s.pnl - dm), 0.0), s.atr_usd, s.ok); PL[g] = dm
        for lab, (kel, a1, a2) in {"current": (False, 1.2, 0.8), best.config: (True, best.f1, best.f2)}.items():
            fps = plans(PS, a1, a2, kel, placebo=PL)
            r = evaluate(fps, ref, n_dev, n); print(f"placebo (edge removed) {lab:<22} dev mean {r['dev_mean']:8,.0f}  green {r['green']}/10  worst {r['dev_worst']:8,.0f}  seen {r['seen']}")


if __name__ == "__main__":
    main()
