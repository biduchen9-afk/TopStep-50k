"""Pre-registration 10: hybrid funded sizing - volatility rule (1.2 x cushion / ATR) before the first payout, Kelly (f2 x full Kelly) after it. Topstep default split."""
from __future__ import annotations
import sys
from dataclasses import replace
import numpy as np, pandas as pd
from kelly_sizing import kelly_unit
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import Stream
from run_firm_comparison import mkplan
import run_kelly as K

pd.set_option("display.width", 250, "display.max_columns", 40)
A1, T1 = 1.2, 2000.0
F2S = (1.0, 2.0, 3.0)


def plans_h(ST, f2, placebo=None, window=120):
    fps = []
    for g, k in (("noise", 3), ("gap", 2)):
        s = ST[g]; pnl = s.pnl if placebo is None else placebo[g]
        fa2 = None if f2 is None else [kelly_unit(pnl, window=window, min_obs=min(60, window))]
        sp = replace(K.S["spec"], name=g, max_funded=k, max_evals=1)
        fps.append(FirmPlan(sp, [s], K.R, mkplan(K.S, T1, A1, 0.8 if f2 is None else f2), p_est=1.0, refill="all", funded_atr2=fa2))
    return fps


def accept(base, r):
    why = []
    if not r["dev_mean"] > base["dev_mean"]: why.append("lower mean")
    if not r["dev_p25"] > base["dev_p25"]: why.append("worse p25")
    if not r["dev_worst"] >= (base["dev_worst"] * 1.25 if base["dev_worst"] < 0 else base["dev_worst"] * 0.75): why.append("deeper worst farm")
    if not r["paid_share"] >= base["paid_share"] - 0.05: why.append("fewer accounts paid")
    if not r["days_first_payout"] <= 1.1 * base["days_first_payout"]: why.append("first payout >10% later")
    return why


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}
    ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    rows = [dict(config="current: ATR 1.2 / 0.8", f2=np.nan, **K.evaluate(plans_h(ST, None), ref, n_dev, n))]
    for f2 in F2S:
        rows.append(dict(config=f"hybrid: ATR 1.2, then Kelly {f2:g}x", f2=f2, **K.evaluate(plans_h(ST, f2), ref, n_dev, n)))
    H = pd.DataFrame(rows); base = rows[0]
    H["verdict"] = ["Current setting"] + [("Passes" if not (w := accept(base, r)) else "Rejected: " + ", ".join(w)) for r in rows[1:]]
    H.to_csv("results/hybrid_study.csv", index=False)
    print(H.drop(columns=["seen"]).round(2).to_string(index=False)); print("\nseen starts 2021 / 2023 / 2025-01 / 2025-07:"); print(H[["config", "seen", "full_net"]].to_string(index=False))
    ok = [r for r, v in zip(rows[1:], H.verdict[1:]) if v == "Passes"]
    print("\nPASS:", [r["config"] for r in ok] or "none")
    # descriptive checks
    dev = lambda s: (s.dates <= DEV_END) & s.ok
    PL, PS = {}, {}
    for g in ST:
        s = ST[g]; mu_R = float(np.mean(s.pnl[dev(s)] / s.atr_usd[dev(s)])); dm = np.where(s.ok, s.pnl - mu_R * np.nan_to_num(s.atr_usd), s.pnl)
        PS[g] = Stream(s.name, s.dates, dm, np.minimum(s.low - (s.pnl - dm), 0.0), s.atr_usd, s.ok); PL[g] = dm
    best_f2 = max(ok, key=lambda r: r["dev_p25"])["f2"] if ok else max(rows[1:], key=lambda r: r["dev_mean"])["f2"]
    print(f"\nplacebo (edge removed); cell shown = {'best passing' if ok else 'highest-mean (none passed)'} f2 = {best_f2:g}")
    for lab, f2 in (("current", None), (f"hybrid {best_f2:g}x", best_f2)):
        r = K.evaluate(plans_h(PS, f2, placebo=PL), ref, n_dev, n)
        print(f"  {lab:<12} dev mean {r['dev_mean']:8,.0f} p25 {r['dev_p25']:8,.0f} green {r['green']}/10 worst {r['dev_worst']:8,.0f} seen {r['seen']} full {r['full_net']:,.0f}")
    print(f"\nKelly window stability for f2 = {best_f2:g} (descriptive)")
    for w in (60, 120, 250):
        r = K.evaluate(plans_h(ST, best_f2, window=w), ref, n_dev, n)
        print(f"  window {w:>3}: dev mean {r['dev_mean']:8,.0f} p25 {r['dev_p25']:8,.0f} worst {r['dev_worst']:8,.0f} paid {r['paid_share']:.0%} days_first {r['days_first_payout']:.0f} seen {r['seen']} full {r['full_net']:,.0f}")


if __name__ == "__main__":
    main()
