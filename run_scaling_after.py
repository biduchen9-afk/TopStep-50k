"""Pre-registration 13: scaling the size AFTER the first payout (profit ladder, recent performance, time ramp, number of payouts). Topstep default split; Combine and pre-payout stage unchanged."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from portfolio import DEV_END
import run_kelly as K
import run_paid_share as P
from scaling_common import accept, halves, placebo_streams

pd.set_option("display.width", 270, "display.max_columns", 60, "display.max_colwidth", 60)


def grid():
    cells = [dict(family="baseline", desc="constant a2 = 0.8", a2=0.8, scale2=None)]
    for base, top, m in itertools.product((0.4, 0.8), (1.2, 1.6, 2.4), (2000.0, 3500.0)):
        cells.append(dict(family="A ladder", desc=f"up: {base} -> {top} at ${m:,.0f}", a2=base, scale2=("ladder", ((m, top),))))
    for base, m in itertools.product((1.2, 1.6), (2000.0, 3500.0)):
        cells.append(dict(family="A ladder", desc=f"down: {base} -> 0.4 at ${m:,.0f}", a2=base, scale2=("ladder", ((m, 0.4),))))
    for base, L, (fu, fd) in itertools.product((0.8, 1.2), (5, 10), ((1.5, 0.5), (1.25, 0.75))):
        cells.append(dict(family="B performance", desc=f"base {base}, last {L} days: x{fu} if up else x{fd}", a2=base, scale2=("perf", L, fu, fd)))
    for end, d in itertools.product((0.8, 1.2), (10, 20)):
        cells.append(dict(family="C ramp", desc=f"after each payout 0.4 -> {end} over {d} sessions", a2=end, scale2=("ramp", d, 0.4)))
    for d in (10, 20):
        cells.append(dict(family="C ramp", desc=f"after each payout 1.6 -> 0.8 over {d} sessions", a2=0.8, scale2=("ramp", d, 1.6)))
    for desc, steps in (("after 2 payouts 1.2, after 4 payouts 1.6", ((2, 1.2), (4, 1.6))), ("after 2 payouts 0.6, after 4 payouts 0.4", ((2, 0.6), (4, 0.4))), ("after 3 payouts 1.2", ((3, 1.2),)), ("after 3 payouts 0.5", ((3, 0.5),))):
        cells.append(dict(family="D payouts", desc=desc, a2=0.8, scale2=("npay", steps)))
    for a in (0.4, 0.6, 1.0):
        cells.append(dict(family="E constant", desc=f"constant a2 = {a}", a2=a, scale2=None))
    return cells


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}
    ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right")); SPLIT = [("noise", 3), ("gap", 2)]
    out = []
    for c in grid():
        f = K.evaluate(P.plans_g(ST, SPLIT, 1.2, 2000.0, c["a2"], None, None, c["scale2"]), ref, n_dev, n)
        out.append(dict(family=c["family"], desc=c["desc"], half1=halves(f["nets"])[0], half2=halves(f["nets"])[1], nets=f["nets"], died_share=f["died_share"],
                        **{k: f[k] for k in ("dev_mean", "dev_p25", "dev_worst", "green", "paid_share", "pay_per_paid", "n_pay_per_paid", "days_first_payout", "seen", "full_net")}))
        print(f"  done {c['family']:<14} {c['desc']:<52} dev mean {out[-1]['dev_mean']:>8,.0f}  p25 {out[-1]['dev_p25']:>7,.0f}  worst {out[-1]['dev_worst']:>8,.0f}  pay/paid {out[-1]['pay_per_paid']:>6,.0f}  n/paid {out[-1]['n_pay_per_paid']:.2f}", flush=True)
    G = pd.DataFrame(out); base = out[0]
    G["verdict"] = ["Baseline"] + [("Passes" if not (w := accept(base, r)) else "Rejected: " + ", ".join(w)) for r in out[1:]]
    G.drop(columns=["nets"]).to_csv("results/scaling_after_grid.csv", index=False)
    print(f"\nbaseline: dev mean {base['dev_mean']:,.0f}, p25 {base['dev_p25']:,.0f}, worst {base['dev_worst']:,.0f}, paid {base['paid_share']:.1%}, pay/paid {base['pay_per_paid']:,.0f}, n/paid {base['n_pay_per_paid']:.2f}, halves {halves(base['nets'])[0]:,.0f}/{halves(base['nets'])[1]:,.0f}")
    print("\nALL CELLS sorted by dev mean"); print(G.drop(columns=["nets", "seen", "family", "full_net", "half1", "half2"]).sort_values("dev_mean", ascending=False).round(3).to_string(index=False))
    ok = G[G.verdict == "Passes"].sort_values(["dev_p25", "dev_mean"], ascending=False)
    print("\nPASSING:", ok[["desc", "dev_mean", "dev_p25", "dev_worst", "pay_per_paid"]].round(3).to_dict("records") or "none")
    better = G.iloc[1:][G.iloc[1:].dev_mean > base["dev_mean"]]
    pick = ok.iloc[0] if len(ok) else (better.sort_values("dev_p25", ascending=False).iloc[0] if len(better) else G.iloc[1:].sort_values("dev_mean", ascending=False).iloc[0])
    cell = [c for c in grid() if c["desc"] == pick.desc][0]
    print(f"\nCHOSEN/closest: [{pick.family}] {pick.desc}: mean {pick.dev_mean:,.0f} p25 {pick.dev_p25:,.0f} worst {pick.dev_worst:,.0f} halves {pick.half1:,.0f}/{pick.half2:,.0f} seen {pick.seen} full {pick.full_net:,.0f} -> {pick.verdict}")
    PS = placebo_streams(ST)
    for lab, c in (("baseline", grid()[0]), ("cell", cell)):
        r = K.evaluate(P.plans_g(PS, SPLIT, 1.2, 2000.0, c["a2"], None, None, c["scale2"]), ref, n_dev, n)
        print(f"  placebo {lab:<9} dev mean {r['dev_mean']:8,.0f} p25 {r['dev_p25']:8,.0f} green {r['green']}/10 worst {r['dev_worst']:8,.0f} seen {r['seen']}")


if __name__ == "__main__":
    main()
