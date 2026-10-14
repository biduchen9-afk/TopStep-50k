"""Pre-registration 12, round 2 (declared after round 1 was seen): refine the UP ladder (start small, step up at balance milestones)."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from portfolio import DEV_END
from real_engine import Stream
import run_kelly as K
import run_paid_share as P

pd.set_option("display.width", 270, "display.max_columns", 60, "display.max_colwidth", 70)
T1, A2 = 2000.0, 0.8


def grid():
    cells = []
    for lo, hi, m in itertools.product((0.4, 0.5), (1.6, 2.0, 2.4), (750.0, 1000.0, 1250.0, 1500.0)):
        cells.append(dict(family="L1 two-step", desc=f"{lo} -> {hi} at ${m:,.0f}", a1=lo, scale=("ladder", ((m, hi),))))
    for a0, (m1, a_mid), (m2, a_hi) in (((0.4), (500.0, 1.2), (1000.0, 2.0)), (0.4, (750.0, 1.2), (1250.0, 2.0)), (0.4, (500.0, 1.0), (1000.0, 1.6)), (0.4, (500.0, 1.2), (1000.0, 1.6)),
                                         (0.5, (500.0, 1.0), (1250.0, 2.0)), (0.5, (750.0, 1.2), (1500.0, 2.4))):
        cells.append(dict(family="L2 three-step", desc=f"{a0} -> {a_mid} at ${m1:,.0f} -> {a_hi} at ${m2:,.0f}", a1=a0, scale=("ladder", ((m1, a_mid), (m2, a_hi)))))
    return cells


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}
    ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    AC = {k: P.Acc(ST[k], n_dev) for k in ST}; SPLIT = [("noise", 3), ("gap", 2)]

    def run_cell(c, streams=ST):
        parts = {g: AC[g].run(c["a1"], T1, A2, None, c["scale"]) for g in AC}
        acc = P.pooled([(parts["noise"], 3), (parts["gap"], 2)]); a = P.summ(acc)
        f = K.evaluate(P.plans_g(streams, SPLIT, c["a1"], T1, A2, None, c["scale"]), ref, n_dev, n)
        return acc, parts, a, f

    base_c = dict(family="baseline", desc="constant 1.2", a1=1.2, scale=None)
    acc0, parts0, a0, f0 = run_cell(base_c)
    base = dict(acct_paid=a0["paid_share"], dev_mean=f0["dev_mean"], dev_p25=f0["dev_p25"], dev_worst=f0["dev_worst"], nets=f0["nets"])
    print(f"baseline: paid {base['acct_paid']:.1%} mean {base['dev_mean']:,.0f} p25 {base['dev_p25']:,.0f} worst {base['dev_worst']:,.0f}", flush=True)
    out = []
    for c in grid():
        acc, parts, a, f = run_cell(c)
        nets = np.array(f["nets"])
        out.append(dict(family=c["family"], desc=c["desc"], acct_paid=a["paid_share"], paid_noise=P.summ(parts["noise"])["paid_share"], paid_gap=P.summ(parts["gap"])["paid_share"], acct_days=a["days_start_to_payout"],
                        paid_early=P.summ(acc[acc.start <= "2016-12-31"])["paid_share"], paid_late=P.summ(acc[acc.start > "2016-12-31"])["paid_share"],
                        half1=nets[:5].mean(), half2=nets[5:].mean(), **{k: f[k] for k in ("dev_mean", "dev_p25", "dev_worst", "green", "paid_share", "pay_per_paid", "n_pay_per_paid", "days_first_payout", "seen", "full_net")}))
        print(f"  done {c['family']:<14} {c['desc']:<46} acct paid {out[-1]['acct_paid']:.0%}  dev mean {out[-1]['dev_mean']:>8,.0f}  p25 {out[-1]['dev_p25']:>7,.0f}  days/acct {out[-1]['acct_days']:.0f}", flush=True)
    G = pd.DataFrame(out).rename(columns={"paid_share": "farm_paid"}); G["kept"] = G.dev_mean / base["dev_mean"]
    G["qual"] = (G.acct_paid >= 0.80) & (G.dev_mean >= 0.85 * base["dev_mean"]) & (G.dev_p25 >= base["dev_p25"] - 1000) & (G.dev_worst >= (base["dev_worst"] * 1.25 if base["dev_worst"] < 0 else base["dev_worst"] * 0.75))
    h1, h2 = float(np.mean(base["nets"][:5])), float(np.mean(base["nets"][5:]))
    G["halves_ok"] = (G.half1 > h1) & (G.half2 > h2) & (G.paid_early >= 0.75) & (G.paid_late >= 0.75)
    G.to_csv("results/scaling_size_grid2.csv", index=False)
    print(f"\nbaseline halves: farms 2011-15 {h1:,.0f}, 2016-20 {h2:,.0f}; paid early {P.summ(acc0[acc0.start <= '2016-12-31'])['paid_share']:.0%} / late {P.summ(acc0[acc0.start > '2016-12-31'])['paid_share']:.0%}")
    print("\nROUND 2 sorted by account-level paid share"); print(G.drop(columns=["seen", "family", "farm_paid", "full_net"]).sort_values("acct_paid", ascending=False).round(3).to_string(index=False))
    q = G[G.qual].sort_values("dev_mean", ascending=False)
    print("\nqualifying:", q[["desc", "acct_paid", "dev_mean", "dev_p25", "dev_worst", "acct_days", "days_first_payout", "halves_ok"]].round(3).to_dict("records") or "none")
    print("\nFRONTIER (best paid share at each profit retention):")
    for ret in (1.5, 1.0, 0.9, 0.85, 0.75, 0.6):
        sub = G[G.kept >= ret]
        if len(sub):
            b = sub.sort_values(["acct_paid", "dev_mean"], ascending=False).iloc[0]
            print(f"  retain >= {ret:.0%}: paid {b.acct_paid:.1%} [{b.desc}] mean {b.dev_mean:,.0f}, p25 {b.dev_p25:,.0f}, worst {b.dev_worst:,.0f}, days/acct {b.acct_days:.0f}, days/farm {b.days_first_payout:.0f}, seen {b.seen}")
    pick = q.iloc[0] if len(q) else G[G.kept >= 0.85].sort_values(["acct_paid", "dev_mean"], ascending=False).iloc[0]
    cell = [c for c in grid() if c["desc"] == pick.desc][0]
    print(f"\nCHOSEN/closest: [{pick.family}] {pick.desc}: paid noise {pick.paid_noise:.0%} gap {pick.paid_gap:.0%}, early {pick.paid_early:.0%} late {pick.paid_late:.0%}, farm halves {pick.half1:,.0f}/{pick.half2:,.0f} (baseline {h1:,.0f}/{h2:,.0f}), seen {pick.seen}, full {pick.full_net:,.0f}, halves_ok {pick.halves_ok}")
    dev = lambda s: (s.dates <= DEV_END) & s.ok
    PS = {}
    for g in ST:
        s = ST[g]; mu_R = float(np.mean(s.pnl[dev(s)] / s.atr_usd[dev(s)])); dm = np.where(s.ok, s.pnl - mu_R * np.nan_to_num(s.atr_usd), s.pnl)
        PS[g] = Stream(s.name, s.dates, dm, np.minimum(s.low - (s.pnl - dm), 0.0), s.atr_usd, s.ok)
    for lab, c in (("baseline", base_c), ("cell", cell)):
        r = K.evaluate(P.plans_g(PS, SPLIT, c["a1"], T1, A2, None, c["scale"]), ref, n_dev, n)
        print(f"  placebo {lab:<9} dev mean {r['dev_mean']:8,.0f} p25 {r['dev_p25']:8,.0f} green {r['green']}/10 worst {r['dev_worst']:8,.0f} seen {r['seen']}")


if __name__ == "__main__":
    main()
