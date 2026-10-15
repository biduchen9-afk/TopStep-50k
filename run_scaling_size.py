"""Pre-registration 12: scaling the funded risk size before the first payout (profit ladder, recent performance, time ramp). Topstep default split."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from portfolio import DEV_END
from real_engine import Stream
import run_kelly as K
import run_paid_share as P

pd.set_option("display.width", 270, "display.max_columns", 60, "display.max_colwidth", 60)
T1, A2 = 2000.0, 0.8


def grid():
    cells = [dict(family="baseline", desc="constant 1.2", a1=1.2, scale=None)]
    for lo, hi, m, d in itertools.product((0.4, 0.8), (1.2, 2.0), (500.0, 1000.0), ("up", "down")):
        if d == "up":
            cells.append(dict(family="A ladder", desc=f"up: {lo} -> {hi} at ${m:,.0f}", a1=lo, scale=("ladder", ((m, hi),))))
        else:
            cells.append(dict(family="A ladder", desc=f"down: {hi} -> {lo} at ${m:,.0f}", a1=hi, scale=("ladder", ((m, lo),))))
    for a1, L, (fu, fd) in itertools.product((0.8, 1.2), (5, 10), ((1.5, 0.5), (1.25, 0.75))):
        cells.append(dict(family="B performance", desc=f"base {a1}, last {L} days, x{fu} if up else x{fd}", a1=a1, scale=("perf", L, fu, fd)))
    for a0, a1, d in itertools.product((0.4, 0.6), (1.2, 1.6), (10, 20, 40)):
        cells.append(dict(family="C ramp", desc=f"{a0} -> {a1} over {d} sessions", a1=a1, scale=("ramp", d, a0)))
    return cells


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}
    ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    AC = {k: P.Acc(ST[k], n_dev) for k in ST}; SPLIT = [("noise", 3), ("gap", 2)]
    out = []
    for c in grid():
        parts = {g: AC[g].run(c["a1"], T1, A2, None, c["scale"]) for g in AC}
        acc = P.pooled([(parts["noise"], 3), (parts["gap"], 2)]); a = P.summ(acc)
        f = K.evaluate(P.plans_g(ST, SPLIT, c["a1"], T1, A2, None, c["scale"]), ref, n_dev, n)
        out.append(dict(family=c["family"], desc=c["desc"], acct_paid=a["paid_share"], paid_noise=P.summ(parts["noise"])["paid_share"], paid_gap=P.summ(parts["gap"])["paid_share"], acct_days=a["days_start_to_payout"],
                        first_payout=acc[acc.paid]["first"].mean(), paid_early=P.summ(acc[acc.start <= "2016-12-31"])["paid_share"], paid_late=P.summ(acc[acc.start > "2016-12-31"])["paid_share"],
                        **{k: f[k] for k in ("dev_mean", "dev_p25", "dev_worst", "green", "paid_share", "pay_per_paid", "n_pay_per_paid", "days_first_payout", "seen", "full_net")}))
        print(f"  done {c['family']:<14} {c['desc']:<44} acct paid {out[-1]['acct_paid']:.0%}  dev mean {out[-1]['dev_mean']:>8,.0f}", flush=True)
    G = pd.DataFrame(out).rename(columns={"paid_share": "farm_paid"}); base = G.iloc[0]
    G["kept"] = G.dev_mean / base.dev_mean
    G["qual"] = (G.acct_paid >= 0.80) & (G.dev_mean >= 0.85 * base.dev_mean) & (G.dev_p25 >= base.dev_p25 - 1000) & (G.dev_worst >= (base.dev_worst * 1.25 if base.dev_worst < 0 else base.dev_worst * 0.75))
    G.to_csv("results/scaling_size_grid.csv", index=False)
    print(f"\nbaseline: acct paid {base.acct_paid:.1%}, dev mean {base.dev_mean:,.0f}, p25 {base.dev_p25:,.0f}, worst {base.dev_worst:,.0f}, days/acct {base.acct_days:.0f}, days/farm {base.days_first_payout:.0f}")
    print("\nALL CELLS sorted by account-level paid share"); print(G.drop(columns=["seen", "paid_early", "paid_late", "family"]).sort_values("acct_paid", ascending=False).round(3).to_string(index=False))
    q = G[G.qual].sort_values("dev_mean", ascending=False)
    print("\nqualifying:", q[["desc", "acct_paid", "dev_mean", "dev_p25", "dev_worst", "acct_days", "days_first_payout"]].round(3).to_dict("records") or "none")
    print("\nFRONTIER (best account-level paid share at each profit retention):")
    for ret in (1.0, 0.9, 0.85, 0.75, 0.6):
        sub = G[G.kept >= ret]; b = sub.sort_values(["acct_paid", "dev_mean"], ascending=False).iloc[0]
        print(f"  retain >= {ret:.0%}: paid {b.acct_paid:.1%} [{b.desc}] dev mean {b.dev_mean:,.0f}, p25 {b.dev_p25:,.0f}, worst {b.dev_worst:,.0f}, days/acct {b.acct_days:.0f}, days/farm {b.days_first_payout:.0f}, seen {b.seen}")
    pick = q.iloc[0] if len(q) else G[G.kept >= 0.85].sort_values(["acct_paid", "dev_mean"], ascending=False).iloc[0]
    cell = [c for c in grid() if c["desc"] == pick.desc][0]
    print(f"\nCHOSEN/closest: [{pick.family}] {pick.desc}: paid noise {pick.paid_noise:.0%} gap {pick.paid_gap:.0%}, early {pick.paid_early:.0%} late {pick.paid_late:.0%}, seen {pick.seen}, full {pick.full_net:,.0f}")
    dev = lambda s: (s.dates <= DEV_END) & s.ok
    PS = {}
    for g in ST:
        s = ST[g]; mu_R = float(np.mean(s.pnl[dev(s)] / s.atr_usd[dev(s)])); dm = np.where(s.ok, s.pnl - mu_R * np.nan_to_num(s.atr_usd), s.pnl)
        PS[g] = Stream(s.name, s.dates, dm, np.minimum(s.low - (s.pnl - dm), 0.0), s.atr_usd, s.ok)
    for lab, c in (("baseline", grid()[0]), ("cell", cell)):
        r = K.evaluate(P.plans_g(PS, SPLIT, c["a1"], T1, A2, None, c["scale"]), ref, n_dev, n)
        print(f"  placebo {lab:<9} dev mean {r['dev_mean']:8,.0f} p25 {r['dev_p25']:8,.0f} green {r['green']}/10 worst {r['dev_worst']:8,.0f} seen {r['seen']}")


if __name__ == "__main__":
    main()
