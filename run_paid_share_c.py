"""Pre-registration 11, Part C (declared after Part B was seen): a smaller first payout (lower balance trigger and/or smaller withdrawn share) as the lever for the paid share."""
from __future__ import annotations
import itertools, sys
import numpy as np, pandas as pd
from portfolio import DEV_END
from real_engine import Stream
import run_kelly as K
import run_paid_share as P

pd.set_option("display.width", 260, "display.max_columns", 50)
A1S, T1S, W1S, A2 = (0.8, 1.0, 1.2), (1000.0, 1500.0, 2000.0), (0.3, 0.5), 0.8


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}
    ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    AC = {k: P.Acc(ST[k], n_dev) for k in ST}; SPLIT = [("noise", 3), ("gap", 2)]
    B = pd.read_csv("results/paid_share_grid.csv"); cur = B[(B.a1 == 1.2) & (B.t1 == 2000) & (B.a2 == 0.8)].iloc[0]
    out = []
    for a1, t1, w1 in itertools.product(A1S, T1S, W1S):
        parts = {g: AC[g].run(a1, t1, A2, w1) for g in AC}
        acc = P.pooled([(parts["noise"], 3), (parts["gap"], 2)]); a = P.summ(acc)
        f = K.evaluate(P.plans_g(ST, SPLIT, a1, t1, A2, w1), ref, n_dev, n)
        out.append(dict(a1=a1, t1=t1, w1=w1, acct_paid=a["paid_share"], paid_noise=P.summ(parts["noise"])["paid_share"], paid_gap=P.summ(parts["gap"])["paid_share"], acct_days=a["days_start_to_payout"],
                        paid_early=P.summ(acc[acc.start <= "2016-12-31"])["paid_share"], paid_late=P.summ(acc[acc.start > "2016-12-31"])["paid_share"], first_payout=acc[acc.paid]["first"].mean(),
                        **{k: f[k] for k in ("dev_mean", "dev_p25", "dev_worst", "green", "paid_share", "pay_per_paid", "n_pay_per_paid", "days_first_payout", "seen", "full_net")}))
        print(f"  done a1={a1} t1={t1:.0f} w1={w1}: acct paid {out[-1]['acct_paid']:.0%}, dev mean {out[-1]['dev_mean']:,.0f}", flush=True)
    G = pd.DataFrame(out).rename(columns={"paid_share": "farm_paid"})
    G["qual"] = (G.acct_paid >= 0.80) & (G.dev_mean >= 0.85 * cur.dev_mean) & (G.dev_p25 >= cur.dev_p25 - 1000) & (G.dev_worst >= (cur.dev_worst * 1.25 if cur.dev_worst < 0 else cur.dev_worst * 0.75))
    G.to_csv("results/paid_share_grid_c.csv", index=False)
    print(f"\ncurrent (Part B): acct paid {cur.acct_paid:.0%}, dev mean {cur.dev_mean:,.0f}, p25 {cur.dev_p25:,.0f}, worst {cur.dev_worst:,.0f}, days/acct {cur.acct_days:.0f}, days/farm {cur.days_first_payout:.0f}")
    print("\nPART C - sorted by account-level paid share"); print(G.drop(columns=["seen", "paid_early", "paid_late"]).sort_values("acct_paid", ascending=False).round(3).to_string(index=False))
    print("\nqualifying cells:", G[G.qual][["a1", "t1", "w1", "acct_paid", "dev_mean", "dev_p25", "acct_days", "days_first_payout"]].round(3).to_dict("records") or "none")
    ALL = pd.concat([B.assign(w1=0.5)[["a1", "t1", "w1", "a2", "acct_paid", "dev_mean", "dev_p25", "dev_worst", "acct_days", "days_first_payout", "seen", "full_net"]], G.assign(a2=A2)[["a1", "t1", "w1", "a2", "acct_paid", "dev_mean", "dev_p25", "dev_worst", "acct_days", "days_first_payout", "seen", "full_net"]]]).drop_duplicates(["a1", "t1", "w1", "a2"])
    print("\nFRONTIER over all 48 cells (best account-level paid share at each profit retention level):")
    for ret in (1.0, 0.9, 0.85, 0.75, 0.6, 0.5):
        sub = ALL[ALL.dev_mean >= ret * cur.dev_mean]; b = sub.sort_values(["acct_paid", "dev_mean"], ascending=False).iloc[0]
        print(f"  retain >= {ret:.0%}: paid {b.acct_paid:.0%} at a1 {b.a1} t1 {b.t1:.0f} w1 {b.w1} a2 {b.a2} (dev mean {b.dev_mean:,.0f}, p25 {b.dev_p25:,.0f}, worst {b.dev_worst:,.0f}, days/acct {b.acct_days:.0f}, days/farm {b.days_first_payout:.0f}, seen {b.seen}, full {b.full_net:,.0f})")
    pick = G[G.qual].sort_values("dev_mean", ascending=False).iloc[0] if G.qual.any() else G.sort_values(["acct_paid", "dev_mean"], ascending=False).iloc[0]
    dev = lambda s: (s.dates <= DEV_END) & s.ok
    PS = {}
    for g in ST:
        s = ST[g]; mu_R = float(np.mean(s.pnl[dev(s)] / s.atr_usd[dev(s)])); dm = np.where(s.ok, s.pnl - mu_R * np.nan_to_num(s.atr_usd), s.pnl)
        PS[g] = Stream(s.name, s.dates, dm, np.minimum(s.low - (s.pnl - dm), 0.0), s.atr_usd, s.ok)
    print(f"\nplacebo (edge removed) for {'the chosen' if G.qual.any() else 'the highest-paid-share'} Part C cell a1={pick.a1} t1={pick.t1:.0f} w1={pick.w1}")
    for lab, (a1, t1, w1) in (("current", (1.2, 2000.0, 0.5)), ("cell", (pick.a1, pick.t1, pick.w1))):
        r = K.evaluate(P.plans_g(PS, SPLIT, a1, t1, A2, w1), ref, n_dev, n)
        print(f"  {lab:<8} dev mean {r['dev_mean']:8,.0f} p25 {r['dev_p25']:8,.0f} green {r['green']}/10 worst {r['dev_worst']:8,.0f} seen {r['seen']} full {r['full_net']:,.0f}")


if __name__ == "__main__":
    main()
