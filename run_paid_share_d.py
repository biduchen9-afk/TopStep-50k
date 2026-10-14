"""Pre-registration 11, Part D (declared after Parts B and C were seen): different sizing for the noise-area group and the gap-VWAP group."""
from __future__ import annotations
import itertools
from dataclasses import replace
import numpy as np, pandas as pd
from multi_firm import FirmPlan
from portfolio import DEV_END
from real_engine import Stream
import run_kelly as K
import run_paid_share as P

pd.set_option("display.width", 260, "display.max_columns", 50)
NOISE_A1, NOISE_T1, GAP_A1 = (0.8, 1.0, 1.2), (1000.0, 2000.0), (0.4, 0.6)
GAP_T1, A2 = 2000.0, 0.8


def plans_pg(ST, noise, gap):
    """noise / gap = (a1, t1); a2 = 0.8, first-payout share 50% for both."""
    return [FirmPlan(replace(P.S["spec"], name="noise", max_funded=3, max_evals=1), [ST["noise"]], P.R, P.mk_plan(noise[0], noise[1], A2), p_est=1.0, refill="all"),
            FirmPlan(replace(P.S["spec"], name="gap", max_funded=2, max_evals=1), [ST["gap"]], P.R, P.mk_plan(gap[0], gap[1], A2), p_est=1.0, refill="all")]


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}
    ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    AC = {k: P.Acc(ST[k], n_dev) for k in ST}
    B = pd.read_csv("results/paid_share_grid.csv"); cur = B[(B.a1 == 1.2) & (B.t1 == 2000) & (B.a2 == 0.8)].iloc[0]
    out = []
    for na1, nt1, ga1 in itertools.product(NOISE_A1, NOISE_T1, GAP_A1):
        pn, pg = AC["noise"].run(na1, nt1, A2), AC["gap"].run(ga1, GAP_T1, A2)
        acc = P.pooled([(pn, 3), (pg, 2)]); a = P.summ(acc)
        f = K.evaluate(plans_pg(ST, (na1, nt1), (ga1, GAP_T1)), ref, n_dev, n)
        out.append(dict(noise_a1=na1, noise_t1=nt1, gap_a1=ga1, acct_paid=a["paid_share"], paid_noise=P.summ(pn)["paid_share"], paid_gap=P.summ(pg)["paid_share"], acct_days=a["days_start_to_payout"],
                        paid_early=P.summ(acc[acc.start <= "2016-12-31"])["paid_share"], paid_late=P.summ(acc[acc.start > "2016-12-31"])["paid_share"],
                        **{k: f[k] for k in ("dev_mean", "dev_p25", "dev_worst", "green", "paid_share", "pay_per_paid", "days_first_payout", "seen", "full_net")}))
        print(f"  done noise {na1}/{nt1:.0f} gap {ga1}: acct paid {out[-1]['acct_paid']:.0%}, dev mean {out[-1]['dev_mean']:,.0f}", flush=True)
    G = pd.DataFrame(out).rename(columns={"paid_share": "farm_paid"})
    G["qual"] = (G.acct_paid >= 0.80) & (G.dev_mean >= 0.85 * cur.dev_mean) & (G.dev_p25 >= cur.dev_p25 - 1000) & (G.dev_worst >= (cur.dev_worst * 1.25 if cur.dev_worst < 0 else cur.dev_worst * 0.75))
    G.to_csv("results/paid_share_grid_d.csv", index=False)
    print(f"\ncurrent: acct paid {cur.acct_paid:.0%} (noise {cur.paid_noise:.0%}, gap {cur.paid_gap:.0%}), dev mean {cur.dev_mean:,.0f}, p25 {cur.dev_p25:,.0f}, worst {cur.dev_worst:,.0f}, days/acct {cur.acct_days:.0f}, days/farm {cur.days_first_payout:.0f}, farm paid {cur.farm_paid:.0%}")
    print("\nPART D - sorted by account-level paid share"); print(G.drop(columns=["seen", "paid_early", "paid_late"]).sort_values("acct_paid", ascending=False).round(3).to_string(index=False))
    q = G[G.qual].sort_values(["dev_mean"], ascending=False)
    print("\nqualifying cells:", q[["noise_a1", "noise_t1", "gap_a1", "acct_paid", "paid_noise", "dev_mean", "dev_p25", "dev_worst", "acct_days", "days_first_payout", "farm_paid"]].round(3).to_dict("records") or "none")
    if len(q):
        b = q.iloc[0]
        print(f"\nCHOSEN: noise a1 {b.noise_a1} t1 {b.noise_t1:.0f}; gap a1 {b.gap_a1}; seen {b.seen} full {b.full_net:,.0f}; paid early/late {b.paid_early:.0%}/{b.paid_late:.0%}")
        dev = lambda s: (s.dates <= DEV_END) & s.ok
        PS = {}
        for g in ST:
            s = ST[g]; mu_R = float(np.mean(s.pnl[dev(s)] / s.atr_usd[dev(s)])); dm = np.where(s.ok, s.pnl - mu_R * np.nan_to_num(s.atr_usd), s.pnl)
            PS[g] = Stream(s.name, s.dates, dm, np.minimum(s.low - (s.pnl - dm), 0.0), s.atr_usd, s.ok)
        for lab, (nz, gp) in (("current", ((1.2, 2000.0), (1.2, 2000.0))), ("chosen", ((b.noise_a1, b.noise_t1), (b.gap_a1, GAP_T1)))):
            r = K.evaluate(plans_pg(PS, nz, gp), ref, n_dev, n)
            print(f"  placebo {lab:<8} dev mean {r['dev_mean']:8,.0f} p25 {r['dev_p25']:8,.0f} green {r['green']}/10 worst {r['dev_worst']:8,.0f} seen {r['seen']} full {r['full_net']:,.0f}")


if __name__ == "__main__":
    main()
