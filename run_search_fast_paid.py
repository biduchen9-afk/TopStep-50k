"""Pre-registration 14: search for > 80% of funded accounts paid AND a first payout within 90 calendar days (per-group constant sizes, first-payout balances)."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from portfolio import DEV_END
import run_kelly as K
import run_paid_share as P
import run_paid_share_d as D
from scaling_common import placebo_streams

pd.set_option("display.width", 270, "display.max_columns", 60, "display.max_colwidth", 60)
NOISE = list(itertools.product((0.6, 0.8, 1.0, 1.2, 1.6), (750.0, 1000.0, 1500.0, 2000.0)))
GAP = list(itertools.product((0.3, 0.4, 0.5, 0.6, 0.8, 1.2), (750.0, 1000.0, 1500.0, 2000.0)))
A2 = 0.8


def pack(df):
    f = df[df.passed]; p = f[f.paid]; early = p.start <= "2016-12-31"
    return dict(funded=len(f), paid=len(p), funded_e=int((f.start <= "2016-12-31").sum()), paid_e=int(early.sum()), funded_l=int((f.start > "2016-12-31").sum()), paid_l=int((~early).sum()),
                d=p.d_total.to_numpy(float), d_e=p[early].d_total.to_numpy(float), d_l=p[~early].d_total.to_numpy(float))


def mix(pn, pg, key="", wn=3, wg=2):
    k = lambda a: a + key
    fund = wn * pn[k("funded")] + wg * pg[k("funded")]; paid = wn * pn[k("paid")] + wg * pg[k("paid")]
    d = np.concatenate([pn[k("d")]] * wn + [pg[k("d")]] * wg)
    return (paid / fund if fund else np.nan), (float(np.median(d)) if len(d) else np.nan)


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}; ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt)))
    AC = {k: P.Acc(ST[k], n_dev) for k in ST}; AS = {k: P.Acc(ST[k], n, starts=range(idx("2023-01-01"), n - 300, 3)) for k in ST}
    dev = {("noise", a, t): pack(AC["noise"].run(a, t, A2)) for a, t in NOISE}; dev.update({("gap", a, t): pack(AC["gap"].run(a, t, A2)) for a, t in GAP})
    seen = {("noise", a, t): pack(AS["noise"].run(a, t, A2)) for a, t in NOISE}; seen.update({("gap", a, t): pack(AS["gap"].run(a, t, A2)) for a, t in GAP})
    rows = []
    for (na, nt), (ga, gt) in itertools.product(NOISE, GAP):
        pn, pg = dev[("noise", na, nt)], dev[("gap", ga, gt)]; paid, days = mix(pn, pg); pe, de = mix(pn, pg, "_e") if False else (None, None)
        fe = 3 * pn["funded_e"] + 2 * pg["funded_e"]; fl = 3 * pn["funded_l"] + 2 * pg["funded_l"]
        paid_e = (3 * pn["paid_e"] + 2 * pg["paid_e"]) / fe; paid_l = (3 * pn["paid_l"] + 2 * pg["paid_l"]) / fl
        days_e = float(np.median(np.concatenate([pn["d_e"]] * 3 + [pg["d_e"]] * 2))); days_l = float(np.median(np.concatenate([pn["d_l"]] * 3 + [pg["d_l"]] * 2)))
        sn, sg = seen[("noise", na, nt)], seen[("gap", ga, gt)]; paid_s, days_s = mix(sn, sg)
        rows.append(dict(noise_a1=na, noise_t1=nt, gap_a1=ga, gap_t1=gt, paid=paid, days=days, paid_noise=pn["paid"] / max(pn["funded"], 1), paid_gap=pg["paid"] / max(pg["funded"], 1), paid_early=paid_e, days_early=days_e, paid_late=paid_l, days_late=days_l, paid_2023=paid_s, days_2023=days_s))
    G = pd.DataFrame(rows)
    G["qual"] = (G.paid >= 0.80) & (G.days <= 90)
    G["stable"] = G.qual & (G.paid_early >= 0.75) & (G.paid_late >= 0.75) & (G.days_early <= 100) & (G.days_late <= 100) & (G.paid_2023 >= 0.70) & (G.days_2023 <= 100)
    print(f"combinations: {len(G)}; paid >= 80%: {(G.paid >= .8).sum()}; days <= 90: {(G.days <= 90).sum()}; both (qualifying): {G.qual.sum()}; qualifying and stable: {G.stable.sum()}")
    print("\nFRONTIER 1 - best paid share with days <= 90"); print(G[G.days <= 90].sort_values("paid", ascending=False).head(8).round(3).to_string(index=False))
    print("\nFRONTIER 2 - lowest days with paid >= 80%"); print(G[G.paid >= 0.8].sort_values("days").head(8).round(3).to_string(index=False))
    Q = G[G.qual].sort_values("paid", ascending=False).head(30).copy()
    out = []
    SPLIT_ST = ST
    for r in Q.itertuples():
        f = K.evaluate(D.plans_pg(SPLIT_ST, (r.noise_a1, r.noise_t1), (r.gap_a1, r.gap_t1)), ref, n_dev, n)
        out.append({**{c: getattr(r, c) for c in G.columns}, **{k: f[k] for k in ("dev_mean", "dev_p25", "dev_worst", "green", "paid_share", "pay_per_paid", "n_pay_per_paid", "days_first_payout", "seen", "full_net", "died_share")}, "nets": f["nets"]})
    F = pd.DataFrame(out)
    G.to_csv("results/search_fast_paid_all.csv", index=False)
    if len(F):
        F.drop(columns=["nets"]).to_csv("results/search_fast_paid_qualifying.csv", index=False)
        print("\nQUALIFYING (account level), farm-level metrics; baseline: paid 61.5%, 78 days, mean 17,486, p25 -1,639, worst -12,393")
        print(F.drop(columns=["nets", "seen", "full_net", "qual", "died_share"]).rename(columns={"paid_share": "farm_paid"}).round(3).sort_values("dev_mean", ascending=False).to_string(index=False))
        pool = F[F.stable] if F.stable.any() else F
        pick = pool.sort_values("dev_mean", ascending=False).iloc[0]
        print(f"\nCHOSEN ({'qualifying and stable' if F.stable.any() else 'qualifying but NOT stable - best mean among qualifiers'}): noise a1 {pick.noise_a1} t1 {pick.noise_t1:.0f} | gap a1 {pick.gap_a1} t1 {pick.gap_t1:.0f}")
        print(f"  paid {pick.paid:.1%} (noise {pick.paid_noise:.0%}, gap {pick.paid_gap:.0%}), days {pick.days:.0f}; early {pick.paid_early:.0%}/{pick.days_early:.0f}d late {pick.paid_late:.0%}/{pick.days_late:.0f}d; 2023+ {pick.paid_2023:.0%}/{pick.days_2023:.0f}d; farm mean {pick.dev_mean:,.0f} p25 {pick.dev_p25:,.0f} worst {pick.dev_worst:,.0f} green {pick.green}/10 died {pick.died_share:.0%} farm days {pick.days_first_payout:.0f} seen {pick.seen} full {pick.full_net:,.0f}")
        PS = placebo_streams(ST)
        for lab, (nz, gp) in (("baseline", ((1.2, 2000.0), (1.2, 2000.0))), ("chosen", ((pick.noise_a1, pick.noise_t1), (pick.gap_a1, pick.gap_t1)))):
            r = K.evaluate(D.plans_pg(PS, nz, gp), ref, n_dev, n)
            print(f"  placebo {lab:<9} dev mean {r['dev_mean']:8,.0f} p25 {r['dev_p25']:8,.0f} green {r['green']}/10 worst {r['dev_worst']:8,.0f} seen {r['seen']}")
    else:
        print("\nno combination qualifies")


if __name__ == "__main__":
    main()
