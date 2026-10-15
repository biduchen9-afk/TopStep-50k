"""Pre-registration 11: why fewer funded accounts are paid, and whether > 80% is reachable without losing much profit. Topstep default split."""
from __future__ import annotations
import itertools, sys
from dataclasses import replace
import numpy as np, pandas as pd
from firm_engine import run_eval_firm, run_funded_firm
from multi_firm import FirmPlan
from portfolio import DEV_END
from real_engine import FundedPlan, Stream
from run_firm_comparison import mkplan
import run_kelly as K

pd.set_option("display.width", 260, "display.max_columns", 50)
S = K.S; R = K.R
A1S, T1S, A2S = (0.4, 0.6, 0.8, 1.0, 1.2), (2000.0, 3000.0, 4000.0), (0.8, 1.2)
CUR = (1.2, 2000.0, 0.8)


def mk_plan(a1, t1, a2, w1=None, scale=None, scale2=None):
    """Same as run_firm_comparison.mkplan, with an optional first-payout share w1 (default: the firm's maximum, 50%)."""
    sp = S["spec"].funded; frac = sp.payout_frac
    return FundedPlan("p", "x", t1, frac if w1 is None else w1, 1.0, 0.0, frac, 1.0, S["max_payouts"], a1, a2, scale, scale2)


class Acc:
    """Combine outcomes of one sleeve on sequential dev starts, computed once; funded runs per plan."""
    def __init__(self, s, n_dev, starts=None):
        spec = S["spec"]; self.s, self.spec, self.n_dev = s, spec, n_dev
        self.x, self.lo, self.atr = s.pnl[:n_dev].tolist(), s.low[:n_dev].tolist(), s.atr_usd[:n_dev].tolist(); self.nd = s.desired_micros(R)[:n_dev]
        first = int(np.flatnonzero(s.ok)[0]) + 260; self.starts = list(range(first, n_dev - 400, 3)) if starts is None else list(starts); self.d = pd.DatetimeIndex(s.dates)
        self.ev = [(i0,) + run_eval_firm(spec.eval_phases, self.x, self.lo, self.nd, i0) for i0 in self.starts]

    def run(self, a1, t1, a2, w1=None, scale=None):
        plan = mk_plan(a1, t1, a2, w1, scale); rows = []
        for i0, st, tau in self.ev:
            r = dict(start=self.d[i0], passed=st == 1, paid=False, d_total=np.nan, first=np.nan, total=0.0)
            if st == 1:
                act = min(i0 + tau + 1, self.n_dev - 1); tm = []
                pays, *_ = run_funded_firm(self.spec.funded, self.x, self.lo, self.nd, act, plan, self.atr, tm)
                if pays:
                    r.update(paid=True, d_total=(self.d[min(tm[0], self.n_dev - 1)] - self.d[i0]).days, first=pays[0], total=sum(pays))
            rows.append(r)
        return pd.DataFrame(rows)


def pooled(parts):
    """parts = [(DataFrame, weight)]: replicate by weight, as one pool of accounts."""
    return pd.concat([df for df, w in parts for _ in range(w)], ignore_index=True)


def summ(df):
    p = df[df.passed]
    return dict(pass_rate=df.passed.mean(), funded=int(len(p)), paid_share=p.paid.mean() if len(p) else np.nan, days_start_to_payout=p[p.paid].d_total.median(), paid_per_funded=p.total.mean() if len(p) else np.nan)


def plans_g(ST, groups, a1, t1, a2, w1=None, scale=None, scale2=None, eval_scale=None):
    return [FirmPlan(replace(S["spec"], name=g, max_funded=k, max_evals=1), [ST[g]], R, mk_plan(a1, t1, a2, w1, scale, scale2), p_est=1.0, refill="all", eval_scale=eval_scale) for g, k in groups]


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"]), "basket": gs(["nq_noise", "nq_vwap", "nq_gap30"])}
    ref = ST["basket"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    AC = {k: Acc(ST[k], n_dev) for k in ST}
    SPLIT = [("noise", 3), ("gap", 2)]
    # ------------------------------------------------ Part A: decomposition
    rows = []
    cases = [("old safe: basket x5, 0.4 / $4,000", [("basket", 5)], (0.4, 4000.0, 0.8)), ("basket x5, balanced 1.2 / $2,000", [("basket", 5)], CUR), ("default split, balanced 1.2 / $2,000", SPLIT, CUR),
             ("noise-area x5 alone, balanced", [("noise", 5)], CUR), ("gap-VWAP x5 alone, balanced", [("gap", 5)], CUR), ("default split, safe 0.4 / $4,000", SPLIT, (0.4, 4000.0, 0.8))]
    for lab, groups, (a1, t1, a2) in cases:
        acc = pooled([(AC[g].run(a1, t1, a2), k) for g, k in groups])
        f = K.evaluate(plans_g(ST, groups, a1, t1, a2), ref, n_dev, n)
        rows.append(dict(case=lab, acct_paid_share=summ(acc)["paid_share"], acct_pass_rate=summ(acc)["pass_rate"], acct_days_to_payout=summ(acc)["days_start_to_payout"], farm_paid_share=f["paid_share"], farm_dev_mean=f["dev_mean"], farm_p25=f["dev_p25"], farm_days_first_payout=f["days_first_payout"]))
    D = pd.DataFrame(rows); D.to_csv("results/paid_share_decomposition.csv", index=False); print("PART A - where the drop came from"); print(D.round(3).to_string(index=False), flush=True)
    # ------------------------------------------------ Part B: grid
    out = []
    for a1, t1, a2 in itertools.product(A1S, T1S, A2S):
        parts = {g: AC[g].run(a1, t1, a2) for g in ("noise", "gap")}
        acc = pooled([(parts["noise"], 3), (parts["gap"], 2)]); a = summ(acc)
        f = K.evaluate(plans_g(ST, SPLIT, a1, t1, a2), ref, n_dev, n)
        out.append(dict(a1=a1, t1=t1, a2=a2, acct_pass=a["pass_rate"], acct_paid=a["paid_share"], paid_noise=summ(parts["noise"])["paid_share"], paid_gap=summ(parts["gap"])["paid_share"], acct_days=a["days_start_to_payout"],
                        paid_early=summ(acc[acc.start <= "2016-12-31"])["paid_share"], paid_late=summ(acc[acc.start > "2016-12-31"])["paid_share"], **{k: f[k] for k in ("dev_mean", "dev_p25", "dev_worst", "green", "paid_share", "pay_per_paid", "days_first_payout", "seen", "full_net")}))
        print(f"  done a1={a1} t1={t1:.0f} a2={a2}: acct paid {out[-1]['acct_paid']:.0%}, dev mean {out[-1]['dev_mean']:,.0f}", flush=True)
    G = pd.DataFrame(out).rename(columns={"paid_share": "farm_paid"}); G.to_csv("results/paid_share_grid.csv", index=False)
    cur = G[(G.a1 == CUR[0]) & (G.t1 == CUR[1]) & (G.a2 == CUR[2])].iloc[0]
    G["qual"] = (G.acct_paid >= 0.80) & (G.dev_mean >= 0.85 * cur.dev_mean) & (G.dev_p25 >= cur.dev_p25 - 1000) & (G.dev_worst >= (cur.dev_worst * 1.25 if cur.dev_worst < 0 else cur.dev_worst * 0.75))
    show = G.drop(columns=["seen", "paid_early", "paid_late"]).sort_values("acct_paid", ascending=False)
    print("\nPART B - grid sorted by account-level paid share"); print(show.round(3).to_string(index=False))
    print("\nqualifying cells:", G[G.qual][["a1", "t1", "a2", "acct_paid", "dev_mean", "dev_p25", "acct_days", "days_first_payout"]].round(3).to_dict("records") or "none")
    print("\nFRONTIER (best account-level paid share at each profit-retention level; retention = dev mean / current dev mean, p25 and worst unconstrained here):")
    for ret in (1.0, 0.9, 0.85, 0.75):
        sub = G[G.dev_mean >= ret * cur.dev_mean]; b = sub.sort_values(["acct_paid", "dev_mean"], ascending=False).iloc[0]
        print(f"  retain >= {ret:.0%}: paid {b.acct_paid:.0%} at a1 {b.a1} t1 {b.t1:.0f} a2 {b.a2} (dev mean {b.dev_mean:,.0f}, p25 {b.dev_p25:,.0f}, worst {b.dev_worst:,.0f}, days/acct {b.acct_days:.0f}, days/farm {b.days_first_payout:.0f}, seen {b.seen}, full {b.full_net:,.0f})")
    G.to_csv("results/paid_share_grid.csv", index=False)
    # ------------------------------------------------ descriptive: placebo for the chosen / closest cell
    dev = lambda s: (s.dates <= DEV_END) & s.ok
    PS = {}
    for g in ("noise", "gap"):
        s = ST[g]; mu_R = float(np.mean(s.pnl[dev(s)] / s.atr_usd[dev(s)])); dm = np.where(s.ok, s.pnl - mu_R * np.nan_to_num(s.atr_usd), s.pnl)
        PS[g] = Stream(s.name, s.dates, dm, np.minimum(s.low - (s.pnl - dm), 0.0), s.atr_usd, s.ok)
    pick = G[G.qual].sort_values("dev_mean", ascending=False).iloc[0] if G.qual.any() else G[G.acct_paid >= 0.80].sort_values("dev_mean", ascending=False).iloc[0] if (G.acct_paid >= 0.80).any() else G.sort_values("acct_paid", ascending=False).iloc[0]
    print(f"\nplacebo (edge removed) for the {'chosen' if G.qual.any() else 'closest (highest-mean cell with the highest paid share available)'} cell a1={pick.a1} t1={pick.t1:.0f} a2={pick.a2}")
    for lab, (a1, t1, a2) in (("current", CUR), ("cell", (pick.a1, pick.t1, pick.a2))):
        r = K.evaluate(plans_g(PS, SPLIT, a1, t1, a2), ref, n_dev, n)
        print(f"  {lab:<8} dev mean {r['dev_mean']:8,.0f} p25 {r['dev_p25']:8,.0f} green {r['green']}/10 worst {r['dev_worst']:8,.0f} seen {r['seen']} full {r['full_net']:,.0f}")


if __name__ == "__main__":
    main()
