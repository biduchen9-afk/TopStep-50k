"""Pre-registration 13: scaling the contracts in the Combine (profit ladder, recent performance, time ramp). Topstep default split; funded stage unchanged."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from firm_engine import run_eval_firm
from portfolio import DEV_END
import run_kelly as K
import run_paid_share as P
from scaling_common import accept, halves, placebo_streams

pd.set_option("display.width", 270, "display.max_columns", 60, "display.max_colwidth", 60)


def grid():
    cells = [dict(family="baseline", desc="constant R = 9,600", scale=None)]
    for (dirn, fs), m in itertools.product((("up", (1.5, 2.0)), ("down", (0.5, 0.25))), (3000.0, 6000.0)):
        for f in fs:
            cells.append(dict(family="A ladder", desc=f"{dirn}: x{f} at +${m:,.0f}", scale=("ladder", ((m, f),))))
    for L, (fu, fd) in itertools.product((3, 5, 10), ((1.5, 0.5), (1.25, 0.75))):
        cells.append(dict(family="B performance", desc=f"last {L} days: x{fu} if up else x{fd}", scale=("perf", L, fu, fd)))
    for f0, d in itertools.product((0.5, 1.5), (5, 10, 20)):
        cells.append(dict(family="C ramp", desc=f"x{f0} -> x1 over {d} sessions", scale=("ramp", d, f0)))
    for f in (0.75, 1.25):
        cells.append(dict(family="D constant", desc=f"constant x{f} (R = {9600 * f:,.0f})", scale=("const", f)))
    return cells


class CombineAcc:
    def __init__(self, s, n_dev):
        self.spec = P.S["spec"]; self.x, self.lo = s.pnl[:n_dev].tolist(), s.low[:n_dev].tolist(); self.nd = s.desired_micros(P.R)[:n_dev]; self.d = pd.DatetimeIndex(s.dates)
        first = int(np.flatnonzero(s.ok)[0]) + 260; self.starts = list(range(first, n_dev - 400, 3))

    def run(self, scale):
        rows = []
        for i0 in self.starts:
            st, tau = run_eval_firm(self.spec.eval_phases, self.x, self.lo, self.nd, i0, scale)
            bills = 1 + (max(tau, 1) - 1) // self.spec.cycle
            rows.append(dict(passed=st == 1, days=(self.d[min(i0 + tau, len(self.d) - 1)] - self.d[i0]).days if st == 1 else np.nan, fee=bills * self.spec.eval_fee + (self.spec.activation_fee if st == 1 else 0.0)))
        return pd.DataFrame(rows)


def main() -> None:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}
    ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    CA = {k: CombineAcc(ST[k], n_dev) for k in ST}; SPLIT = [("noise", 3), ("gap", 2)]
    out = []
    for c in grid():
        pooled = pd.concat([CA["noise"].run(c["scale"])] * 3 + [CA["gap"].run(c["scale"])] * 2, ignore_index=True)
        f = K.evaluate(P.plans_g(ST, SPLIT, 1.2, 2000.0, 0.8, None, None, None, c["scale"]), ref, n_dev, n)
        out.append(dict(family=c["family"], desc=c["desc"], pass_rate=pooled.passed.mean(), days_to_pass=pooled[pooled.passed].days.median(), fees_per_funded=pooled.fee.sum() / max(int(pooled.passed.sum()), 1),
                        half1=halves(f["nets"])[0], half2=halves(f["nets"])[1], nets=f["nets"], **{k: f[k] for k in ("dev_mean", "dev_p25", "dev_worst", "green", "paid_share", "pay_per_paid", "n_pay_per_paid", "days_first_payout", "seen", "full_net")}))
        print(f"  done {c['family']:<14} {c['desc']:<38} pass {out[-1]['pass_rate']:.1%}  days {out[-1]['days_to_pass']:.0f}  fees/funded {out[-1]['fees_per_funded']:,.0f}  dev mean {out[-1]['dev_mean']:>8,.0f}  p25 {out[-1]['dev_p25']:>7,.0f}", flush=True)
    G = pd.DataFrame(out); base = out[0]
    G["verdict"] = ["Baseline"] + [("Passes" if not (w := accept(base, r, True)) else "Rejected: " + ", ".join(w)) for r in out[1:]]
    G.drop(columns=["nets"]).to_csv("results/scaling_combine_grid.csv", index=False)
    print(f"\nbaseline: pass {base['pass_rate']:.1%}, days {base['days_to_pass']:.0f}, fees/funded {base['fees_per_funded']:,.0f}, dev mean {base['dev_mean']:,.0f}, p25 {base['dev_p25']:,.0f}, worst {base['dev_worst']:,.0f}, paid {base['paid_share']:.1%}, days/farm {base['days_first_payout']:.0f}, halves {halves(base['nets'])[0]:,.0f}/{halves(base['nets'])[1]:,.0f}")
    print("\nALL CELLS sorted by dev mean"); print(G.drop(columns=["nets", "seen", "family", "full_net", "half1", "half2"]).sort_values("dev_mean", ascending=False).round(3).to_string(index=False))
    ok = G[G.verdict == "Passes"].sort_values(["dev_p25", "dev_mean"], ascending=False)
    print("\nPASSING:", ok[["desc", "pass_rate", "dev_mean", "dev_p25", "dev_worst"]].round(3).to_dict("records") or "none")
    pick = ok.iloc[0] if len(ok) else G.iloc[1:][(G.iloc[1:].dev_mean > base["dev_mean"])].sort_values("dev_p25", ascending=False).iloc[0] if (G.iloc[1:].dev_mean > base["dev_mean"]).any() else G.iloc[1:].sort_values("dev_mean", ascending=False).iloc[0]
    cell = [c for c in grid() if c["desc"] == pick.desc][0]
    print(f"\nCHOSEN/closest: [{pick.family}] {pick.desc}: pass {pick.pass_rate:.1%} days {pick.days_to_pass:.0f} fees/funded {pick.fees_per_funded:,.0f} mean {pick.dev_mean:,.0f} p25 {pick.dev_p25:,.0f} worst {pick.dev_worst:,.0f} halves {pick.half1:,.0f}/{pick.half2:,.0f} seen {pick.seen} -> {pick.verdict}")
    PS = placebo_streams(ST)
    for lab, c in (("baseline", grid()[0]), ("cell", cell)):
        r = K.evaluate(P.plans_g(PS, SPLIT, 1.2, 2000.0, 0.8, None, None, None, c["scale"]), ref, n_dev, n)
        print(f"  placebo {lab:<9} dev mean {r['dev_mean']:8,.0f} p25 {r['dev_p25']:8,.0f} green {r['green']}/10 worst {r['dev_worst']:8,.0f} seen {r['seen']}")


if __name__ == "__main__":
    main()
