"""PR16 follow-up (exploratory): does picking the best (noise cell, gap cell) pair under a paid-share and days constraint generalise across dev halves?
Train on starts <= 2016 and test on starts > 2016, and the reverse. Uses the S1 grid cells only (216 per group), all constraints on the farm mix."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

import retest as T
from run_retest import A1S, A2S, BASE_CELL, RS, T1S

OUT = "results"
D = T.load(); ST, N_DEV = D["ST"], D["n_dev"]
bf = json.load(open(f"{OUT}/retest_base_farm.json")); ws, wf = bf["w_start"], bf["w_fund"]


def stats(df: pd.DataFrame) -> dict:
    p = df[df.passed]
    return dict(rate=df.ev.sum() / df.tau.sum(), tau=df.tau.mean(), paid=p.paid.mean() if len(p) else np.nan, w90=(p.d_total <= 90).mean() if len(p) else np.nan)


rows = []
for g in ST:
    for R in RS:
        ae = T.AcctEval(ST[g], N_DEV, R)
        for a1, t1, a2 in itertools.product(A1S, T1S, A2S):
            df = ae.run(a1, t1, a2)
            for half, m in (("h0", df.start <= "2016-12-31"), ("h1", df.start > "2016-12-31")):
                rows.append(dict(group=g, a1=a1, t1=t1, a2=a2, R=R, half=half, **stats(df[m])))
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/retest_cv_cells.csv", index=False)


def pair_arrays(half: str):
    N = C[(C.group == "noise") & (C.half == half)].reset_index(drop=True); G = C[(C.group == "gap") & (C.half == half)].reset_index(drop=True)
    tn, tg = N.tau.values[:, None], G.tau.values[None, :]
    tsn = ws["noise"] * tn / (ws["noise"] * tn + ws["gap"] * tg)
    rate = tsn * N.rate.values[:, None] + (1 - tsn) * G.rate.values[None, :]
    paid = wf["noise"] * N.paid.values[:, None] + wf["gap"] * G.paid.values[None, :]
    w90 = wf["noise"] * N.w90.values[:, None] + wf["gap"] * G.w90.values[None, :]
    return N, G, rate, paid, w90 / paid


def base_ij(N, G):
    f = lambda D: D[(D.a1 == BASE_CELL[0]) & (D.t1 == BASE_CELL[1]) & (D.a2 == BASE_CELL[2]) & (D.R == BASE_CELL[3])].index[0]
    return f(N), f(G)


out = []
for train, test in (("h0", "h1"), ("h1", "h0")):
    Nt, Gt, rt, pt, qt = pair_arrays(train); Ns, Gs, rs, ps, qs = pair_arrays(test)
    bi, bj = base_ij(Nt, Gt)
    print(f"\ntrain {train} -> test {test}: baseline test rate {rs[bi, bj]:.1f}, paid {ps[bi, bj]:.1%}, paid-within-90 share {qs[bi, bj]:.1%}")
    for lab, mask in (("unconstrained", np.ones_like(rt, bool)), ("paid >= 70% & median days <= 90", (pt >= 0.70) & (qt >= 0.5)), ("paid >= 60% & median days <= 90", (pt >= 0.60) & (qt >= 0.5)), ("paid >= 80%", pt >= 0.80)):
        m = np.where(mask, rt, -np.inf)
        if not np.isfinite(m).any():
            print(f"  {lab:<36} no pair qualifies on the train half"); continue
        i, j = np.unravel_index(np.argmax(m), m.shape)
        r = dict(train=train, test=test, rule=lab, train_rate=float(rt[i, j]), test_rate=float(rs[i, j]), base_test_rate=float(rs[bi, bj]), test_gain=float(rs[i, j] - rs[bi, bj]), test_paid=float(ps[i, j]), base_test_paid=float(ps[bi, bj]),
                 test_paid_in90=float(qs[i, j]), base_test_paid_in90=float(qs[bi, bj]), noise=f"{Nt.a1[i]}/{Nt.t1[i]:.0f}/{Nt.a2[i]}/{Nt.R[i]:.0f}", gap=f"{Gt.a1[j]}/{Gt.t1[j]:.0f}/{Gt.a2[j]}/{Gt.R[j]:.0f}")
        out.append(r)
        print(f"  {lab:<36} train rate {r['train_rate']:6.1f} -> TEST rate {r['test_rate']:6.1f} (baseline {r['base_test_rate']:.1f}, gain {r['test_gain']:+.1f}); test paid {r['test_paid']:.1%} (base {r['base_test_paid']:.1%}); paid-in-90 {r['test_paid_in90']:.0%} (base {r['base_test_paid_in90']:.0%}); noise {r['noise']} gap {r['gap']}")
pd.DataFrame(out).to_csv(f"{OUT}/retest_cv.csv", index=False)
