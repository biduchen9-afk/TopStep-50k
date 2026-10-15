"""Time to pass and time to payout, shown separately (calendar days), final config. Starts every 3 sessions. Descriptive (includes seen data); 'dev' = starts <= 2022."""
import numpy as np, pandas as pd
from final_strategy import FINAL, PATH, build_stream
from pricing_config import load_config
from real_engine import FundedPlan, rules_from_config, run_combine, run_funded_detail
s = build_stream(); cfg = load_config(); rules = rules_from_config(cfg, PATH); d = pd.DatetimeIndex(s.dates)
x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist(); nd = s.desired_micros(FINAL["R"])
pl = FundedPlan("p", "standard", FINAL["t1"], .5, 1., 0., .5, 1., 99, FINAL["a1"], FINAL["a2"])
first = int(np.flatnonzero(s.ok)[0]) + 260; rows = []
for i0 in range(first, len(d) - 300, 3):
    st, tau = run_combine(x, lo, nd, i0, rules); r = dict(y=d[i0].year, passed=int(st == 1), days_to_pass=np.nan, days_act_to_payout=np.nan, paid=0)
    if st == 1:
        act = min(i0 + tau + 1, len(d) - 1)                                   # activation the session after the pass
        r["days_to_pass"] = (d[act] - d[i0]).days
        tm = []; pays, *_ = run_funded_detail(x, lo, nd, i0 + tau + 1, rules, pl, atr, tm)
        if tm: r["paid"] = 1; r["days_act_to_payout"] = (d[min(tm[0], len(d) - 1)] - d[act]).days
    rows.append(r)
df = pd.DataFrame(rows); f = df[df.passed == 1]
def show(lab, a, b):
    sub = df[(df.y >= a) & (df.y <= b)]; fs = sub[sub.passed == 1]; pd_ = fs[fs.paid == 1]
    q = lambda v: ("n/a" if len(v) == 0 else f"median {v.median():.0f} (25-75%: {v.quantile(.25):.0f}-{v.quantile(.75):.0f})")
    print(f"{lab:<11} starts {len(sub):4d} | Combine pass rate {sub.passed.mean():.0%} | days start->pass: {q(fs.days_to_pass)}, passed within 30d: {(fs.days_to_pass<=30).mean():.0%}")
    print(f"{'':<11} funded accts {len(fs):4d} | ever paid {fs.paid.mean():.0%} | days activation->first payout: {q(pd_.days_act_to_payout)}, within 90d of activation: {(pd_.days_act_to_payout<=90).mean():.0%}")
    both = (pd_.days_act_to_payout + fs.loc[pd_.index, 'days_to_pass']); print(f"{'':<11} total start->first payout (pass + payout, paid accounts only): {q(both)}")
show("2011-2022", 2011, 2022); show("2023-2026", 2023, 2026); show("all", 2011, 2026)
