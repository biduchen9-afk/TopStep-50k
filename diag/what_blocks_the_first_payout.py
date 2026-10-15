"""For funded accounts that DO reach their first payout: which condition arrives last - the 5 winning days (>= $150 each) or the balance trigger? (Topstep standard path)"""
import numpy as np, pandas as pd
from final_strategy import PATH, build_stream
from pricing_config import load_config
from real_engine import FundedPlan, rules_from_config, run_combine, xfa_micro_cap
rules = rules_from_config(load_config(), PATH); s = build_stream(); d = pd.DatetimeIndex(s.dates)
x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist(); nd = s.desired_micros(9600.0); n = len(x)
def first_payout_blockers(i0, a1, t1):
    bal = peak = 0.0; floor = -rules.mll; wins = 0; t_bal = t_win = None
    for t in range(i0, n):
        k = 1 if not (atr[t] > 0) else max(1, int(round(a1 * max(bal - floor, 250.0) / atr[t]))); k = min(k, xfa_micro_cap(bal, rules.ladder))
        p, w = x[t] * k, lo[t] * k; lim = floor - bal
        first = max(lim, rules.dll)
        if w <= first:
            if rules.dll > lim: p = rules.dll
            else: return None
        bal += p; wins += p >= rules.win_day
        if bal > peak: peak = bal; floor = min(peak - rules.mll, 0.0)
        if t_bal is None and bal >= t1: t_bal = t - i0 + 1
        if t_win is None and wins >= rules.standard_win_days: t_win = t - i0 + 1
        if t_bal is not None and t_win is not None and (bal * 0.5 >= rules.payout_min):
            return t_bal, t_win, t - i0 + 1, k
    return None
first = int(np.flatnonzero(s.ok)[0]) + 260
for a1, t1 in ((0.4, 4000.), (0.4, 1500.), (2.4, 6000.), (2.4, 1500.)):
    rows = []
    for i0 in range(first, n - 300, 3):
        st, tau = run_combine(x, lo, nd, i0, rules)
        if st == 1:
            r = first_payout_blockers(i0 + tau + 1, a1, t1)
            if r: rows.append(r)
    df = pd.DataFrame(rows, columns=["t_bal", "t_win", "t_pay", "k"])
    print(f"a1={a1} t1=${int(t1)}: {len(df)} paid accounts | median sessions to reach the balance {df.t_bal.median():.0f} | to get 5 winning days {df.t_win.median():.0f} | to payout {df.t_pay.median():.0f} | the balance was the LAST condition in {(df.t_bal >= df.t_win).mean():.0%} of cases, the winning days in {(df.t_win > df.t_bal).mean():.0%}")
