import sys
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53')
import numpy as np, pandas as pd
from dataclasses import replace
from real_engine import Stream, rules_from_config, FundedPlan
from farm_real import run_farm, FarmSpec
from run_farm_robustness import build, cfg
from run_farm_dev import plan, SZ
from portfolio import DEV_END
from run_dynamic_contract import PATH

base = build()


def farm(stream, pl, rules=None, path=PATH, evals=20, step=5):
    n_h = int(np.searchsorted(stream.dates, DEV_END, side='right'))
    rules = rules or rules_from_config(cfg, path)
    r = pd.DataFrame([run_farm([stream], SZ, None, pl, cfg, rules, path, o, FarmSpec(max_evals=evals, max_funded=5, p_est=1.0)) for o in range(n_h, n_h + 60, step)])
    return round(r.net.mean()), round(r.payouts.mean()), round(r.fees.mean()), round(r.n_funded.mean(), 1), round(r.died.mean(), 1)

for name in ('portfolio', 'NQ noise-area'):
    s = base[name]
    print(name, 'baseline net,payouts,fees,funded,died', farm(s, plan(.4, .4)))
    # neutralise the single best day
    i = int(np.argmax(s.pnl))
    pnl = s.pnl.copy(); low = s.low.copy(); pnl[i] = 0; low[i] = 0
    s2 = replace(s, pnl=pnl, low=low)
    print(name, 'minus', s.dates[i], farm(s2, plan(.4, .4)))
    # winsorise top 5 days to the 99.5 pct of ok days
    thr = np.percentile(s.pnl[s.ok], 99.5)
    pnl = np.minimum(s.pnl, thr)
    s3 = replace(s, pnl=pnl)
    print(name, 'winsorised at p99.5', round(thr), farm(s3, plan(.4, .4)))
    # worst-case: intraday low = max(low, ...) already; stress: lows 1.5x worse
    s4 = replace(s, low=s.low * 1.5)
    print(name, 'lows x1.5', farm(s4, plan(.4, .4)))
    # funded phase without DLL (rules.dll=None applies to both phases -> run with standard path rules)
    r_nodll = rules_from_config(cfg, PATH, dll=None)
    print(name, 'no DLL anywhere (fees as DLL path)', farm(s, plan(.4, .4), rules=r_nodll))
    # payout approvals: 25% of payouts delayed by requiring 10 win days
    print(name, 'win days 8', farm(s, plan(.4, .4), rules=rules_from_config(cfg, PATH, standard_win_days=8)))
