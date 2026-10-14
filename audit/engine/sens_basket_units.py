import sys, pickle
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53')
import numpy as np, pandas as pd
import farm_real
from real_engine import rules_from_config, FundedPlan, Stream
from farm_real import run_farm, FarmSpec
from run_farm_robustness import build, cfg, W
from run_farm_dev import plan, SZ
from portfolio import DEV_END
from run_dynamic_contract import PATH

print('basket weights', {k: round(v, 2) for k, v in W.items()}, 'sum', round(sum(W.values()), 2))
base = build()
s = base['portfolio']
n_h = int(np.searchsorted(s.dates, DEV_END, side='right'))

# instrument: record k (baskets) per funded day via spy on run_funded_detail's internals is awkward; instead replicate k logic
_rf = farm_real.run_funded_detail
stats = {'days': 0, 'over': 0, 'maxk': 0}


def spy(x, lo, nd, t, r, plan_, atr):
    from real_engine import xfa_micro_cap
    # re-simulate with logging of k
    bal = peak = 0.0; floor = -r.mll; reset = False
    res = _rf(x, lo, nd, t, r, plan_, atr)
    return res
farm_real.run_funded_detail = spy

def run(rules, label, sz=SZ):
    r = pd.DataFrame([run_farm([s], sz, None, plan(.4, .4), cfg, rules, PATH, o, FarmSpec(max_evals=20, max_funded=5, p_est=1.0)) for o in range(n_h, n_h + 60, 5)])
    print(label, 'net', round(r.net.mean()), 'payouts', round(r.payouts.mean()), 'fees', round(r.fees.mean()), 'funded', round(r.n_funded.mean(), 1))

rules = rules_from_config(cfg, PATH)
run(rules, 'as published (caps counted in baskets)')
tot = sum(W.values())
ladder = tuple((b, l / tot) for b, l in rules.ladder)
run(rules_from_config(cfg, PATH, ladder=ladder, combine_micros=int(150 / tot)), f'caps in micros (baskets = {tot:.1f} micros)')

# how often the cap binds in the published run: baskets k vs cap in baskets
xs = s.pnl.tolist(); ls = s.low.tolist(); at = s.atr_usd.tolist()
# Combine fixed sizing
nd = s.desired_micros(SZ.value)
nd = np.array(nd)
print('combine fixed 6400: baskets/day percentiles', np.percentile(nd[s.ok], [50, 90, 99, 100]), 'cap(micros) 150 -> in baskets', round(150 / tot, 1), 'share days above', round(float((nd[s.ok] > 150 / tot).mean()), 3))
