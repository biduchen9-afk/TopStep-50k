import sys, itertools
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53')
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53/audit')
import numpy as np
from real_engine import RuleSet, FundedPlan, run_combine, run_funded_detail
from ref import ref_combine, ref_funded

rng = np.random.default_rng(123)
mism = {'combine': 0, 'funded': 0}
tot = {'combine': 0, 'funded': 0}
status_hist = {}
examples = []
fs = {}


def gen(n, sig, drift, dipfrac):
    pnl = rng.normal(drift, sig, n)
    # heavy tail days
    fat = rng.random(n) < 0.03
    pnl = np.where(fat, pnl * 3, pnl)
    dip = -np.abs(rng.normal(0, sig * dipfrac, n))
    low = np.minimum(np.minimum(pnl, 0), pnl + dip)  # low <= min(pnl,0)
    atr = np.abs(rng.normal(sig * 3, sig * .5, n)) + 1
    atr[rng.random(n) < 0.01] = np.nan
    nd = rng.integers(1, 40, n)
    return pnl.tolist(), low.tolist(), nd.tolist(), atr.tolist()

for trial in range(4000):
    n = int(rng.integers(30, 700))
    sig = float(rng.choice([2, 5, 10, 20, 40, 80]))   # $ per micro
    drift = float(rng.choice([-0.3, 0, 0.3, 0.8, 2])) * sig / 4
    dip = float(rng.choice([0.2, 0.8, 1.5]))
    x, lo, nd, atr = gen(n, sig, drift, dip)
    dll = None if rng.random() < .5 else -3000.0
    intr = bool(rng.random() < .8)
    r = RuleSet(dll=dll, intraday_mll=intr)
    i0 = int(rng.integers(0, max(1, n // 2)))
    dyn = None if rng.random() < .5 else float(rng.choice([0.25, 0.75, 1.5]))
    s1 = run_combine(x, lo, nd, i0, r, dyn, atr)
    s2 = ref_combine(x, lo, nd, atr, i0, dll=dll, a=dyn, intraday=intr)
    tot['combine'] += 1
    status_hist[s1[0]] = status_hist.get(s1[0], 0) + 1
    if s1 != s2:
        mism['combine'] += 1
        if len(examples) < 5:
            examples.append(('combine', n, sig, i0, dll, dyn, intr, s1, s2))
    # funded
    path = 'standard' if rng.random() < .5 else 'consistency'
    a1 = None if rng.random() < .5 else float(rng.choice([0.05, 0.2, 0.4]))
    a2 = None if (a1 is None or rng.random() < .3) else float(rng.choice([0.05, 0.2, 0.4]))
    t1 = float(rng.choice([1500, 2000, 3000]))
    t2 = None if rng.random() < .3 else float(rng.choice([0.0, 1500, 2000]))
    r1 = float(rng.choice([1.0, 0.5, 1.7]))
    r2 = float(rng.choice([1.0, 0.5]))
    mp = int(rng.choice([2, 3, 99]))
    w1 = float(rng.choice([0.3, 0.5, 0.9]))
    plan = FundedPlan('p', path, t1, w1, r1, t2, 0.5, r2, mp, a1, a2)
    f1 = run_funded_detail(x, lo, nd, i0, r, plan, atr)
    f2 = ref_funded(x, lo, nd, atr, i0, path=path, t1=t1, w1=w1, r1=r1, t2=t2, r2=r2, max_payouts=mp, a1=a1, a2=a2, dll=dll, intraday=intr)
    tot['funded'] += 1
    fs.setdefault((f1[2], min(len(f1[0]),3)),0); fs[(f1[2], min(len(f1[0]),3))]+=1
    same = (len(f1[0]) == len(f2[0]) and np.allclose(f1[0], f2[0]) and f1[1] == f2[1] and f1[2] == f2[2] and abs(f1[3] - f2[3]) < 1e-6)
    if not same:
        mism['funded'] += 1
        if len(examples) < 10:
            examples.append(('funded', n, sig, i0, dll, plan, intr, f1, f2))
print(tot, mism, status_hist)
print('funded (state,npay):', sorted(fs.items()))
for e in examples:
    print(e)
