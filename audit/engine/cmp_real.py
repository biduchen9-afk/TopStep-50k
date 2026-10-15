import sys, pickle
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53')
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53/audit')
import numpy as np
from real_engine import RuleSet, FundedPlan, Stream, run_combine, run_funded_detail, rules_from_config
from pricing_config import load_config
from ref import ref_combine, ref_funded

cfg = load_config()
D = pickle.load(open('/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/streams.pkl', 'rb'))
mm = 0; tot = 0
stats = {}
for name, (nm, dates, pnl, low, atr, ok) in D.items():
    s = Stream(nm, dates, pnl, low, atr, ok)
    x, lo, atrl = pnl.tolist(), low.tolist(), atr.tolist()
    n = len(x)
    for pathkey in ('standard', 'no_activation_dll'):
        r = rules_from_config(cfg, pathkey)
        for kind, val in (('fixed', 3200.), ('fixed', 6400.), ('dyn', .75), ('dyn', 1.5)):
            nd = s.desired_micros(val) if kind == 'fixed' else [1] * n
            dyn = val if kind == 'dyn' else None
            starts = list(range(0, n - 300, 3))
            res = []
            for i0 in starts:
                a = run_combine(x, lo, nd, i0, r, dyn, atrl)
                b = ref_combine(x, lo, nd, atrl, i0, dll=r.dll, a=dyn)
                tot += 1
                if a != b:
                    mm += 1
                    if mm < 6: print('COMBINE MISMATCH', name, pathkey, kind, val, i0, a, b)
                res.append(a)
                if a[0] == 1:
                    for plan, Rf in ((FundedPlan('d', 'standard', 1500, .5, 1., 0., .5, 1., 99, .2, .1), None),
                                     (FundedPlan('d2', 'consistency', 3000, .5, 1., 0., .5, 1., 99, .4, .05), None),
                                     (FundedPlan('f', 'standard', 2000, .5, 1., None, .5, 1., 2), 1200.)):
                        ndf = s.desired_micros(Rf) if Rf else [1] * n
                        i_f = i0 + a[1]
                        f1 = run_funded_detail(x, lo, ndf, i_f, r, plan, atrl)
                        f2 = ref_funded(x, lo, ndf, atrl, i_f, path=plan.path, t1=plan.t1, w1=plan.w1, t2=plan.t2, max_payouts=plan.max_payouts,
                                        a1=plan.a1, a2=plan.a2, dll=r.dll)
                        tot += 1
                        same = (len(f1[0]) == len(f2[0]) and np.allclose(f1[0], f2[0]) and f1[1:3] == f2[1:3] and abs(f1[3] - f2[3]) < 1e-6)
                        if not same:
                            mm += 1
                            if mm < 6: print('FUNDED MISMATCH', name, pathkey, plan, i_f, f1, f2)
            st = np.array([q[0] for q in res])
            stats[(name, pathkey, kind, val)] = (round((st == 1).mean(), 3), round((st == 0).mean(), 3), round((st == -1).mean(), 3))
print('total compared', tot, 'mismatches', mm)
for k, v in stats.items(): print(k, v)
