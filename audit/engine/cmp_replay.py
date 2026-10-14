import sys
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53')
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53/audit')
import numpy as np
from real_engine import rules_from_config, FundedPlan, Stream
from chain_real import replay_detail
from pricing_config import load_config
from ref import ref_combine, ref_funded
from cmp_chain import ref_chain, FEES

cfg = load_config()
rng = np.random.default_rng(99)


def ref_replay(x, lo, atr, nd_c, nd_f, i0, n_end, path, policy, plan, dll, dyn_c):
    x, lo = x[:n_end], lo[:n_end]
    t = i0
    out = dict(fees=0., payouts=0., attempts=0, funded=0, n_payouts=0)
    while t < n_end - 5:
        sp, fees, att = ref_chain(x, lo, nd_c, t, n_end, policy, path, dll) if dyn_c is None else ref_chain_dyn(x, lo, atr, t, n_end, path, dll, dyn_c)
        out['fees'] += fees; out['attempts'] += att
        if sp is None: break
        out['funded'] += 1
        i_f = sp + 1
        if i_f >= n_end - 5: break
        pays, days, state, bal = ref_funded(x, lo, nd_f, atr, i_f, path=plan.path, t1=plan.t1, w1=plan.w1, r1=plan.r1, t2=plan.t2, w2=plan.w2, r2=plan.r2,
                                            max_payouts=plan.max_payouts, a1=plan.a1, a2=plan.a2, dll=dll)
        out['payouts'] += sum(pays); out['n_payouts'] += len(pays)
        t = i_f + days
        if state == 0: break
    return out


def ref_chain_dyn(x, lo, atr, t0, n_end, path, dll, a):
    sub, act, rst = FEES[path]
    t = t0; tot = 0.; att = 0
    while t < n_end - 5:
        tot += sub
        st, tau = ref_combine(x, lo, None, atr, t, dll=dll, a=a)
        att += 1
        tot += sub * (-(-tau // 21) - 1)
        if st == 1: return t + tau - 1, tot + act, att
        t += tau
    return None, tot, att

bad = 0; N = 0; pays = 0
for trial in range(600):
    n = int(rng.integers(300, 1500))
    sig = float(rng.choice([5, 10, 20, 40]))
    drift = float(rng.choice([0, .5, 1.5])) * sig / 3
    pnl = rng.normal(drift, sig, n)
    low = np.minimum(pnl, 0) - np.abs(rng.normal(0, sig * .5, n))
    atr = np.abs(rng.normal(sig * 3, sig, n)) + 5
    ok = np.ones(n, bool)
    s = Stream('s', np.arange(n).astype('datetime64[D]'), pnl, low, atr, ok)
    path = str(rng.choice(list(FEES)))
    r = rules_from_config(cfg, path)
    plan = FundedPlan('p', str(rng.choice(['standard', 'consistency'])), float(rng.choice([1500, 2000])), .5, 1., float(rng.choice([0., 1500])), .5, 1., int(rng.choice([2, 99])),
                      *( (None, None) if rng.random() < .5 else (0.2, 0.1)))
    Rc, Rf = float(rng.choice([1500, 3200, 6000])), float(rng.choice([1200, 3600]))
    dyn = None if rng.random() < .6 else 0.75
    pol = 'cancel_and_reopen' if dyn is not None else str(rng.choice(['cancel_and_reopen', 'immediate_reset', 'wait_for_rebill']))
    i0 = int(rng.integers(0, n // 3))
    ms = None if rng.random() < .5 else int(rng.integers(100, 600))
    a = replay_detail(s, Rc, s, Rf, plan, cfg, r, path, i0, ms, pol, dyn)
    n_end = n if ms is None else min(n, i0 + ms)
    ndc = s.desired_micros(Rc); ndf = s.desired_micros(Rf)
    b = ref_replay(pnl.tolist(), low.tolist(), atr.tolist(), ndc, ndf, i0, n_end, path, pol, plan, r.dll, dyn)
    N += 1; pays += b['n_payouts']
    adj = a['fees']
    # wait_for_rebill trailing extra rebill (documented conservative quirk) tolerated
    ok_ = (abs(a['payouts'] - b['payouts']) < 1e-6 and a['n_payouts'] == b['n_payouts'] and a['attempts'] == b['attempts'] and a['funded'] == b['funded'])
    fee_ok = abs(a['fees'] - b['fees']) < 1e-6 or (pol == 'wait_for_rebill' and (a['fees'] - b['fees']) in (FEES[path][0], 2 * FEES[path][0]))
    if not (ok_ and fee_ok):
        bad += 1
        if bad < 6: print('MISMATCH', path, pol, dyn, plan, i0, ms, {k: a[k] for k in ('fees', 'payouts', 'attempts', 'funded', 'n_payouts')}, b)
print('replays', N, 'bad', bad, 'total payouts', pays)
