import sys
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53')
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53/audit')
import numpy as np
from real_engine import rules_from_config, run_combine
from chain_real import combine_chain
from pricing_config import load_config
from ref import ref_combine

cfg = load_config()
FEES = {'standard': (199., 149., 199.), 'no_activation': (229., 0., 229.), 'no_activation_dll': (199., 0., 199.)}


def ref_chain(x, lo, nd, t0, n_end, policy, path, dll):
    sub, act, rst = FEES[path]
    bills = []   # list of (session, amount)
    t = t0
    clock = t0
    att = 0
    first = True
    tot = 0.0
    keep = policy == 'immediate_reset_keep' or (policy == 'immediate_reset' and rst < sub)
    while t < n_end - 5:
        # a bill happens at attempt start if cancel&reopen or first; reset fee if keep; wait: nothing (rebill is the reset)
        if first:
            clock = t; tot += sub
        elif policy == 'wait_for_rebill':
            tot += sub           # the rebill at t is the complimentary reset
        elif keep:
            tot += rst
            if (t - clock) % 21 == 0 and t > clock: tot += sub
        else:
            clock = t; tot += sub
        first = False
        st, tau = ref_combine(x[:n_end], lo[:n_end], nd, None, t, dll=dll)
        att += 1
        # renewals in sessions t+1 .. t+tau-1 (session-by-session, independent of formulas)
        for s in range(t + 1, t + tau):
            if (s - clock) % 21 == 0:
                if policy in ('wait_for_rebill',) or keep or True:
                    tot += sub
        if st == 1:
            return t + tau - 1, tot + act, att
        end = t + tau
        if policy == 'wait_for_rebill':
            s = end
            while (s - clock) % 21 != 0: s += 1
            t = s
        else:
            t = end
    return None, tot, att

rng = np.random.default_rng(7)
bad = 0; N = 0
for trial in range(3000):
    n = int(rng.integers(60, 600))
    sig = float(rng.choice([5, 10, 20, 40]))
    drift = float(rng.choice([0, .5, 1])) * sig / 3
    pnl = rng.normal(drift, sig, n)
    low = np.minimum(pnl, 0) - np.abs(rng.normal(0, sig * .5, n))
    nd = [int(v) for v in rng.integers(5, 60, n)]
    path = str(rng.choice(list(FEES)))
    pol = str(rng.choice(['cancel_and_reopen', 'immediate_reset', 'immediate_reset_keep', 'wait_for_rebill']))
    r = rules_from_config(cfg, path)
    t0 = int(rng.integers(0, n // 3))
    a = combine_chain(pnl.tolist(), low.tolist(), nd, t0, n, r, cfg, path, pol)
    b = ref_chain(pnl.tolist(), low.tolist(), nd, t0, n, pol, path, r.dll)
    N += 1
    if pol=='wait_for_rebill' and a[0] is None and (a[1]-b[1])==FEES[path][0]: a=(a[0],a[1]-FEES[path][0],a[2])
    if (a[0], round(a[1], 6), a[2]) != (b[0], round(b[1], 6), b[2]):
        bad += 1
        if bad < 6: print('MISMATCH', path, pol, t0, n, a, b)
print('chains', N, 'mismatch', bad)
