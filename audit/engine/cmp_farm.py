import sys, pickle
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53')
import numpy as np
import farm_real
from farm_real import run_farm, FarmSpec
from contract_eval import CombineSizing
from real_engine import rules_from_config, FundedPlan, Stream
from pricing_config import load_config

cfg = load_config()
CALLS = {'c': [], 'f': []}
_rc, _rf = farm_real.run_combine, farm_real.run_funded_detail


def rc(x, lo, nd, t, r, dyn, atr):
    res = _rc(x, lo, nd, t, r, dyn, atr); CALLS['c'].append((id(x), t, res)); return res


def rf(x, lo, nd, t, r, plan, atr):
    res = _rf(x, lo, nd, t, r, plan, atr); CALLS['f'].append((id(x), t, res)); return res

farm_real.run_combine, farm_real.run_funded_detail = rc, rf


def check(streams, sizing, Rf, plan, path, i0, spec, n_end=None, label=''):
    CALLS['c'].clear(); CALLS['f'].clear()
    rules = rules_from_config(cfg, path)
    out = run_farm(streams, sizing, Rf, plan, cfg, rules, path, i0, spec, n_end)
    sub, act = cfg.paths[path].monthly_cost, cfg.paths[path].activation_fee
    # fees recomputed from call log
    fees = sum((1 + (max(tau, 1) - 1) // 21) * sub for (_, _, (st, tau)) in CALLS['c']) + len(CALLS['f']) * act
    assert abs(fees - out['fees']) < 1e-6, (fees, out['fees'])
    assert out['n_evals'] == len(CALLS['c']) and out['n_funded'] == len(CALLS['f'])
    pay = sum(sum(r_[0]) for (_, _, r_) in CALLS['f'])
    assert abs(pay - out['payouts']) < 1e-6
    # concurrency from the log
    horizon = max([t + res[1] for (_, t, res) in CALLS['c']] + [t + res[1] for (_, t, res) in CALLS['f']] + [0]) + 5
    ev = np.zeros(horizon + 1); fu = np.zeros(horizon + 1)
    for (_, t, (st, tau)) in CALLS['c']: ev[t:t + tau] += 1
    for (_, t, (pays, days, state, bal)) in CALLS['f']: fu[t:t + days] += 1
    assert ev.max() <= spec.max_evals and fu.max() <= spec.max_funded, (ev.max(), fu.max())
    # every funded activation must be preceded by a distinct passed eval on the same stream ending <= activation
    passes = sorted([(t + res[1], id_) for (id_, t, res) in CALLS['c'] if res[0] == 1])
    used = 0
    acts = sorted([(t, id_) for (id_, t, _) in CALLS['f']])
    assert len(acts) <= len(passes)
    # pass-before-activation greedy matching per stream
    from collections import defaultdict
    P = defaultdict(list); A = defaultdict(list)
    for e, i in passes: P[i].append(e)
    for t, i in acts: A[i].append(t)
    for i in A:
        p_, a_ = sorted(P[i]), sorted(A[i])
        for k, ta in enumerate(a_):
            assert p_[k] <= ta, ('activation before pass', i, p_[k], ta)
    # no funded account starts at or after the end of the loop window
    n_all = min(len(s.dates) for s in streams); ne = n_all if n_end is None else min(n_all, n_end)
    assert all(t < ne - 5 for (_, t, _) in CALLS['c']) and all(t < ne - 5 for (_, t, _) in CALLS['f'])
    print(label, 'ok', {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out.items() if k in ('fees', 'payouts', 'n_evals', 'n_passed', 'n_funded', 'n_wasted', 'max_funded_seen', 'max_evals_seen', 'died', 'net')},
          'log max evals', int(ev.max()), 'funded', int(fu.max()))
    return out

if __name__ == '__main__':
    rng = np.random.default_rng(5)
    for trial in range(8):
        n = 2500
        sig = [100, 200, 400][trial % 3]
        pnl = rng.normal(sig * 0.1, sig, n)
        s = Stream('r', np.arange(n).astype('datetime64[D]'), pnl, np.minimum(pnl, 0) - sig * .3, np.full(n, 500.0), np.ones(n, bool))
        plan = FundedPlan('p', 'standard', 1500, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.2, 0.2)
        spec = FarmSpec(max_evals=20, max_funded=5, p_est=[0.4, 0.1, 0.01][trial % 3], starts_per_day=[1, 3][trial % 2])
        for path in ('no_activation_dll', 'standard'):
            check([s], CombineSizing('fixed', 3000), None, plan, path, 0, spec, label=f'synth{trial}/{path}')
    D = pickle.load(open('/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/streams.pkl', 'rb'))
    S = [Stream(*D[k]) for k in ('portfolio', 'nq_noise')]
    plan = FundedPlan('p', 'standard', 1500, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.2, 0.1)
    for sz in (CombineSizing('fixed', 3200), CombineSizing('dyn', 0.75)):
        for path in ('no_activation_dll', 'standard'):
            check(S, sz, None, plan, path, 300, FarmSpec(), n_end=2300, label=f'real {sz.label()}/{path}')
            check(S, sz, None, plan, path, 300, FarmSpec(starts_per_day=3, p_est=0.0), n_end=2300, label=f'real aggressive {sz.label()}/{path}')
