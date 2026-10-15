import sys, pickle
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53')
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53/audit')
import io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from cmp_replay import ref_replay
from real_engine import rules_from_config, FundedPlan, Stream
from chain_real import replay_detail
from pricing_config import load_config
from portfolio import DEV_END

cfg = load_config()
D = pickle.load(open('/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/streams.pkl', 'rb'))
path = 'no_activation_dll'
r = rules_from_config(cfg, path)
for nm, plan, Rc in (('portfolio', FundedPlan('a', 'standard', 3000, .5, 1., 0., .5, 1., 99, .2, .2), 6400.),
                     ('nq_noise', FundedPlan('b', 'standard', 3000, .5, 1., 0., .5, 1., 99, .1, .05), 6400.)):
    s = Stream(*D[nm])
    x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist()
    n_dev = int(np.searchsorted(s.dates, DEV_END, side='right'))
    ndc = s.desired_micros(Rc); ndf = [1] * len(x)
    tot_a = tot_b = 0
    for o in range(n_dev, n_dev + 60, 4):
        a = replay_detail(s, Rc, s, 0.0, plan, cfg, r, path, o)
        b = ref_replay(x, lo, atr, ndc, ndf, o, len(x), path, 'cancel_and_reopen', plan, r.dll, None)
        assert abs(a['fees'] - b['fees']) < 1e-6 and abs(a['payouts'] - b['payouts']) < 1e-6 and a['attempts'] == b['attempts'] and a['funded'] == b['funded'], (nm, o, a, b)
        tot_a += a['payouts'] - a['fees']; tot_b += b['payouts'] - b['fees']
    print(nm, 'replay matches reference; mean net', tot_a / 15, tot_b / 15)
