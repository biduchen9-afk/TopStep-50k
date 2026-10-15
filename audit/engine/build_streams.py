import sys, pickle, time
sys.path.insert(0, '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53')
t = time.time()
from run_dynamic_contract import streams
S, w = streams()
print({k: (len(v.dates), int(v.ok.sum())) for k, v in S.items()}, w, time.time() - t)
pickle.dump({k: (v.name, v.dates, v.pnl, v.low, v.atr_usd, v.ok) for k, v in S.items()}, open('/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/streams.pkl', 'wb'))
