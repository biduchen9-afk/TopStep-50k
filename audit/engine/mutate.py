import subprocess, shutil, os, sys
ROOT = '/home/user/TopStep-50k/.claude/worktrees/agent-adb3c4dffd3b61f53'
MUT = '/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/mut'
MUTANTS = [
    ('real_engine.py', 'DLL liquidation removed (combine)', "                if dll is not None and dll > lim:\n                    p = dll                     # DLL reached first: liquidate, day over", "                if dll is not None and dll > lim:\n                    pass"),
    ('real_engine.py', 'DLL booked at p (no cap) in combine elif', "            elif dll is not None and w <= dll:\n                p = dll", "            elif dll is not None and w <= dll:\n                pass"),
    ('real_engine.py', 'funded: MLL not reset after payout', "                floor, reset = 0.0, True", "                floor, reset = floor, False"),
    ('real_engine.py', 'funded: payout not deducted from balance', "                bal -= gross\n", "                pass\n"),
    ('real_engine.py', 'consistency tested vs bal*1.0', "if bal >= r.profit_target and best <= r.consistency_ratio * bal:", "if bal >= r.profit_target:"),
    ('real_engine.py', 'bills fewer: tau//cyc', "bills = 1 + (np.maximum(tau, 1) - 1) // cfg.billing_cycle_days", "bills = 1 + (np.maximum(tau, 1) - 1) // (cfg.billing_cycle_days + 7)"),
    ('real_engine.py', 'activation fee never charged (combine_outcomes)', "np.where(status == 1, p.activation_fee, 0.0)", "0.0"),
    ('real_engine.py', 'funded breach test looser (<)', "        if w <= first:\n            if dll is not None and dll > lim:\n                p = dll\n            elif r.intraday_mll:", "        if w < first - 1.0:\n            if dll is not None and dll > lim:\n                p = dll\n            elif r.intraday_mll:"),
    ('real_engine.py', 'xfa ladder not applied', "        k = min(k, xfa_micro_cap(bal, r.ladder))\n", "        k = min(k, 150)\n"),
    ('real_engine.py', 'funded floor never trails', "            floor = min(peak - r.mll, 0.0)", "            floor = floor"),
    ('real_engine.py', 'funded floor trails w/o cap at 0', "            floor = min(peak - r.mll, 0.0)", "            floor = peak - r.mll"),
    ('real_engine.py', 'combine MLL no lock', "            if r.combine_mll_locks and floor > 0.0:\n                floor = 0.0", "            pass"),
    ('real_engine.py', 'combine floor trails from intraday not EOD (peak uses p>0 clip)', "        if bal > peak:\n            peak = bal\n            floor = peak - r.mll", "        if bal > peak:\n            peak = bal\n            floor = peak - r.mll + 500"),
    ('real_engine.py', 'dyn combine cushion min 500->5000', "max(bal - floor, 500.0)", "max(bal - floor, 5000.0)"),
    ('real_engine.py', 'dyn funded cushion min 250->2500', "max(bal - floor, 250.0)", "max(bal - floor, 2500.0)"),
    ('real_engine.py', 'a2 ignored (always a1)', "a = plan.a1 if (stage == 1 or plan.a2 is None) else plan.a2", "a = plan.a1"),
    ('real_engine.py', 'win-day threshold ignored', "        if p >= r.win_day:", "        if p > 0:"),
    ('real_engine.py', 'payout split 100%', "pays.append(r.split * gross)", "pays.append(gross)"),
    ('real_engine.py', 'payout cap ignored', "gross = min(min(frac, 0.5) * bal, cap_pay)", "gross = min(frac, 0.5) * bal"),
    ('real_engine.py', 'payout fraction >50% allowed', "gross = min(min(frac, 0.5) * bal, cap_pay)", "gross = min(frac * bal, cap_pay)"),
    ('real_engine.py', 'consistency xfa 40->100', "best <= r.consistency_xfa * prof", "best <= 1.0 * prof"),
    ('real_engine.py', 'combine micros cap ignored (dyn)', "r.combine_micros)\n        p, w", "10**9)\n        p, w"),
    ('chain_real.py', 'chain: activation fee omitted', "return end - 1, fees + act, attempts, end - 1", "return end - 1, fees, attempts, end - 1"),
    ('chain_real.py', 'chain: in-attempt renewals omitted (cancel_and_reopen)', "            fees += sub * ((max(tau, 1) - 1) // cyc)", "            pass"),
    ('chain_real.py', 'chain: wait idle rebill omitted', "            fees += sub                             # renewal that grants the free reset", "            pass"),
    ('chain_real.py', 'chain: keep_clock reset fee omitted', "            fees += rst\n            if t > c0", "            if t > c0"),
    ('chain_real.py', 'chain: reopen sub omitted', "            fees += sub\n            c0 = t\n        status", "            c0 = t\n        status"),
    ('chain_real.py', 'replay: fees dropped', 'out["fees"] += fees', 'out["fees"] += 0'),
    ('chain_real.py', 'replay: funded days lost (t not advanced)', "t = i_f + days", "t = i_f"),
    ('farm_real.py', 'farm: activation not charged', 'out["fees"] += act', 'out["fees"] += 0'),
    ('farm_real.py', 'farm: eval renewals omitted', 'out["fees"] += (1 + (max(tau, 1) - 1) // cyc) * sub', 'out["fees"] += sub'),
    ('farm_real.py', 'farm: funded cap 5 ignored', "if len(funded) < spec.max_funded:", "if True:"),
    ('farm_real.py', 'farm: eval cap ignored', "while (len(evals) < spec.max_evals and", "while (True and"),
    ('farm_real.py', 'farm: funded slot freed immediately', "funded.append(t + days)", "funded.append(t + 1)"),
    ('farm_real.py', 'farm: waiting pass converted without a slot but payouts', "out[\"payouts\"] += sum(pays)", "out[\"payouts\"] += 2 * sum(pays)"),
    ('contract_eval.py', 'contract: tot uses first only', "tot[k] = sum(pays)", "tot[k] = pays[0] if pays else 0.0"),
    ('contract_eval.py', 'contract: fees dropped from net', "net = tot - c[\"fees\"]", "net = tot"),
]

def run(tests):
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider'] + tests, cwd=MUT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]

if __name__ == '__main__':
    survived = []
    for f, name, old, new in MUTANTS:
        if os.path.exists(MUT): shutil.rmtree(MUT)
        shutil.copytree(ROOT, MUT, ignore=shutil.ignore_patterns('.git', 'audit', '__pycache__', 'results'))
        s = open(os.path.join(MUT, f)).read()
        if old not in s:
            print('PATTERN NOT FOUND', name); continue
        open(os.path.join(MUT, f), 'w').write(s.replace(old, new, 1))
        code, last = run(['test_valuation.py','test_engine_audit.py'])
        status = 'killed' if code != 0 else 'SURVIVED'
        print(f'{status:9s} {f:16s} {name}  [{last}]')
        if code == 0: survived.append(name)
    print('\nSURVIVORS', len(survived), 'of', len(MUTANTS))
