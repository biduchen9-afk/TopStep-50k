"""Placebo 6b: zero-edge Gaussian streams, vol scan; farm (holdout window) and per-start contract EV."""
from common import *
import sys

real = real_stream()
ok = real.ok
r = np.where(ok, real.pnl / np.where(ok, real.atr_usd, 1), 0)
c_sd = r[ok].std(ddof=1)
n_h = int(np.searchsorted(real.dates, DEV_END, side="right"))
n = len(real.dates)
spec = FarmSpec(max_evals=20, max_funded=5, p_est=1.0)
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rows = []
# real reference
offs = range(n_h, n_h + 60, 10)
rr = pd.DataFrame([run_farm([real], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, o, spec) for o in offs])
print("REAL farm holdout net", rr.net.mean(), "fees", rr.fees.mean(), "payouts", rr.payouts.mean(), "funded", rr.n_funded.mean(), "evals", rr.n_evals.mean())
starts = np.arange(n_h, n - 300)
q = pd.DatetimeIndex(real.dates[starts]); grp = (q.year * 4 + (q.month - 1) // 3).to_numpy()
m = contract_metrics(real, real, SZ, plan(0.4, 0.4), None, cfg, rules, starts, grp)
print("REAL contract metrics holdout", {k: round(float(v), 3) for k, v in m.items()})
for scale in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    for cost in (0.0, COST_DAY):
        nets, evs, evf, pp, pf = [], [], [], [], []
        for seed in range(NSEED):
            s = gauss_stream(real, seed, c_sd, cost=cost, scale=scale)
            res = pd.DataFrame([run_farm([s], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, o, spec) for o in offs])
            nets.append(res.net.mean())
            if seed < 15:
                mm = contract_metrics(s, s, SZ, plan(0.4, 0.4), None, cfg, rules, starts, grp)
                evs.append(mm["ev_att"]); evf.append(mm["ev_funded"]); pp.append(mm["p_pass"]); pf.append(mm["p_first"])
        nets = np.array(nets)
        rows.append(dict(vol_scale=scale, cost=round(cost, 2), farm_net_mean=nets.mean(), farm_se=nets.std(ddof=1) / np.sqrt(len(nets)), farm_sd=nets.std(ddof=1),
                         frac_pos=(nets > 0).mean(), ev_per_combine_start=np.mean(evs), ev_per_funded=np.mean(evf), p_pass=np.mean(pp), p_first_payout=np.mean(pf)))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
print(df.round(2).to_string(index=False))
df.to_csv("audit/t6b_results.csv", index=False)
