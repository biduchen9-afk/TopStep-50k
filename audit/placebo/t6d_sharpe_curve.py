"""Farm net vs true Sharpe for a Gaussian stream with the real stream's vol (ATR-normalised sd), hold-out window."""
from common import *
real = real_stream(); ok = real.ok
r = np.where(ok, real.pnl / np.where(ok, real.atr_usd, 1), 0)
c_sd = r[ok].std(ddof=1)
n_h = int(np.searchsorted(real.dates, DEV_END, side="right"))
spec = FarmSpec(max_evals=20, max_funded=5, p_est=1.0)
offs = range(n_h, n_h + 60, 10)
rows = []
for sh in (0.0, 0.25, 0.5, 0.75, 0.9, 1.25, 1.5):
    mu = sh / np.sqrt(252) * c_sd
    nets = []
    for seed in range(80):
        s = gauss_stream(real, seed, c_sd, mean_per_atr=mu, cost=COST_DAY)   # sharpe is GROSS; costs subtracted
        nets.append(pd.DataFrame([run_farm([s], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, o, spec) for o in offs]).net.mean())
    nets = np.array(nets)
    rows.append(dict(gross_sharpe=sh, net_mean=nets.mean(), sd=nets.std(ddof=1), se=nets.std(ddof=1) / np.sqrt(len(nets)), p10=np.percentile(nets, 10), frac_pos=(nets > 0).mean()))
    print(rows[-1], flush=True)
print(pd.DataFrame(rows).round(1).to_string(index=False))
