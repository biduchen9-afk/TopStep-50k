"""Zero-edge Gaussian (with cost drag) farm net in rolling 756-session windows: does the option value depend on era/ATR level?"""
from common import *
real = real_stream()
ok = real.ok
r = np.where(ok, real.pnl / np.where(ok, real.atr_usd, 1), 0)
c_sd = r[ok].std(ddof=1)
spec = FarmSpec(max_evals=20, max_funded=5, p_est=1.0)
first = int(np.flatnonzero(ok)[0])
rows = []
for w0 in range(first, len(real.dates) - 760, 250):
    nets = []
    for seed in range(30):
        s = gauss_stream(real, seed, c_sd, cost=0.0)
        nets.append(run_farm([s], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, w0, spec, n_end=w0 + 756)["net"])
    nets = np.array(nets)
    rows.append(dict(start=str(real.dates[w0]), mean_atr_usd=np.nanmean(real.atr_usd[w0:w0 + 756]), net_mean=nets.mean(), se=nets.std(ddof=1) / np.sqrt(30), frac_pos=(nets > 0).mean()))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows); print(df.round(1).to_string(index=False)); print("grand mean", df.net_mean.mean())
