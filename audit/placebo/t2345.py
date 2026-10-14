"""Placebos 2 (sign flip), 3 (delay), 4 (time shift), 5 (drift / legs) on real NQ."""
from common import *
from lib import *
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions

sess = build_sessions("nq")
ctx = ac.context(sess)
DEV = sess.dates <= DEV_END
rows = []


def add(label, d, extra=None):
    for nm, mk in (("full", None), ("dev<=2022", DEV), ("hold 2023+", ~DEV)):
        s = stats(d, mk)
        rows.append(dict(test=label, period=nm, **{k: round(v, 3) for k, v in s.items()}))


ref = ac.run_sleeve(sess, "noise_area", ctx=ctx)
add("0 project run_sleeve noise_area", ref)
mine = run_na(sess, ctx)
add("0b my reimplementation", mine)
print("max abs diff my vs project pnl:", np.abs(mine.pnl - ref.pnl).max(), np.abs(mine.minpnl - ref.minpnl).max(), (mine.trades != ref.trades).sum())

# 2. sign flip
add("2 sign flip (dirmult=-1)", run_na(sess, ctx, dirmult=-1))
# gross (zero-cost) versions to separate cost effect
zc = st.Costs(commission_rt=0.0, slippage_ticks_rt=0.0)
add("2b gross (no cost) normal", run_na(sess, ctx, costs=zc))
add("2c gross (no cost) flipped", run_na(sess, ctx, costs=zc, dirmult=-1))

# 3. delay
for dl in (1, 2, 5, 10):
    add(f"3 delay {dl} bar(s)", run_na(sess, ctx, delay=dl))
for dl in (1, 5):
    add(f"3b delay {dl} gross(no cost)", run_na(sess, ctx, costs=zc, delay=dl))

# 4. time shift
for s_ in (1, 2, 5):
    add(f"4 signal of day D-{s_} applied to day D", run_na(sess, ctx, shift=s_))
# random permutation of signal days (many seeds)
ok = ref.traded_ok
idx_ok = np.flatnonzero(ok)
shs = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    perm = np.arange(len(sess)); perm[idx_ok] = rng.permutation(idx_ok)
    d = run_na(sess, ctx, perm=perm)
    shs.append(stats(d)["sh_atr"])
rows.append(dict(test="4b signal from a RANDOM other day (20 perms): mean/sd/min/max Sharpe", period="full", sh_atr=round(np.mean(shs), 3), sh_raw=round(np.std(shs), 3), mean_usd=round(min(shs), 3), total_usd=round(max(shs), 3)))

# 5. drift / legs
add("5 long-only leg (allow=+1)", run_na(sess, ctx, allow=1))
add("5 short-only leg (allow=-1)", run_na(sess, ctx, allow=-1))
add("5 long-only gross", run_na(sess, ctx, costs=zc, allow=1))
add("5 short-only gross", run_na(sess, ctx, costs=zc, allow=-1))

# buy & hold same hours: long 09:30 open -> 15:55 close (one RT cost), every ok day; also short
cp = st.Costs().points(sess)
n = len(sess)
bh = np.zeros(n); inmkt = np.zeros(n)
for i in np.flatnonzero(ok):
    o, h, l, c, v, m = sess.day(i)
    a = int(np.searchsorted(m, st.CASH_OPEN)); e = int(np.searchsorted(m, st.FLAT_BY))
    if a >= len(c) - 20 or e - a < 100:
        continue
    bh[i] = (c[e - 1] - o[a]) * sess.mult
    inmkt[i] = (e - a)
bh_net = bh - cp * sess.mult
from intraday_strategies import DailySeries
d_bh = DailySeries(sess.dates, bh_net, np.zeros(n), np.ones(n, dtype=np.int32), ok, ref.atr, sess.mult)
add("5 buy&hold 09:30-15:55 (1 RT cost)", d_bh)
d_bh0 = DailySeries(sess.dates, bh, np.zeros(n), np.ones(n, dtype=np.int32), ok, ref.atr, sess.mult)
add("5 buy&hold 09:30-15:55 gross", d_bh0)
# overnight (prev close to 09:30 open) drift for context
on = np.zeros(n)
for i in np.flatnonzero(ok):
    o, h, l, c, v, m = sess.day(i)
    a = int(np.searchsorted(m, st.CASH_OPEN))
    if a >= len(c) - 20:
        continue
    on[i] = (o[a] - ctx["prev_close"][i]) * sess.mult
add("5 overnight gap long (prev close->09:30 open) gross", DailySeries(sess.dates, on, np.zeros(n), np.ones(n, dtype=np.int32), ok, ref.atr, sess.mult))

# exposure of noise_area: net signed minutes / time in market, beta to BH
def beta_alpha(d, bh_series, mask):
    x, y = bh_series[mask], d.pnl[mask]
    b = np.cov(x, y)[0, 1] / x.var(ddof=1)
    resid = y - b * x
    r = resid / (d.atr[mask] * d.mult)
    return b, resid.mean(), r.mean() / r.std(ddof=1) * np.sqrt(252), resid.mean() / resid.std(ddof=1) * np.sqrt(len(resid))

for nm, mk in (("full", ok), ("dev", ok & DEV), ("hold", ok & ~DEV)):
    b, a_, sh, t = beta_alpha(ref, bh, mk)
    print(f"beta of noise_area daily $ pnl on buy&hold(same hours) [{nm}]: beta={b:.3f}  alpha $/day={a_:.2f}  residual ATR-Sharpe={sh:.3f} t={t:.2f}")
    L = run_na(sess, ctx, allow=1); S = run_na(sess, ctx, allow=-1)
    bl, al, shl, tl = beta_alpha(L, bh, mk); bs_, as_, shs_, ts_ = beta_alpha(S, bh, mk)
    print(f"   long leg beta={bl:.3f} resid Sharpe={shl:.3f} t={tl:.2f} | short leg beta={bs_:.3f} resid Sharpe={shs_:.3f} t={ts_:.2f}")

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("audit/t2345_results.csv", index=False)
