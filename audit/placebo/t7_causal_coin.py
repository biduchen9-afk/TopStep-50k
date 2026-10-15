"""(a) look-ahead perturbation test on noise_area decisions; (b) 'coin' control through the same farm for many seeds; (c) per-year P&L."""
from common import *
from lib import na_day, stats
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions
from real_engine import Stream

sess = build_sessions("nq")
ctx = ac.context(sess)
cp = st.Costs().points(sess)
ref = ac.run_sleeve(sess, "noise_area", ctx=ctx)

# (a) causality: replace every bar AFTER the first bar at/after cutoff T with garbage; decisions at times <= T must be unchanged
rng = np.random.default_rng(5)
days = rng.choice(np.flatnonzero(ref.traded_ok), 400, replace=False)
bad = 0; checked = 0
for T in (st.CASH_OPEN + 60, st.CASH_OPEN + 180, st.CASH_OPEN + 270):
    for i in days:
        o, h, l, c, v, m = [x.copy() for x in sess.day(i)]
        e0 = []; na_day(o, h, l, c, v, m, ctx["prev_close"][i], ctx["sigma"][i], cp, events=e0)
        idxT = int(np.searchsorted(m, T))
        k = len(c) - idxT - 1
        if k <= 5: continue
        step = rng.normal(0, 8, k).cumsum()
        c2, o2, h2, l2, v2 = c.copy(), o.copy(), h.copy(), l.copy(), v.copy()
        base = c[idxT]
        c2[idxT + 1:] = base + step
        o2[idxT + 1:] = np.r_[base, c2[idxT + 1:-1]]
        h2[idxT + 1:] = np.maximum(o2[idxT + 1:], c2[idxT + 1:]) + 2
        l2[idxT + 1:] = np.minimum(o2[idxT + 1:], c2[idxT + 1:]) - 2
        v2[idxT + 1:] = rng.integers(1, 1000, k)
        e1 = []; na_day(o2, h2, l2, c2, v2, m, ctx["prev_close"][i], ctx["sigma"][i], cp, events=e1)
        a = [(t, x) for t, x in e0 if t <= T and t < m[-1]]
        b = [(t, x) for t, x in e1 if t <= T and t < m[-1]]
        checked += 1
        if a != b:
            bad += 1
print(f"(a) causality perturbation: {checked} (day,cutoff) cases, decisions before cutoff changed in {bad}")

# (b) coin control through the farm
n_h = int(np.searchsorted(sess.dates, DEV_END, side="right"))
n = len(sess)
spec = FarmSpec(max_evals=20, max_funded=5, p_est=1.0)
offs = range(n_h, n_h + 60, 10)
starts = np.arange(n_h, n - 300)
q = pd.DatetimeIndex(sess.dates[starts]); grp = (q.year * 4 + (q.month - 1) // 3).to_numpy()
rows = []
for stop in (None,):
    for seed in range(60):
        d = st.run(sess, "coin", seed=seed, s=stop)
        s = Stream.from_series("coin", d)
        res = pd.DataFrame([run_farm([s], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, o, spec) for o in offs])
        r = s.pnl[s.ok] / s.atr_usd[s.ok]
        mm = contract_metrics(s, s, SZ, plan(0.4, 0.4), None, cfg, rules, starts, grp) if seed < 20 else {}
        rows.append(dict(seed=seed, sh=stats(d)["sh_atr"], atr_sd=r.std(ddof=1), farm_net=res.net.mean(), ev_att=mm.get("ev_att", np.nan), p_pass=mm.get("p_pass", np.nan)))
df = pd.DataFrame(rows)
print("(b) coin control, 60 seeds, holdout farm:", df.describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
print("   fraction of seeds with farm_net>0:", (df.farm_net > 0).mean())
df.to_csv("audit/t7_coin.csv", index=False)

# (c) per-year P&L of the real sleeve (per mini)
yr = pd.DatetimeIndex(ref.dates).year
y = pd.DataFrame({"y": yr, "pnl": ref.pnl, "ok": ref.traded_ok})
y = y[y.ok].groupby("y").pnl.agg(["sum", "mean", "std", "count"])
y["sharpe"] = y["mean"] / y["std"] * np.sqrt(252)
print("(c) per-year noise_area $ per mini:\n", y.round(1).to_string())
