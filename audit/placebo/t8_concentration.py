from common import *
from lib import stats
import academic_strategies as ac
from market_data import build_sessions
from real_engine import Stream
sess = build_sessions("nq")
d = ac.run_sleeve(sess, "noise_area")
r = np.where(d.traded_ok, d.pnl / (d.atr * d.mult), np.nan)
yr = pd.DatetimeIndex(d.dates).year
t = pd.DataFrame({"y": yr, "r": r}).dropna().groupby("y").r.agg(["mean", "std", "count"])
t["sharpe"] = t["mean"] / t["std"] * np.sqrt(252)
print(t.round(3).T.to_string())
m = d.traded_ok
for lo, hi in ((2010, 2017), (2018, 2022), (2023, 2026), (2010, 2022)):
    mk = (yr >= lo) & (yr <= hi)
    print(lo, hi, {k: round(v, 3) for k, v in stats(d, mk).items()})
# top-day dependence on the farm (hold-out)
real = Stream.from_series("nq", d)
n_h = int(np.searchsorted(real.dates, DEV_END, side="right"))
spec = FarmSpec(max_evals=20, max_funded=5, p_est=1.0)
offs = range(n_h, n_h + 60, 10)
def farm(s):
    return pd.DataFrame([run_farm([s], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, o, spec) for o in offs]).net.mean()
print("farm hold-out net, as is:", farm(real))
import sys
if len(sys.argv)>1: pass
rr = np.where(real.ok, real.pnl / np.where(real.ok, real.atr_usd, 1), 0)
hold = np.arange(len(rr)) >= n_h
for k in (5, 10, 20):
    idx = np.argsort(-np.where(hold, rr, -9))[:k]
    pnl = real.pnl.copy(); low = real.low.copy()
    pnl[idx] = 0.0; low[idx] = 0.0
    s2 = Stream("x", real.dates, pnl, low, real.atr_usd, real.ok)
    print(f"remove top {k} hold-out days (set to flat): farm net", round(farm(s2)), "| removed ATR-r sum", round(rr[idx].sum(), 2), "of total", round(rr[hold].sum(), 2))
# iid bootstrap of hold-out ATR-normalised days: edge preserved, path structure destroyed
rows = []
for seed in range(40):
    rng = np.random.default_rng(seed)
    src = np.flatnonzero(real.ok & hold)
    pick = rng.choice(src, size=len(real.pnl) - n_h)
    pnl = real.pnl.copy(); low = real.low.copy(); atr = real.atr_usd.copy()
    sc = np.nan_to_num(real.atr_usd[n_h:] / real.atr_usd[pick], nan=0.0)
    sc = np.where(real.ok[n_h:], sc, 0.0)
    pnl[n_h:] = real.pnl[pick] * sc; low[n_h:] = real.low[pick] * sc
    s2 = Stream("x", real.dates, pnl, low, real.atr_usd, real.ok)
    rows.append(farm(s2))
print("iid bootstrap of hold-out ATR-normalised days (edge kept): farm net mean %.0f sd %.0f min %.0f max %.0f" % (np.mean(rows), np.std(rows), np.min(rows), np.max(rows)))
