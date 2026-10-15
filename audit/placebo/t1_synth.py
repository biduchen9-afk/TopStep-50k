"""Placebo 1: synthetic zero-edge NQ -> noise_area sleeve + dynamic Combine/funded/farm pipeline."""
from common import *
from lib import stats, run_na
from synth import make_synth
import academic_strategies as ac
import intraday_strategies as st
from market_data import build_sessions
from real_engine import Stream
import sys, time

NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20
sess = build_sessions("nq")
DEV = sess.dates <= DEV_END
n_h = int(np.searchsorted(sess.dates, DEV_END, side="right"))
n = len(sess)
spec = FarmSpec(max_evals=20, max_funded=5, p_est=1.0)
offs = range(n_h, n_h + 60, 10)
starts = np.arange(n_h, n - 300)
q = pd.DatetimeIndex(sess.dates[starts]); grp = (q.year * 4 + (q.month - 1) // 3).to_numpy()


def pipeline(d):
    s = Stream.from_series("x", d)
    res = pd.DataFrame([run_farm([s], SZ, None, plan(0.4, 0.4), cfg, rules, PATH, o, spec) for o in offs])
    mm = contract_metrics(s, s, SZ, plan(0.4, 0.4), None, cfg, rules, starts, grp)
    return dict(farm_net=res.net.mean(), farm_fees=res.fees.mean(), farm_pay=res.payouts.mean(), n_funded=res.n_funded.mean(), p_pass=mm["p_pass"], ev_att=mm["ev_att"], p_first=mm["p_first"])


# sanity on the first synthetic set
sy = make_synth(sess, 0, "bar")
dsy = np.abs(np.diff(sy.c)) ; dre = np.abs(np.diff(sess.c))
print("sanity: mean |dc| real", dre.mean(), "synth", dsy.mean(), "| corr of abs steps by minute-of-session ok:", np.allclose(np.abs(np.diff(sy.c))[sess.offsets[1:-1] - 1 + 0 * 1].mean(), np.abs(np.diff(sess.c))[sess.offsets[1:-1] - 1].mean()) or "n/a")
print("real: first close", sess.c[0], "last", sess.c[-1], " synth bar: first", sy.c[0], "last", sy.c[-1], "min", sy.l.min())

# real reference
ref = ac.run_sleeve(sess, "noise_area")
real_row = dict(mode="REAL", seed=-1, **stats(ref), **{f"h_{k}": v for k, v in stats(ref, ~DEV).items() if k in ("sh_atr",)}, **pipeline(ref))
rows = [real_row]
print(real_row, flush=True)
for mode in ("bar", "session", "permute"):
    for seed in range(NSEED):
        t0 = time.time()
        sy = make_synth(sess, 1000 + seed, mode)
        ctx = ac.context(sy)
        d = ac.run_sleeve(sy, "noise_area", ctx=ctx)
        row = dict(mode=mode, seed=seed, **stats(d), h_sh_atr=stats(d, ~DEV)["sh_atr"])
        row["sh_atr_flipped"] = stats(run_na(sy, ctx, dirmult=-1))["sh_atr"]
        row.update(pipeline(d))
        rows.append(row)
        print(mode, seed, round(row["sh_atr"], 3), round(row["farm_net"]), round(row["ev_att"]), f"{time.time()-t0:.1f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv("audit/t1_results.csv", index=False)
g = df[df["mode"] != "REAL"].groupby("mode")[["sh_atr", "sh_raw", "mean_usd", "t", "h_sh_atr", "sh_atr_flipped", "farm_net", "ev_att", "p_pass", "p_first", "n_funded"]].agg(["mean", "std"])
print(g.round(3).T.to_string())
print(df[df["mode"] == "REAL"].round(3).to_string())
