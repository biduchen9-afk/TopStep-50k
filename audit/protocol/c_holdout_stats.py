"""Audit C/D: reproduce the nq_noise hold-out contract metrics, dev-grid row, and look at the statistical support behind them."""
import sys, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
import intraday_strategies as st
from portfolio import DEV_END, basket, dev_weights, make_sleeve
from real_engine import FundedPlan, Stream, rules_from_config, combine_outcomes, run_funded_detail, _lists
from contract_eval import CombineSizing, contract_metrics
from pricing_config import load_config
from run_dynamic_contract import PATH, plan_grid, sleeves

pd.set_option("display.width", 250, "display.max_columns", 40)
cfg = load_config(); RULES = rules_from_config(cfg, PATH)
S = sleeves()
w = dev_weights(S, shrink=0.5)
wp = pickle.load(open("/home/user/data_raw/portfolio_weights.pkl", "rb"))
print("dev weights recomputed", {k: round(v, 3) for k, v in w.items()})
print("saved weights equal:", all(abs(w[k] - wp[k]) < 1e-9 for k in w))
nq = basket([S[0]], name="nq_noise")
n = len(nq.dates); n_dev = int(np.searchsorted(nq.dates, DEV_END, side="right"))

# ---- dev row reproduction (nq_noise, Rc6400, frozen plan)
dgrid = pd.read_csv("results/dynamic_dev_grid.csv")
row = dgrid[(dgrid.stream == "nq_noise") & dgrid.covers & (dgrid.ev_funded > 0)].sort_values(["p_first", "ev_funded"], ascending=False).iloc[0]
print("frozen row:", row["combine"], row.plan, "p_first", round(row.p_first, 3), "ev_funded", round(row.ev_funded))
print("n qualifying nq_noise configs:", int(((dgrid.stream == "nq_noise") & dgrid.covers & (dgrid.ev_funded > 0)).sum()), "of", int((dgrid.stream == "nq_noise").sum()))
top = dgrid[(dgrid.stream == "nq_noise") & dgrid.covers & (dgrid.ev_funded > 0)].sort_values(["p_first", "ev_funded"], ascending=False).head(6)
print(top[["combine", "plan", "p_pass", "p_first", "payouts_per_funded", "ev_funded"]].round(2).to_string())
plan, Rf = next((p, r) for p, r in plan_grid() if p.name == row.plan)
sz = CombineSizing("fixed", float(row["combine"][2:])) if row["combine"].startswith("Rc") else CombineSizing("dyn", float(row["combine"][4:]))
cut = Stream(nq.name, nq.dates[:n_dev], nq.pnl[:n_dev], nq.low[:n_dev], nq.atr_usd[:n_dev], nq.ok[:n_dev])
first = int(np.flatnonzero(cut.ok)[0])
starts = np.arange(first, n_dev - 400)
yrs = pd.DatetimeIndex(cut.dates[starts]).year.to_numpy()
m = contract_metrics(cut, cut, sz, plan, Rf, cfg, RULES, starts, yrs)
print("DEV reproduced:", {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items() if k in ("p_pass", "p_first", "ev_funded", "payouts_per_funded", "n_funded")}, "csv ev", round(row.ev_funded, 1))

# ---- hold-out reproduction
hs = np.arange(n_dev, n - 300)
q = pd.DatetimeIndex(nq.dates[hs]); grp = (q.year * 4 + (q.month - 1) // 3).to_numpy()
mh = contract_metrics(nq, nq, sz, plan, Rf, cfg, RULES, hs, grp)
print("HOLD reproduced:", {k: round(v, 3) if isinstance(v, float) else v for k, v in mh.items()})

# ---- per-start detail to examine overlap / effective N
xc, lc = _lists(nq)
c = combine_outcomes(xc, lc, nq.desired_micros(sz.value), cfg, PATH, RULES, hs)
ps = c["status"] == 1
fund_idx = (hs + c["tau"])[ps]
print("hold-out starts:", len(hs), "passed:", int(ps.sum()), "distinct funding dates:", len(np.unique(fund_idx)))
dates_f = pd.DatetimeIndex(nq.dates[np.minimum(fund_idx, n - 1)])
print("funding dates by quarter:\n", pd.Series(1, index=dates_f).resample("QE").sum().to_string())
# per funded: payouts, death
atr_f = nq.atr_usd.tolist(); xf, lf = _lists(nq)
res = []
for i_f in np.unique(fund_idx):
    pays, d, state, bal = run_funded_detail(xf, lf, [1] * n, int(i_f), RULES, plan, atr_f)
    res.append((i_f, sum(pays), len(pays), d, state))
r = pd.DataFrame(res, columns=["i_f", "tot", "n", "days", "state"])
print("distinct funded trajectories:", len(r), " died:", int((r.state == 1).sum()), " alive at data end:", int((r.state == 0).sum()), " mean payouts", r.n.mean().round(2))
# trajectories collapse into few distinct paths: count distinct (tot,n,days-end) end signatures
end_sig = (r.i_f + r.days)
print("distinct end sessions of funded trajectories:", end_sig.nunique())
# 'independent' blocks: cluster per quarter-of-funding; per-quarter mean payout
r["q"] = pd.DatetimeIndex(nq.dates[np.minimum(r.i_f, n - 1)]).to_period("Q")
print(r.groupby("q").agg(n=("tot", "size"), tot=("tot", "mean"), pays=("n", "mean"), alive=("state", lambda s: (s == 0).mean())).round(1).to_string())

# ---- bootstrap of EV with different block definitions
net_att = np.zeros(len(hs))
tot = {}
for k in np.flatnonzero(ps):
    i_f = int(hs[k] + c["tau"][k])
    pays, d, state, _ = run_funded_detail(xf, lf, [1] * n, i_f, RULES, plan, atr_f)
    net_att[k] = sum(pays)
net = net_att - c["fees"]
rng = np.random.default_rng(0)
def boot(groups, B=2000):
    ug = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in ug}
    out = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug))
        out.append(np.concatenate([net[idx[g]] for g in pick]).mean())
    return np.percentile(out, [2.5, 50, 97.5]).round(0), len(ug)
print("EV/attempt point", net.mean().round(0))
print("quarter blocks (analyst):", boot(grp))
yr = q.year.to_numpy()
print("calendar-year blocks:", boot(yr))
half = ((q.year - 2023) * 2 + (q.month > 6)).to_numpy()
print("half-year blocks:", boot(half))
# moving block bootstrap on the underlying daily R stream (what matters for edge)
r_d = np.where(nq.ok, nq.pnl / np.where(nq.ok, nq.atr_usd, 1), 0)[n_dev:]
ok = nq.ok[n_dev:]
x = r_d[ok]
sh = lambda a: a.mean() / a.std(ddof=1) * np.sqrt(252)
print("hold-out NQ-noise Sharpe", round(sh(x), 2), "n days", len(x))
for bl in (1, 10, 21, 63):
    reps = []
    for _ in range(3000):
        nb = int(np.ceil(len(x) / bl)); st_ = rng.integers(0, len(x) - bl + 1, nb)
        reps.append(sh(np.concatenate([x[s:s + bl] for s in st_])[:len(x)]))
    print(f"  block={bl:>3}d Sharpe 95% CI", np.percentile(reps, [2.5, 97.5]).round(2))
xd = r_d[:0]
dev_x = (np.where(nq.ok, nq.pnl / np.where(nq.ok, nq.atr_usd, 1), 0)[:n_dev])[nq.ok[:n_dev]]
print("dev Sharpe", round(sh(dev_x), 2), "n", len(dev_x))
ys = pd.Series(np.where(nq.ok, nq.pnl / np.where(nq.ok, nq.atr_usd, 1), 0), index=pd.DatetimeIndex(nq.dates))[nq.ok]
print("Sharpe by year:\n", ys.groupby(ys.index.year).apply(lambda a: a.mean() / a.std() * np.sqrt(252)).round(2).to_string())
