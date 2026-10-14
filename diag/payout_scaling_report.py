"""Per-start-year nets for the pre-registered winner and the fast candidate, plus descriptive hold-out starts and the no-edge control; and a deliberately broken (one-session-shifted) stream as a leakage check."""
import numpy as np, pandas as pd
import intraday_strategies as st
from combine_policy import CombinePolicy
from final_strategy import build_stream
from market_data import build_sessions
from real_engine import FundedPlan, Stream, costs_from_config
from pricing_config import load_config
from risk_units import with_risk_unit
from run_funded_payout_policy import farm_plan
g = pd.read_pickle("results/payout_scaling_dev_grid.pkl")
s = build_stream(); d = pd.DatetimeIndex(s.dates)
n_dev = int(np.searchsorted(s.dates, np.datetime64("2022-12-31"), side="right"))
idx = lambda dt: int(np.searchsorted(s.dates, np.datetime64(dt)))
CFG = {"baseline 3000/0.4/1.6": (3000., .4, 1.6), "WINNER 4000/0.4/0.8": (4000., .4, .8), "FAST 1500/0.8/0.8": (1500., .8, .8)}
pol = CombinePolicy(R=9600.)
def mk(t1, a1, a2): return FundedPlan("p", "standard", t1, .5, 1., 0., .5, 1., 99, a1, a2)
print("start-year farm nets (3y, ends <= 2022-12), 2011..2020:")
for k, (t1, a1, a2) in CFG.items():
    r = g[(g.t1 == t1) & (g.a1 == a1) & (g.a2 == a2)].iloc[0]; print(f"  {k:<24}", r.nets, f"green {r.green}/10")
print("\nDESCRIPTIVE (seen) farm starts through end of data:")
starts = ["2021-01-04", "2022-01-03", "2023-01-02", "2024-01-02", "2025-01-02", "2025-07-01"]
coin = [with_risk_unit(Stream.from_series("c", st.run(build_sessions("nq"), "coin", seed=k, s=None, costs=costs_from_config(load_config()))), "atr", 40) for k in range(8)]
for k, (t1, a1, a2) in CFG.items():
    pl = mk(t1, a1, a2)
    real = [round(farm_plan(s, pol, pl, idx(x))["net"]) for x in starts]
    ctl = [round(np.mean([farm_plan(c, pol, pl, idx(x))["net"] for c in coin])) for x in starts]
    full = farm_plan(s, pol, pl, int(np.flatnonzero(s.ok)[0]))
    print(f"  {k:<24} starts {starts}\n  {'':<24} strategy {real}\n  {'':<24} no-edge control (8 seeds) {ctl}\n  {'':<24} full history net {full['net']:,.0f} (payouts {full['payouts']:,.0f}, fees {full['fees']:,.0f})")
# leakage check: shift P&L one session AGAINST the signal timing (trade yesterday's signal's P&L today) -> should destroy the edge
sh = Stream(s.name, s.dates, np.r_[0.0, s.pnl[:-1]], np.r_[0.0, s.low[:-1]], s.atr_usd, s.ok)
pl = mk(4000., .4, .8)
print("\nLEAK CHECK (P&L shifted by one session, i.e. broken alignment) -> winner config nets from 2011/2015/2019/2023 starts:", [round(farm_plan(sh, pol, pl, idx(x))["net"]) for x in ("2011-01-03", "2015-01-02", "2019-01-02", "2023-01-02")])
print("                       (unshifted)                                                                         :", [round(farm_plan(s, pol, pl, idx(x))["net"]) for x in ("2011-01-03", "2015-01-02", "2019-01-02", "2023-01-02")])
