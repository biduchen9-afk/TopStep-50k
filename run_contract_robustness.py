import numpy as np, pandas as pd
import intraday_strategies as st
from market_data import build_sessions
from real_engine import Stream, rules_from_config
from chain_real import replay_detail
from run_contract_final import plan, HOLD_SEEDS
from run_final_checks import build, cfg
nq = build_sessions("nq")
strat, _, n_dev = build("nq", "vwap_atr", {"k": 0.3, "s": 0.2, "start_min": st.CASH_OPEN})
yrs = (strat.dates[-1] - strat.dates[n_dev]).astype(float) / 365.25
offs = list(range(n_dev, n_dev + 60, 6)); seeds = list(HOLD_SEEDS)[:30]
pl = plan(8000, 1.0); Rc, Rf = 6400, 10800
rows = []
def run(label, path, costs=None, **ov):
    rules = rules_from_config(cfg, path, **ov)
    coins = [Stream.from_series("c", st.run(nq, "coin", seed=s, s=None, costs=costs) if costs else st.run(nq, "coin", seed=s, s=None)) for s in seeds]
    per = [np.mean([replay_detail(c, Rc, c, Rf, pl, cfg, rules, path, o)["net"] for o in offs]) for c in coins]
    per = np.array(per)
    rows.append(dict(variant=label, mean=per.mean(), se=per.std(ddof=1)/np.sqrt(len(per)), median=np.median(per), p10=np.percentile(per,10), p90=np.percentile(per,90), pct_seeds_loss=(per<0).mean()))
run("baseline: no_activation_dll, 1 tick, real-time MLL", "no_activation_dll")
run("fee path: standard (no DLL)", "standard")
run("fee path: no_activation (no DLL)", "no_activation")
run("slippage 3 ticks", "no_activation_dll", costs=st.Costs(slippage_ticks_rt=3.0))
run("commission $5 + 3 ticks", "no_activation_dll", costs=st.Costs(commission_rt=5.0, slippage_ticks_rt=3.0))
run("stricter: Combine MLL never locks (spec)", "no_activation_dll", combine_mll_locks=False)
run("looser: breach only at the close", "no_activation_dll", intraday_mll=False)
run("payout cap halved ($2,500 gross)", "no_activation_dll", cap_standard=2500.0)
run("5 winning days -> 10 winning days", "no_activation_dll", standard_win_days=10)
d = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(f"NO-EDGE, frozen dev config Rc{Rc} Rf{Rf} t1=8000, hold-out 3.5y, {len(seeds)} fresh seeds x {len(offs)} offsets")
print(d.round(0).to_string(index=False)); d.to_csv("results/contract_robustness.csv", index=False)
