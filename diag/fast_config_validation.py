"""Validation of the pre-registration-5 candidate (a1=2.4, first-payout balance 6000, post-payout size 0.8) against the previous config: by-year nets, seen starts, placebos,
farm-level time to the first payout (parallel farm and the single-pipeline out-of-pocket rule)."""
import numpy as np, pandas as pd
from combine_policy import CombinePolicy
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from final_strategy import PATH, build_stream
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from pricing_config import load_config
from real_engine import FundedPlan, Stream, rules_from_config
from run_funded_payout_policy import farm_plan
cfg = load_config(); rules = rules_from_config(cfg, PATH)
s = build_stream(); d = pd.DatetimeIndex(s.dates); n = len(d)
n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
idx = lambda dt: int(np.searchsorted(s.dates, np.datetime64(dt)))
CF = {"previous (a1 0.4, t1 4000)": FundedPlan("p", "standard", 4000., .5, 1., 0., .5, 1., 99, .4, .8),
      "balanced (a1 1.2, t1 2000)": FundedPlan("p", "standard", 2000., .5, 1., 0., .5, 1., 99, 1.2, .8),
      "fast (a1 2.4, t1 6000)": FundedPlan("p", "standard", 6000., .5, 1., 0., .5, 1., 99, 2.4, .8)}
pol = CombinePolicy(R=9600.)
mu = float(np.mean(s.pnl[(s.dates <= DEV_END) & s.ok] / s.atr_usd[(s.dates <= DEV_END) & s.ok]))
dmp = np.where(s.ok, s.pnl - mu * np.nan_to_num(s.atr_usd), s.pnl)
variants = {"strategy": s, "edge removed": Stream(s.name, s.dates, dmp, np.minimum(s.low - (s.pnl - dmp), 0.0), s.atr_usd, s.ok),
            "trades flipped": Stream(s.name, s.dates, -s.pnl, np.minimum(-s.pnl, 0.0), s.atr_usd, s.ok)}
print("DEV start-year nets (3-year farms 2011..2020):")
for k, pl in CF.items():
    nets = [round(farm_plan(s, pol, pl, idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["net"]) for y in range(2011, 2021)]
    print(f"  {k:<28} {nets}  mean {np.mean(nets):,.0f}  green {sum(v > 0 for v in nets)}/10  worst {min(nets):,}")
print("\nPLACEBOS (dev mean over the ten farms, green count):")
for k, pl in CF.items():
    for vn, v in variants.items():
        nets = [farm_plan(v, pol, pl, idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["net"] for y in range(2011, 2021)]
        print(f"  {k:<28} {vn:<15} mean {np.mean(nets):10,.0f}  green {sum(x > 0 for x in nets)}/10")
print("\nSEEN starts to end of data (2021, 2022, 2023, 2024, 2025-01, 2025-07) and full history:")
for k, pl in CF.items():
    r = [round(farm_plan(s, pol, pl, idx(x))["net"]) for x in ("2021-01-04", "2022-01-03", "2023-01-02", "2024-01-02", "2025-01-02", "2025-07-01")]
    f = farm_plan(s, pol, pl, int(np.flatnonzero(s.ok)[0]))
    print(f"  {k:<28} {r}  full history net {f['net']:,.0f} (payouts {f['payouts']:,.0f}, fees {f['fees']:,.0f}, funded {f['n_funded']}, died {f['died']})")
print("\nFARM-LEVEL days from the first Combine to the FIRST payout of the farm (starts every 21 sessions):")
def first_payout_days(pl, i0, ne):
    r = run_farm([s], CombineSizing("fixed", 9600.), None, pl, cfg, rules, PATH, i0, FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill="hyst"), n_end=ne, log=True)
    t = [e[0] for e in r["events"] if e[1] == "payout"]
    return (d[min(min(t), n - 1)] - d[i0]).days if t else np.nan
first = int(np.flatnonzero(s.ok)[0]) + 260
for k, pl in CF.items():
    dev = [first_payout_days(pl, i, min(i + 400, n_dev)) for i in range(first, n_dev - 400, 21)]
    hold = [first_payout_days(pl, i, n) for i in range(idx("2023-01-01"), n - 150, 21)]
    f = lambda v: f"median {np.nanmedian(v):.0f}d | within 90d {np.mean(np.array(v) <= 90):.0%} | within 180d {np.mean(np.array(v) <= 180):.0%} | no payout in window {np.mean(np.isnan(v)):.0%}"
    print(f"  {k:<28} dev: {f(dev)}\n  {'':<28} 2023+: {f(hold)}")
print("\nSINGLE-PIPELINE self-funded rule (out of pocket until first payout; one account at a time), Topstep only:")
ts = specs()["Topstep 150K"]["spec"]
for k, pl in CF.items():
    rows = []
    for y in list(range(2012, 2021)) + [2021, 2023, 2025]:
        i0 = idx(f"{y}-01-01"); ne = min(i0 + 756, n_dev) if y <= 2020 else n
        r = run_multifirm([FirmPlan(ts, [s], 9600., pl)], Policy(k0=1, reserve_cycles=3.0), i0, ne)
        v = r["firms"][ts.name]
        rows.append((y, r["oop_total"], (d[min(v["first_payout_t"], n - 1)] - d[v["first_buy_t"]]).days if v["first_payout_t"] is not None else np.nan, r["net"]))
    df = pd.DataFrame(rows, columns=["start", "oop", "days_to_first_payout", "net"])
    dv = df[df.start <= 2020]
    print(f"  {k:<28} dev 2012-2020: OOP median ${dv.oop.median():,.0f} max ${dv.oop.max():,.0f} | days to first payout median {dv.days_to_first_payout.median():.0f} (never in window: {dv.days_to_first_payout.isna().sum()}/9) | mean net ${dv.net.mean():,.0f}, green {int((dv.net>0).sum())}/9")
    print(f"  {'':<28} seen 2021/2023/2025: days {df[df.start > 2020].days_to_first_payout.round(0).tolist()} | net {df[df.start > 2020].net.round(0).tolist()} | OOP {df[df.start > 2020].oop.round(0).tolist()}")
