"""Valid placebos for the farm configs (the one-session-shift 'leak check' in payout_scaling_report.py is NOT informative: shifting a near-iid daily series changes almost nothing).
  demeaned : same path and volatility, but the strategy's mean ATR-normalised edge is removed (dev-fitted mean) -> what the rules/volatility alone give
  reversed : every trade flipped (pnl -> -pnl)           -> must lose if the edge is real
Reported as mean farm net over the ten dev start-years (2011..2020, 3-year farms) and as descriptive hold-out starts."""
import numpy as np, pandas as pd
from combine_policy import CombinePolicy
from final_strategy import build_stream
from portfolio import DEV_END
from real_engine import FundedPlan, Stream
from run_funded_payout_policy import farm_plan
s = build_stream(); n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
dev = (s.dates <= DEV_END) & s.ok
mu_R = float(np.mean(s.pnl[dev] / s.atr_usd[dev]))                    # dev-fitted mean daily edge in ATR units
dm = np.where(s.ok, s.pnl - mu_R * np.nan_to_num(s.atr_usd), s.pnl)
variants = {"strategy": s,
            "demeaned (edge removed)": Stream(s.name, s.dates, dm, np.minimum(s.low - (s.pnl - dm), 0.0), s.atr_usd, s.ok),
            "reversed (trades flipped)": Stream(s.name, s.dates, -s.pnl, np.minimum(-s.pnl, 0.0), s.atr_usd, s.ok)}
CFG = {"baseline 3000/0.4/1.6": (3000., .4, 1.6), "WINNER 4000/0.4/0.8": (4000., .4, .8), "FAST 1500/0.8/0.8": (1500., .8, .8)}
idx = lambda dt: int(np.searchsorted(s.dates, np.datetime64(dt)))
pol = CombinePolicy(R=9600.)
print(f"dev mean edge removed: {mu_R:.4f} ATR-units/day")
for k, (t1, a1, a2) in CFG.items():
    pl = FundedPlan("p", "standard", t1, .5, 1., 0., .5, 1., 99, a1, a2)
    print(f"\n{k}")
    for vn, v in variants.items():
        nets = [farm_plan(v, pol, pl, idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["net"] for y in range(2011, 2021)]
        hold = [round(farm_plan(v, pol, pl, idx(x))["net"]) for x in ("2021-01-04", "2023-01-02", "2025-01-02")]
        print(f"  {vn:<26} dev mean {np.mean(nets):9,.0f}  green {sum(n > 0 for n in nets)}/10  worst {min(nets):9,.0f} | seen starts 2021/2023/2025: {hold}")
