"""Does spacing Combine purchases out (decorrelated attempts) shorten the wait for the farm's first payout? Topstep only, self-funded rule with k0 Combines allowed before the first payout."""
import numpy as np, pandas as pd
from final_strategy import build_stream
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import FundedPlan
s = build_stream(); d = pd.DatetimeIndex(s.dates); n = len(d); n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
idx = lambda dt: int(np.searchsorted(s.dates, np.datetime64(dt)))
ts = specs()["Topstep 150K"]["spec"]
runs = [(y, min(idx(f"{y}-01-01") + 756, n_dev)) for y in range(2012, 2021)] + [(y, n) for y in (2021, 2023, 2025)]
CF = {"previous": FundedPlan("p", "x", 4000., .5, 1., 0., .5, 1., 99, .4, .8), "fast": FundedPlan("p", "x", 6000., .5, 1., 0., .5, 1., 99, 2.4, .8)}
rows = []
for cf, pl in CF.items():
    for k0, gap in ((1, 0), (3, 0), (3, 5), (3, 10), (5, 5), (5, 10), (5, 21), (8, 10)):
        r_ = []
        for y, ne in runs:
            i0 = idx(f"{y}-01-01")
            r = run_multifirm([FirmPlan(ts, [s], 9600., pl)], Policy(k0=k0, reserve_cycles=3.0, single_pipeline=(k0 == 1), stagger_gap=gap), i0, ne)
            v = r["firms"][ts.name]
            r_.append((y, r["oop_total"], (d[min(v["first_payout_t"], n - 1)] - d[v["first_buy_t"]]).days if v["first_payout_t"] is not None else np.nan, r["net"]))
        df = pd.DataFrame(r_, columns=["y", "oop", "days", "net"]); dv = df[df.y <= 2020]; sn = df[df.y > 2020]
        rows.append(dict(cfg=cf, k0=k0, gap=gap, dev_days_med=dv.days.median(), dev_never=int(dv.days.isna().sum()), dev_in90=float((dv.days <= 90).mean()), dev_in180=float((dv.days <= 180).mean()),
                         dev_oop_med=dv.oop.median(), dev_oop_max=dv.oop.max(), dev_net=dv.net.mean(), seen_days=str(sn.days.round(0).tolist()), seen_oop=str(sn.oop.round(0).tolist()), seen_net=str(sn.net.round(0).tolist())))
R = pd.DataFrame(rows); R.to_csv("results/staggered_starts.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 50)
print(R.round(1).to_string(index=False))
