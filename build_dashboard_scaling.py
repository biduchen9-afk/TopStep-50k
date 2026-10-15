"""Dashboard data: scaling the funded risk size before the first payout (pre-registration 12). Re-evaluates the selected cells and writes results/dashboard_v4/scaling_size.json."""
import json, sys
import numpy as np, pandas as pd
from portfolio import DEV_END
import run_kelly as K
import run_paid_share as P
import run_scaling_size as R1
import run_scaling_size2 as R2

gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
ST = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}; ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
AC = {k: P.Acc(ST[k], n_dev) for k in ST}; SPLIT = [("noise", 3), ("gap", 2)]; T1, A2 = 2000.0, 0.8
cells = {c["desc"]: c for c in R1.grid()}; cells.update({"r2: " + c["desc"]: c for c in R2.grid()})
PICK = [("constant 1.2", "Baseline", "Today: constant size 1.2 before the first payout"),
        ("r2: 0.4 -> 1.2 at $500 -> 1.6 at $1,000", "Meets the 80% rule, fails the stability check", "Start 0.4, step to 1.2 at +$500, to 1.6 at +$1,000"),
        ("up: 0.4 -> 1.2 at $1,000", "More paid, less profit", "Start 0.4, step to 1.2 at +$1,000"),
        ("r2: 0.4 -> 2.0 at $1,250", "Most paid, profit about 86%", "Start 0.4, step to 2.0 at +$1,250"),
        ("up: 0.4 -> 2.0 at $1,000", "More profit, bad-quartile farm worse", "Start 0.4, step to 2.0 at +$1,000"),
        ("up: 0.8 -> 2.0 at $500", "More profit, paid share unchanged", "Start 0.8, step to 2.0 at +$500"),
        ("r2: 0.5 -> 2.0 at $1,250", "Most profit", "Start 0.5, step to 2.0 at +$1,250"),
        ("base 1.2, last 10 days, x1.25 if up else x0.75", "More profit, slower farm", "Size 1.2 x1.25 after a profitable 10 days, x0.75 after a losing 10 days"),
        ("0.4 -> 1.2 over 40 sessions", "Time ramp", "Ramp from 0.4 to 1.2 over the first 40 sessions"),
        ("down: 1.2 -> 0.4 at $1,000", "Start big, step down", "Start 1.2, drop to 0.4 at +$1,000")]
rows = []; base = None
for key, verdict, label in PICK:
    c = cells[key]
    parts = {g: AC[g].run(c["a1"], T1, A2, None, c["scale"]) for g in AC}; acc = P.pooled([(parts["noise"], 3), (parts["gap"], 2)]); a = P.summ(acc)
    f = K.evaluate(P.plans_g(ST, SPLIT, c["a1"], T1, A2, None, c["scale"]), ref, n_dev, n); nets = np.array(f["nets"]); seen = json.loads(f["seen"]) if isinstance(f["seen"], str) else f["seen"]
    r = dict(setting=label, verdict=verdict, paid_pooled=round(a["paid_share"], 3), paid_noise=round(P.summ(parts["noise"])["paid_share"], 3), paid_gap=round(P.summ(parts["gap"])["paid_share"], 3), dev_mean=round(f["dev_mean"]), dev_p25=round(f["dev_p25"]), dev_worst=round(f["dev_worst"]),
             green=int(f["green"]), kept=None, days_account=round(a["days_start_to_payout"]), days_farm=round(f["days_first_payout"]), farms_2011_15=round(nets[:5].mean()), farms_2016_20=round(nets[5:].mean()),
             paid_2010_16=round(P.summ(acc[acc.start <= "2016-12-31"])["paid_share"], 3), paid_2017_22=round(P.summ(acc[acc.start > "2016-12-31"])["paid_share"], 3), seen_2021=seen[0], seen_2023=seen[1], seen_2025=seen[2])
    if base is None: base = r["dev_mean"]
    r["kept"] = round(r["dev_mean"] / base, 2); rows.append(r)
json.dump(rows, open("results/dashboard_v4/scaling_size.json", "w")); print(pd.DataFrame(rows).drop(columns=["verdict"]).to_string(index=False))
