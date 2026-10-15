"""Dashboard data: days to payout and accounts paid for the baseline and the two forward-test candidates. Reads results/days_candidates.csv."""
import json
import pandas as pd

R = pd.read_csv("results/days_candidates.csv"); rows = []
short = {"baseline (today)": "Baseline: constant 1.2 before and 0.8 after the first payout", "candidate 2: after payout 0.4, 1.2 at $3,500": "Option 2: after the first payout start 0.4, step to 1.2 at a $3,500 balance",
         "candidate 3: before payout 0.8, 2.0 at +$500": "Option 3: before the first payout start 0.8, step to 2.0 at +$500 profit"}
nn = lambda v, f=lambda x: x: None if pd.isna(v) else f(v)
for r in R.itertuples():
    if r.period.startswith("whole farm"):
        rows.append(dict(level="Whole farm, ten 3-year farms (2011-2020 starts)", option=short[r.config], accounts_paid=round(r.farm_paid_share, 3), paid_noise=None, paid_gap=None, days_first=round(r.farm_days_med), days_p25=round(r.farm_days_p25), days_p75=round(r.farm_days_p75),
                         within_90=round(r.farms_within_90 / 10, 2), within_180=round(r.farms_within_180 / 10, 2), act_to_first=None, between_payouts=None, payouts_per_paid=None))
    else:
        rows.append(dict(level=f"One account, {r.period}", option=short[r.config], accounts_paid=round(r.funded_paid, 3), paid_noise=round(r.paid_noise, 3), paid_gap=round(r.paid_gap, 3), days_first=round(r.days_first_med), days_p25=round(r.days_first_p25), days_p75=round(r.days_first_p75),
                         within_90=round(r.paid_within_90, 3), within_180=round(r.paid_within_180, 3), act_to_first=round(r.days_act_to_first_med), between_payouts=nn(r.days_between_1st_2nd_med, round), payouts_per_paid=round(r.payouts_per_paid, 2)))
order = {"One account, 2010-2022 starts": 0, "One account, 2023+ starts": 1}
rows.sort(key=lambda x: (order.get(x["level"], 2), x["option"][:8]))
json.dump(rows, open("results/dashboard_v4/candidates_days.json", "w")); print(len(rows))
