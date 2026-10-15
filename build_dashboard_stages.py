"""Dashboard data: scaling the size in the Combine and after the first payout (pre-registration 13). Reads results/scaling_combine_grid.csv and scaling_after_grid.csv."""
import json
import pandas as pd

C = pd.read_csv("results/scaling_combine_grid.csv"); A = pd.read_csv("results/scaling_after_grid.csv")
rows = []


def add(stage, df, desc, label, result):
    r = df[df.desc == desc].iloc[0]; base = df.iloc[0]; seen = json.loads(r.seen)
    isc = stage == "Combine"
    rows.append(dict(stage=stage, setting=label, result=result, dev_mean=round(r.dev_mean), kept=round(r.dev_mean / base.dev_mean, 2), dev_p25=round(r.dev_p25), dev_worst=round(r.dev_worst), green=int(r.green), accounts_paid=round(r.paid_share, 3),
                     pass_rate=round(r.pass_rate, 3) if isc else None, days_to_pass=round(r.days_to_pass) if isc else None, fees_per_funded=round(r.fees_per_funded) if isc else None,
                     died_share=None if isc else round(r.died_share, 3), payouts_per_paid=round(r.n_pay_per_paid, 2), paid_per_paid=round(r.pay_per_paid), days_farm=round(r.days_first_payout),
                     farms_2011_15=round(r.half1), farms_2016_20=round(r.half2), seen_2021=seen[0], seen_2023=seen[1], seen_2025=seen[2]))


add("Combine", C, "constant R = 9,600", "Today: constant risk unit R = 9,600", "Baseline")
add("Combine", C, "x0.5 -> x1 over 5 sessions", "Start at half size, full size after 5 sessions", "Closest on profit, bad-quartile farm much worse")
add("Combine", C, "down: x0.5 at +$3,000", "Halve the size once up $3,000", "Higher pass rate, slower, less profit")
add("Combine", C, "x1.5 -> x1 over 5 sessions", "Start at 1.5x, normal size after 5 sessions", "Less profit")
add("Combine", C, "constant x1.25 (R = 12,000)", "Constant 1.25x (R = 12,000)", "Constant size, for reference")
add("Combine", C, "last 10 days: x1.25 if up else x0.75", "x1.25 after a profitable 10 days, x0.75 after a losing 10", "Less profit")
add("Combine", C, "up: x2.0 at +$6,000", "Double the size once up $6,000", "Much less profit")
add("After payout", A, "constant a2 = 0.8", "Today: constant size 0.8 after the first payout", "Baseline")
add("After payout", A, "up: 0.4 -> 1.2 at $3,500", "Start at 0.4, step to 1.2 once the balance reaches $3,500", "Same profit, safer; fails the rule on mean by $305")
add("After payout", A, "up: 0.4 -> 2.4 at $2,000", "Start at 0.4, step to 2.4 once the balance reaches $2,000", "Less profit, safer")
add("After payout", A, "base 0.8, last 5 days: x1.25 if up else x0.75", "0.8 x1.25 after a profitable 5 days, x0.75 after a losing 5", "Less profit")
add("After payout", A, "after 2 payouts 1.2, after 4 payouts 1.6", "Bigger after 2 and 4 payouts (1.2, 1.6)", "Less profit")
add("After payout", A, "down: 1.6 -> 0.4 at $3,500", "Start 1.6, drop to 0.4 once the balance reaches $3,500", "Less profit")
add("After payout", A, "constant a2 = 0.6", "Constant 0.6", "Constant size, for reference")
add("After payout", A, "constant a2 = 1.0", "Constant 1.0", "Constant size, for reference")
json.dump(rows, open("results/dashboard_v4/scaling_stages.json", "w")); print(pd.DataFrame(rows)[["stage", "setting", "dev_mean", "kept", "dev_p25", "dev_worst", "died_share", "pass_rate"]].to_string(index=False))
