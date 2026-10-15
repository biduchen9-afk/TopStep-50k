"""Dashboard data (v3): final strategy presets month by month + multi-firm comparison, ladder and payout-speed tables. Writes JSON files to argv[1]."""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from combine_policy import CombinePolicy
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from final_strategy import FINAL, PATH, PRESETS, build_stream
from firm_specs import specs
from pricing_config import load_config
from real_engine import FundedPlan, Stream, rules_from_config
from portfolio import DEV_END
from build_dashboard_data import monthly, finish, idx_first
from run_corrected import NEW, streams as old_streams

OUT = sys.argv[1]
cfg = load_config(); rules = rules_from_config(cfg, PATH)
s = build_stream(); d = pd.DatetimeIndex(s.dates)
mu = float(np.mean(s.pnl[(s.dates <= DEV_END) & s.ok] / s.atr_usd[(s.dates <= DEV_END) & s.ok]))
dmp = np.where(s.ok, s.pnl - mu * np.nan_to_num(s.atr_usd), s.pnl)
null = Stream(s.name, s.dates, dmp, np.minimum(s.low - (s.pnl - dmp), 0.0), s.atr_usd, s.ok)
idx = lambda dt: int(np.searchsorted(s.dates, np.datetime64(dt)))


def farm(stream, a1, t1, a2, R, i0, log=False, n_end=None):
    pl = FundedPlan("p", "standard", t1, 0.5, 1.0, 0.0, 0.5, 1.0, 99, a1, a2)
    return run_farm([stream], CombineSizing("fixed", R), None, pl, cfg, rules, PATH, i0, FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill="hyst"), n_end=n_end, log=log)


V = [("safe", "Safe: first payout at $4,000, size 0.4 (default)", "Slowest, most robust. Settings picked on 2010-22 data only.", s, FINAL["a1"], FINAL["t1"], 0.8),
     ("balanced", "Balanced: first payout at $2,000, size 1.2", "About 4x faster per account, more fees, weaker recent starts.", s, 1.2, 2000.0, 0.8),
     ("fast", "Fast: first payout at $6,000, size 2.4", "Fastest per account but only a third of funded accounts are paid.", s, 2.4, 6000.0, 0.8),
     ("control", "Edge removed (same path, mean edge set to zero), safe settings", "What the payout rules and volatility alone give.", null, FINAL["a1"], FINAL["t1"], 0.8)]
variants = []
for vid, label, note, st, a1, t1, a2 in V:
    r = farm(st, a1, t1, a2, 9600.0, idx_first(st), log=True)
    variants.append(dict(id=vid, label=label, note=note, months=finish(monthly(st, r, label)), totals=dict(net=r["net"])))
# previous headline for comparison
nq = old_streams(NEW)["NQ noise-area"]
r = run_farm([nq], CombineSizing("fixed", 6400.0), None, FundedPlan("p", "standard", 3000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.4, 1.6), cfg, rules, PATH, idx_first(nq), FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill="all"), log=True)
variants.append(dict(id="previous", label="Earlier headline (NQ noise-area only, size 1.6)", note="Shown for comparison.", months=finish(monthly(nq, r, "previous")), totals=dict(net=r["net"])))
mrows, vrows, yrows, srows = [], [], [], []
for v in variants:
    ms = pd.DataFrame(v["months"]); ms["month"] = ms.m + "-01"; ms["sample"] = np.where(ms.m <= "2022-12", "in-sample", "out-of-sample")
    for r_ in ms.to_dict("records"):
        mrows.append(dict(variant=v["id"], month=r_["month"], sample=r_["sample"], payouts=round(r_["payouts"], 2), payout_count=round(float(r_["payout_count"]), 2), fees=round(r_["fees"], 2), net=round(r_["net"], 2),
                          cum_net=round(r_["cum_net"], 2), cum_payouts=round(r_["cum_payouts"], 2), cum_fees=round(r_["cum_fees"], 2), evals_passed=round(float(r_["evals_passed"]), 2), evals_failed=round(float(r_["evals_failed"]), 2),
                          activations=round(float(r_["activations"]), 2), funded_died=round(float(r_["funded_died"]), 2), funded_closed=round(float(r_["funded_closed"]), 2), avg_evals=round(r_["avg_evals"], 2), avg_funded=round(r_["avg_funded"], 2)))
    pay, fee = float(ms.payouts.sum()), float(ms.fees.sum())
    vrows.append(dict(id=v["id"], label=v["label"], note=v["note"], net_total=round(pay - fee, 2), net_in_sample=round(float(ms[ms["sample"] == "in-sample"].net.sum()), 2), net_out_of_sample=round(float(ms[ms["sample"] == "out-of-sample"].net.sum()), 2),
                      payouts_total=round(pay, 2), fees_total=round(fee, 2), reinvest_pct=round(fee / pay, 4) if pay > 0 else 0.0, peak_outlay=round(float(-min(0.0, ms.cum_net.min())), 2),
                      funded_accounts=int(round(float(ms.activations.sum()))), payout_count=int(round(float(ms.payout_count.sum()))), months=int(len(ms))))
    ms["year"] = ms.m.str[:4]
    for y, g in ms.groupby("year"):
        yrows.append(dict(variant=v["id"], year=y, payouts=round(float(g.payouts.sum()), 2), fees=round(float(g.fees.sum()), 2), net=round(float(g.net.sum()), 2), payout_count=round(float(g.payout_count.sum()), 2),
                          activations=round(float(g.activations.sum()), 2), funded_died=round(float(g.funded_died.sum()), 2), avg_funded=round(float(g.avg_funded.mean()), 2), sample="in-sample" if y <= "2022" else "out-of-sample"))
starts = ["2012-01-03", "2015-01-02", "2018-01-02", "2019-01-02", "2021-01-04", "2023-01-02", "2024-01-02", "2025-01-02", "2025-07-01"]
for vid, label, note, st, a1, t1, a2 in V[:3]:
    for dt in starts:
        r = farm(st, a1, t1, a2, 9600.0, idx(dt))
        srows.append(dict(config=label, start=dt, net=round(r["net"], 2), payouts=round(r["payouts"], 2), fees=round(r["fees"], 2), n_payouts=int(r["n_payouts"]), funded=int(r["n_funded"])))
# firms
fc = pd.read_csv("results/firm_comparison.csv"); S = specs()
risk = {"Topstep 150K": "Reported: trading must come from your personal device, no VPS.", "FTMO Growth 150K": "Beta product; max 3 funded accounts; automation policy conflicting.",
        "FTMO Pro 150K": "Hard $2,000 daily loss limit also in the funded stage; max 3 funded accounts.", "FundedNext Flex 150K": "Official article bans copying between your own accounts; no bank payouts to Cambodia.",
        "Alpha Advanced 150K": "Bots banned; platform partner exited July 2026, payouts in batches.", "Alpha Standard 150K": "Bots banned; counterparty risk; unsure it is still sold.",
        "Take Profit Trader PRO 150K": "Copy trading and automation restricted; $4,500 buffer after every payout."}
frows = []
for _, x in fc.iterrows():
    seen = json.loads(x.seen_2021_2023_2025)
    frows.append(dict(firm=x.firm, config=f"R {int(x.R)}, size {x.a1}/{x.a2}, first payout ${int(x.t1)}", dev_mean=round(x.dev_mean, 0), dev_worst=round(x.dev_worst, 0), green_years=int(x.green), net_per_fee=round(x.net_per_fee, 2),
                      pass_rate=round(x.pass_rate, 3), days_to_pass=round(x.days_to_pass, 0), p_paid=round(x.p_paid, 3), days_activation_to_payout=round(x.days_act_to_payout, 0), full_net=round(x.full_net, 0),
                      seen_2021=seen[0], seen_2023=seen[1], seen_2025=seen[2], edge_removed=round(x.placebo_dev_mean, 0), rule_risk=risk[x.firm]))
# ladder
lu = pd.read_csv("results/ladder_unlock.csv")
lsum = lu.groupby(["mix", "unlock", "kind"]).agg(mean_net=("net", "mean"), min_net=("net", "min"), green=("net", lambda v: int((v > 0).sum())), runs=("net", "size"), ruined=("ruined", "sum"), oop_max=("oop", "max")).reset_index()
lsum["scope"] = np.where(lsum.unlock.astype(str) == "Topstep only", "Topstep only", "Ladder, cash threshold $" + lsum.unlock.astype(str))
lrows = [dict(mix=r.mix, scope=r.scope, kind=r.kind, mean_net=round(r.mean_net, 0), worst_net=round(r.min_net, 0), green=int(r.green), runs=int(r.runs), ruined=int(r.ruined), oop_max=round(r.oop_max, 0)) for r in lsum.itertuples()]
lseen = [dict(mix=r.mix, scope="Topstep only" if str(r.unlock) == "Topstep only" else f"Ladder, cash threshold ${r.unlock}", start=str(r.start), net=round(r.net, 0)) for r in lu[lu.kind == "seen to end"].itertuples()]
# speed
sp = [dict(preset="safe", label="Safe", size=0.4, first_payout_balance=4000, median_days=258, funded_paid=0.82, first_payout=1962, dev_mean=20119, worst_farm=-6965, fees_full=38805, net_full=134436, farm_days_dev=297, farm_days_2023=254, in90_dev=0.03, in90_2023=0.06),
      dict(preset="balanced", label="Balanced", size=1.2, first_payout_balance=2000, median_days=64, funded_paid=0.63, first_payout=1793, dev_mean=22662, worst_farm=-22806, fees_full=111042, net_full=129923, farm_days_dev=160, farm_days_2023=120, in90_dev=0.19, in90_2023=0.25),
      dict(preset="fast", label="Fast", size=2.4, first_payout_balance=6000, median_days=62, funded_paid=0.34, first_payout=3929, dev_mean=38085, worst_farm=-14232, fees_full=151041, net_full=193559, farm_days_dev=212, farm_days_2023=218, in90_dev=0.12, in90_2023=0.11)]
os.makedirs(OUT, exist_ok=True)
for name, data in (("monthly", mrows), ("variants", vrows), ("yearly", yrows), ("start_dates", srows), ("firms", frows), ("ladder_summary", lrows), ("ladder_seen", lseen), ("speed", sp)):
    json.dump(data, open(f"{OUT}/{name}.json", "w"))
    print(name, len(data), os.path.getsize(f"{OUT}/{name}.json") // 1024, "KB")
print(pd.DataFrame(vrows)[["id", "net_total", "net_out_of_sample", "payouts_total", "fees_total", "peak_outlay", "funded_accounts"]].to_string(index=False))
