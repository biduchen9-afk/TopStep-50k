"""Dashboard data: why fewer funded accounts are paid, and the options for raising it (pre-registration 11). Reads results/paid_share_*.csv, writes results/dashboard_v4/paid_share.json."""
import json
import pandas as pd

D = pd.read_csv("results/paid_share_decomposition.csv"); B = pd.read_csv("results/paid_share_grid.csv"); C = pd.read_csv("results/paid_share_grid_c.csv"); E = pd.read_csv("results/paid_share_grid_d.csv")
cur = B[(B.a1 == 1.2) & (B.t1 == 2000) & (B.a2 == 0.8)].iloc[0]
rows = []
for r in D.itertuples():
    rows.append(dict(part="Where the drop came from", option=r.case, paid_pooled=round(r.acct_paid_share, 3), paid_noise=None, paid_gap=None, dev_mean=round(r.farm_dev_mean), dev_p25=round(r.farm_p25), dev_worst=None,
                     kept=round(r.farm_dev_mean / cur.dev_mean, 2), days_account=round(r.acct_days_to_payout), days_farm=round(r.farm_days_first_payout), seen_2021=None, seen_2023=None, seen_2025=None))


def add(part, label, r, noise=None, gap=None):
    seen = json.loads(r.seen)
    rows.append(dict(part=part, option=label, paid_pooled=round(r.acct_paid, 3), paid_noise=None if noise is None else round(noise, 3), paid_gap=None if gap is None else round(gap, 3), dev_mean=round(r.dev_mean), dev_p25=round(r.dev_p25),
                     dev_worst=round(r.dev_worst), kept=round(r.dev_mean / cur.dev_mean, 2), days_account=round(r.acct_days), days_farm=round(r.days_first_payout), seen_2021=seen[0], seen_2023=seen[1], seen_2025=seen[2]))


pick = lambda df, **kw: df[(df[list(kw)] == pd.Series(kw)).all(axis=1)].iloc[0]
add("Ways to raise it", "Current: all five accounts size 1.2, first payout at $2,000", cur, cur.paid_noise, cur.paid_gap)
c = pick(C, a1=1.2, t1=1000.0, w1=0.5); add("Ways to raise it", "Smaller first payout: same size, first payout at $1,000", c, c.paid_noise, c.paid_gap)
e = pick(E, noise_a1=1.2, noise_t1=1000.0, gap_a1=0.4); add("Ways to raise it", "Two tiers: noise-area accounts as current with first payout at $1,000, gap-VWAP accounts at size 0.4", e, e.paid_noise, e.paid_gap)
e2 = pick(E, noise_a1=1.2, noise_t1=1000.0, gap_a1=0.6); add("Ways to raise it", "Two tiers, milder: gap-VWAP accounts at size 0.6", e2, e2.paid_noise, e2.paid_gap)
c2 = pick(C, a1=0.8, t1=1500.0, w1=0.3); add("Ways to raise it", "Smaller size everywhere: size 0.8, first payout at $1,500 taking 30%", c2, c2.paid_noise, c2.paid_gap)
b = pick(B, a1=0.6, t1=2000.0, a2=0.8); add("Ways to raise it", "Smaller size everywhere: size 0.6, first payout at $2,000", b, b.paid_noise, b.paid_gap)
b = pick(B, a1=0.4, t1=4000.0, a2=1.2); add("Ways to raise it", "Old safe setting, best cell: size 0.4, first payout at $4,000, size 1.2 after it", b, b.paid_noise, b.paid_gap)
b = pick(B, a1=0.4, t1=2000.0, a2=0.8); add("Ways to raise it", "Smallest size: 0.4, first payout at $2,000", b, b.paid_noise, b.paid_gap)
json.dump(rows, open("results/dashboard_v4/paid_share.json", "w")); print(len(rows)); print(pd.DataFrame(rows)[["option", "paid_pooled", "dev_mean", "kept", "days_account"]].to_string(index=False))
