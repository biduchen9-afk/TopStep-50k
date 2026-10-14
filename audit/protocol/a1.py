import pandas as pd
pd.set_option("display.width",250,"display.max_columns",40,"display.max_colwidth",60)
d=pd.read_csv("results/dynamic_dev_grid.csv")
print(len(d), d.stream.unique())
d["kind"]=d.plan.str.contains("dyn").map({True:"dyn",False:"fixed"})
print(d.groupby(["stream","kind"])[["p_first","payouts_per_funded","p_die_no_payout","extraction_per_funded","ev_funded","n_funded"]].mean().round(2))
n=d[d.stream=="nq_noise"]
print(n.groupby("kind").size())
print(n.groupby(["combine","kind"])[["p_pass","p_first","payouts_per_funded","p_die_no_payout"]].mean().round(2))
print(pd.read_csv("results/dynamic_holdout_metrics.csv").T)
print(pd.read_csv("results/dynamic_holdout_replay.csv").T)
