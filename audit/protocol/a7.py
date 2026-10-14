import pandas as pd, numpy as np
pd.set_option("display.width",300,"display.max_columns",40,"display.max_colwidth",70)
d=pd.read_csv("results/dynamic_dev_grid.csv"); d["kind"]=np.where(d.plan.str.contains("dyn"),"dyn","fixed")
n=d[d.stream=="nq_noise"]
for k,g in n.groupby("kind"):
    print(k, len(g)); print(g.sort_values("ev_funded",ascending=False).head(3)[["combine","plan","p_first","payouts_per_funded","p_die_no_payout","extraction_per_funded","ev_funded","ev_lo","ev_hi","days_alive"]].round(2).to_string())
# restrict to same Combine sizing & t1 
print(n.groupby(["kind"])[["ev_funded","ev_lo"]].agg(["mean","max"]).round(0))
# how many configs in dyn have ev_lo>0
print((n.ev_lo>0).groupby(n.kind).mean())
