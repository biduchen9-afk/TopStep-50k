import pandas as pd
pd.set_option("display.width",250,"display.max_columns",40,"display.max_colwidth",60)
d=pd.read_csv("results/farm_dev_grid.csv")
x=d.sort_values("dev_net_3y",ascending=False)
print(x.groupby("streams").head(3).to_string())
n=d[(d.streams=="NQ noise-area")&(d.p_est==1.0)&(d.max_evals==10)]
print(n.round(0).to_string())
# what does dev say for a1=0.4,a2=0.4 NQ?
print(d[(d.streams=="NQ noise-area")&(d.a1==0.4)&(d.a2==0.4)].round(0).to_string())
h=pd.read_csv("results/holdout_variants.csv")
print(h.sort_values("dev_sharpe",ascending=False).head(8).to_string())
print(len(h))
