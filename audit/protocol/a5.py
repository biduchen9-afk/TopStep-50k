import pandas as pd
pd.set_option("display.width",300,"display.max_columns",40,"display.max_colwidth",70)
print(pd.read_csv("results/contract_holdout.csv").round(1).T.to_string())
n=pd.read_csv("results/noedge_contract.csv")
print(len(n)); print(n.describe().T.round(0)[["mean","min","max"]])
print((n.noedge_dev_3y>0).sum(),(n.noedge_hold>0).sum())
print(pd.read_csv("results/contract_robustness.csv").round(0).to_string())
print(pd.read_csv("results/rinse_repeat_holdout.csv").round(1).T.to_string())
