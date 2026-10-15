import pandas as pd, numpy as np
pd.set_option("display.width",300,"display.max_columns",40,"display.max_colwidth",70)
h=pd.read_csv("results/holdout_variants.csv")
print(h.groupby("market")[["dev_sharpe","hold_sharpe"]].median().round(2))
print("all median hold", h.hold_sharpe.median().round(2))
print("rank corr dev/hold", h.dev_sharpe.rank().corr(h.hold_sharpe.rank()).round(2))
print(h.groupby(["market","family"]).apply(lambda x: x.dev_sharpe.rank().corr(x.hold_sharpe.rank())).round(2).unstack())
print(pd.read_csv("results/holdout_results.csv")[["setup","p_pass","cost_funded","p_first","first_net","ev_funded","ev_lo","ev_hi","n_funded"]].round(1).to_string())
print(pd.read_csv("results/holdout_replay.csv").round(0).to_string())
sw=pd.read_csv("results/sweep_variants.csv"); print(len(sw))
print(pd.read_csv("results/final_slippage.csv").round(1).to_string())
print(pd.read_csv("results/final_improvements.csv").round(0).to_string())
for f in ["holdout_dev_grid","rinse_repeat_dev_grid","contract_dev_grid","farm_dev_grid","dynamic_dev_grid","sleeve_sweep","holdout_variants","noedge_contract","farm_robustness","farm_scaling","contract_robustness","portfolio_candidates"]:
    print(f, len(pd.read_csv(f"results/{f}.csv")))
