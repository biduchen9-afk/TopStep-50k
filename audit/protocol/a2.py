import pandas as pd
pd.set_option("display.width",250,"display.max_columns",40,"display.max_colwidth",60)
s=pd.read_csv("results/sleeve_sweep.csv")
cols=["market","sleeve","params","gate","sharpe_full","sharpe_dev","sharpe_hold","t_full"]
print("TOP by dev"); print(s.sort_values("sharpe_dev",ascending=False).head(15)[cols].round(2).to_string())
print("TOP by full"); print(s.sort_values("sharpe_full",ascending=False).head(15)[cols].round(2).to_string())
for q in [("es","vwap_atr","k=0.3,s=0.2","trend=250"),("gc","noise_area",None,"trend=60"),("nq","orb","or_min=60","none")]:
    d=s[(s.market==q[0])&(s.sleeve==q[1])&(s.gate==q[3])]
    if q[2]: d=d[d.params.str.contains(q[2])]
    print(d[cols].round(2).to_string())
# rank of selected among all by dev and by full
s["rk_dev"]=s.sharpe_dev.rank(ascending=False); s["rk_full"]=s.sharpe_full.rank(ascending=False); s["rk_hold"]=s.sharpe_hold.rank(ascending=False)
sel=[("es","vwap_atr","k=0.3,s=0.2","trend=250"),("gc","noise_area",None,"trend=60"),("nq","noise_area",None,"none"),("nq","vwap_atr","k=0.3,s=0.2","none"),("es","noise_area",None,"none"),("nq","orb","or_min=60","none")]
for q in sel:
    d=s[(s.market==q[0])&(s.sleeve==q[1])&(s.gate==q[3])]
    if q[2]: d=d[d.params.str.contains(q[2])]
    print(q, d[["rk_dev","rk_full","rk_hold"]].values.tolist())
print(pd.read_csv("results/portfolio_candidates.csv").round(2).to_string())
