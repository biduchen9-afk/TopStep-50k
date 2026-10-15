"""Per-funded-account stats for the final stream: Combine pass rate, P(>=1 payout), payouts per funded account, mean paid, days to first payout (starts every 3 sessions; descriptive)."""
import numpy as np, pandas as pd
from final_strategy import build_stream, FINAL, PATH
from pricing_config import load_config
from real_engine import rules_from_config, run_combine, run_funded_detail, FundedPlan
cfg=load_config(); rules=rules_from_config(cfg,PATH)
s=build_stream(); d=pd.DatetimeIndex(s.dates); x=s.pnl.tolist(); lo=s.low.tolist(); atr=s.atr_usd.tolist(); nd=s.desired_micros(FINAL["R"])
res={}
for a2 in (1.6,0.4):
    pl=FundedPlan("p","standard",3000.0,0.5,1.0,0.0,0.5,1.0,99,0.4,a2)
    rows=[]; first=int(np.flatnonzero(s.ok)[0])
    for i0 in range(first,len(d)-300,3):
        st,tau=run_combine(x,lo,nd,i0,rules)
        n=0
        row=dict(y=d[i0].year,passed=int(st==1),paid=0,npay=0,gross=0.0,days=np.nan)
        if st==1:
            tm=[]; pays,days,state,_=run_funded_detail(x,lo,nd,i0+tau,rules,pl,atr=atr,times=tm)
            row.update(paid=int(len(pays)>0),npay=len(pays),gross=sum(pays),days=(d[min(tm[0],len(d)-1)]-d[min(i0+tau,len(d)-1)]).days if tm else np.nan)
        rows.append(row)
    df=pd.DataFrame(rows); f=df[df.passed==1]
    print(f"a2={a2}: Combine pass {df.passed.mean():.2f}; of funded accounts: P(>=1 payout) {f.paid.mean():.2f}, mean payouts per funded {f.npay.mean():.2f}, mean total paid per funded ${f.gross.mean():,.0f}, median cal days activation->first payout {f.days.median():.0f}")
    g=f.groupby(f.y>=2023); print("   funded->payout by period (2010-22 / 2023+):", g.paid.mean().round(2).to_dict(), "| mean total paid per funded:", g.gross.mean().round(0).to_dict())
