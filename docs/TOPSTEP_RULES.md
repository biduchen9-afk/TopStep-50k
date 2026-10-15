# Topstep 150K rules: what Topstep's own pages say, against what the engine models

Checked 15 October 2026. The sandbox cannot open `help.topstep.com` or `www.topstep.com` (network policy: host not allowed), so every line below is what a web search returned from Topstep's own help-centre articles. That is first-party text but second-hand reading. Where the search summaries conflict, the line says so. To make this first-hand, allow those two hosts in the environment's network settings and the pages can be read directly.

Sources (Topstep help centre): Payout Policy (8284233), Express Funded Account Parameters (8284215), Trading Combine Parameters (8284197), Daily Loss Limit in the Combine and XFA (10490293), What is the Maximum Loss Limit (8284204), Consistency at Topstep (8284208), Risk Adjustments: High Risk/High Volatility (13613539), Responsible Trading Program (13620045), Pricing and Payment Questions (14289835), Express Funded Account Activation (8284217), What is the Scaling Plan (8284223).

| Rule | Topstep's page (as returned) | Engine | Status |
|---|---|---|---|
| Combine 150K target / loss limit | $9,000 target, $4,500 maximum loss limit, 15 minis = 150 micros | same | confirmed |
| Maximum loss limit | trails the end-of-day high, locks permanently at $0 (the starting balance); monitored in real time on realised and unrealised P&L; breach liquidates | intraday breach, lock at $0 | confirmed (one Topstep blog says end-of-day only; the engine takes the stricter reading) |
| Combine consistency | best day <= 55% of total profit; if exceeded the target rises, no rounding, no buffer | 55% | confirmed (the old 50% sensitivity is not the rule) |
| Daily loss limit | optional at purchase for Combine and XFA, $3,000 on 150K, chosen at purchase and carried into the XFA, cannot be changed; a hit liquidates and blocks trading for the rest of the session, not an account breach | soft limit at -$3,000 in both stages | confirmed; the "no daily loss limit in the funded stage" sensitivity is therefore not applicable if the limit is bought at checkout (it lowers the price, below) |
| Price with the limit | No Activation Fee path $229/month less a $30 "Responsible Trading Discount" for adding the limit = $199, activation fee $0. Standard path $199/month plus $149 per XFA | $199 per 21 sessions, no activation fee | confirmed; one product page shows $149/month, treated as stale |
| Payout, Standard path | 5 winning days of >= $150 net, not consecutive; the day a payout is requested does not count towards the next five; request up to 50% of the balance, cap $5,000 (Consistency path $6,000), minimum $125; **net profit since the last payout must be positive for every payout after the first** | all but the last clause | **gap found and fixed (rules v2)** |
| After a payout | maximum loss limit resets to $0 permanently after each payout (one FAQ says after the first only); the 5-day count restarts | floor to $0 after the first payout and every one | confirmed (current Payout Policy) |
| Payout cap with a limit | limited-time offer: adding a daily loss limit at checkout doubles the cap to $10,000 / $12,000, new purchases only | $5,000, no doubling | conservative; not used |
| Consistency path | 3 trading days with a trade, best day <= 40% of net profit, resets per payout window | tested before (slower, 47% paid) | unchanged |
| Profit split | 90/10 | 90% | not re-read this time |
| Payout processing | "payout timing can be 3-5 days"; no firm figure | 3 calendar days | unresolved, small |
| Scaling plan (150K XFA) | the page says the limit follows the balance; the 3/4/5/10/15-mini ladder at $0/1,500/2,000/3,000/4,500 appears only in the Topstep Labs article, whose product differs | same ladder | **unresolved** |
| Activation | immediate; a passed Combine stays "Passed" until activated; the XFA trades from the next market open; up to 5 active XFAs; Combine profit does not carry over | next session, slot cap 5, <= 21-session wait | conservative |
| Live Funded Account | by invitation after consistent XFA performance, no count given | not modelled (accounts are let to die) | n/a |
| Tier 1 news windows | for CPI, FOMC and NFP-type events: no new opening trades in mini equity indexes and a cap of 9 micros (150K) for a window of 5 minutes either side (CPI 08:25-08:35 ET, FOMC decision 13:55-14:05 ET, press conference 14:25-14:35 ET; NFP window not given) | not modelled | **gap**: affects new entries on those days only; to be quantified |
| Responsible Trading Program | discretionary placement; forces the Consistency path, daily loss limit, $6,000 cap | not modelled | n/a unless Topstep places the account |

## What this settles from the audit's sensitivity list
The audit worried that the sign of the result depended on five assumptions. Against Topstep's pages: the maximum-loss-limit lock at $0, the 55% Combine consistency, the 150-micro cap, the payout cap and the payout-day count are as modelled; the daily loss limit in the funded stage is available and is what the $199 price assumes. The remaining open items are the XFA scaling ladder, the news-window cap, the payout processing time, and how a liquidation by the daily loss limit fills (the engine books it exactly at -$3,000; `TOPSTEP_DLL_SLIP` books it beyond).

## Engine changes (rules v2)
* `net_pos_after_first` (`FundedSpec`, `firm_engine.topstep_150k(rules="v2")` by default, env `TOPSTEP_RULES=v1` reproduces everything up to FINDINGS section 27; mirrored in `paper/accounts.py`).
* `dll_slip` sensitivity (env `TOPSTEP_DLL_SLIP`, dollars beyond the limit, default 0).
