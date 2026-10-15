# Pre-registration 7 (committed BEFORE the runs): ES / GC ideas from SSRN and arXiv, regular trading hours only

Source: two literature searches (SSRN, arXiv, 2026-10-14; abstracts and snippets only, nothing verified). Existing data only. Costs $1.22 per micro round turn + 1 tick. Selection on 2010-2022.
**Overnight rule (user): nothing is traded or held outside 09:30-15:55 ET.** Every sleeve enters at or after 09:30 ET and exits by 15:55 ET; `audit_trade_hours.py` checks this on the trades actually taken, and the 08:20 ET COMEX-anchored ideas are not allowed.
Already tried in pre-registration 6 (60 trials: NQ ports, prior-day breakout, overnight-range breakout, gap fade; none accepted). This round adds 44 trials, so ES/GC will have had 104 looks in total (best-of-104 pure-noise Sharpe is about 0.7).

## Trials (44, nothing else will be run)
- L1 Intraday momentum, last half hour (Gao-Han-Li-Zhou; Baltussen-Da-Lammers-Martens): signal in {Gao: return prior close to 10:00 plus 13:00-13:30 return; Baltussen: return prior close to the window start}, filter in {none, high-volatility days (5/60-session range ratio >= 1)}, window in {15:25-15:55 (ES, GC); COMEX 13:00-13:30 (GC only)}. ES 4 + GC 8 = 12 trials.
- L2 Opening-move reversal (Fung et al.; Grant et al.): fade the 09:30-10:00 move when it is >= k ATR, k in {0.3, 0.6}, exit at noon or 15:55, stop 1.0 ATR. ES and GC: 2 x 2 x 2 = 8.
- L3 Lunch drift (Vojtko-Dujava, SSRN 4934614), ES only: long 11:00-14:00 (a) unconditionally, (b) only if 09:30-11:00 was down. 2 trials.
- L4 5-minute opening-candle breakout (Zarattini-Aziz, SSRN 4416622): trade the direction of the 09:30-09:35 candle at 09:35, stop at the other end of the candle, target 10R or none (exit 15:55). ES and GC: 4 trials.
- L5 Opening-range breakout only in high-volatility states (Holmberg et al.): 15- or 30-minute range from 09:30, 0.1 ATR buffer, stop 0.5 ATR, trade only when the 5/60-session range ratio >= 1. ES and GC: 4 trials.
- L6 Same-time-of-day continuation (Heston-Korajczyk-Sadka): trade a half-hour slot (10:00, 14:00 or 15:00 ET) in the direction of its mean return over the previous 10 sessions. ES and GC: 6 trials.
- L7 Cross-asset (exploratory), GC only: trade GC 10:00-11:00 in the same or opposite direction of the ES or NQ return over 09:30-10:00, stop 1.0 ATR. 4 trials.
- L8 Narrow-range day breakout (Crabel): when the previous session range is the narrowest of the last N in {4, 7}, enter when price closes beyond the 09:30 open +- 0.5 ATR, stop 1.0 ATR, exit 15:55. ES and GC: 4 trials.

## Acceptance (same as pre-registration 6)
Best-dev configuration per (market, idea). Accepted only if dev Sharpe >= 0.5 (ATR-normalised daily R), positive in >= 60% of dev calendar years, and hold-out (2023+) Sharpe > 0. Dev Sharpe < 0.7 is labelled "weak evidence". At most the 3 best accepted sleeves join the pool.
If a sleeve is accepted, the three lists of pre-registration 6 are re-run once with it added to the diversified basket; nothing else is tuned.
Reading of the papers' known decay: the pre-publication years (2010-2015) may look better than 2023+; the hold-out check is deliberately kept.
