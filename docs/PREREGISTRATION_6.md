# Pre-registration 6 (committed BEFORE the runs): ES / GC components, one-at-a-time Combine buying, and big / moderate / conservative lists

User decisions: balanced is the default funded preset; Combines are bought ONE AT A TIME (buy one, buy the next only when it resolves; keep going until the funded slots are full);
lists are defined BOTH by components and by sizing preset; the five funded slots are split 1 big / 2 moderate / 2 conservative; ES and GC are searched for new components with a fixed trial budget.
Existing data only (NQ/ES/GC 1-minute). Costs as before ($1.22 per micro round turn + 1 tick). Selection on 2010-2022 only.

## Part A - ES / GC component search: 60 trials in total, nothing else will be tried
Ports (30 trials = 15 per market, ES and GC):
- noise-area momentum with trend gate in {none, 60, 250}  (3)
- VWAP+ATR momentum, k in {0.2, 0.3, 0.5} x stop s in {0.2, 0.4}  (6)
- the same VWAP+ATR but only when overnight gap and first-30-minute move agree ("gap30")  (6)
New ideas (30 trials):
- N1 prior-day-range breakout (ES and GC): close beyond yesterday's high/low +- buf*ATR after anchor+30 min, buf in {0, 0.1} x stop in {0.3, 0.6}  (4 per market = 8)
- N2 overnight-range breakout at the market's open (ES anchor 09:30 ET, GC anchor 08:20 ET COMEX open): close beyond the 18:00-to-anchor range +- buf*ATR after anchor+15 min, buf in {0, 0.1} x stop in {0.4, 0.8}  (4 per market = 8)
- N3 COMEX-open opening-range breakout (GC only, anchor 08:20 ET): opening range 15/30/60 min x stop in {0.5, 1.0}  (6)
- N4 overnight-gap fade at 09:30 ET (ES and GC), gap >= thr*ATR with thr in {0.5, 1.0} x stop in {0.5, 1.0}  (4 per market = 8)
All flat by 15:55 ET, signals at a bar close, fills at the next open.
Expected best-of-60 Sharpe under pure noise is about 0.65 (daily-return SE about 0.29 over 13 dev years), so a dev Sharpe near 0.4 is NOT evidence.
Acceptance (all three): dev Sharpe >= 0.5 (ATR-normalised daily R, 2010-2022), positive in >= 60% of the dev calendar years, hold-out (2023+) Sharpe > 0. Accepted sleeves with dev Sharpe < 0.7 are labelled "weak evidence".
Only the best-dev configuration per (market, idea) can be accepted. At most the 3 best accepted new sleeves join the pool.

## Part B - lists (tiers) and the five funded slots
Pool = the NQ sleeves already in use (noise-area, VWAP+ATR, gap30) plus accepted ES/GC sleeves.
- Conservative list (2 slots): safe sizing (pre-payout 0.4, first payout at $4,000) on the most diversified basket: all pool components with dev-only weights.
- Moderate list (2 slots): balanced sizing (1.2, $2,000) on the NQ basket (noise-area + VWAP+ATR + gap30), as now.
- Big-risk list (1 slot): fast sizing (2.4, $6,000) on the single pool component with the highest dev Sharpe.
Combine risk R = 9,600, post-payout size 0.8 for every list. Each list has its own pipeline with ONE open Combine at a time and its own funded slots (1 / 2 / 2).
Comparisons (same Topstep rules and fees, no_activation_dll): (i) all five slots balanced on the NQ basket with one-at-a-time buying (the new default), (ii) the three lists together.
Metrics: dev 3-year farms (start years 2011..2020): mean, worst, number of green start years, fees, correlation of the lists' monthly net; months with at least one payout; days from first Combine to first payout; and seen starts 2021, 2023, 2025 and the full history (descriptive).
Classification of what the current NQ strategy is: by outcome at each sizing preset - conservative if >= 75% of funded accounts get paid, moderate if 50-75%, big risk if < 50% - plus the worst 3-year farm.
