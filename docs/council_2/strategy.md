# Strategy seat: is there any realistic edge or robustness left on NQ (and ES/GC)?

**Position.** No new strategy work is justified until forward data exists; the one free exception is shadow columns (VWAP, a frozen equal-weight basket) in the forward harness. Baskets, re-costed candidates, trend-day add-ons and volatility/event policies all land at or below the existing sleeves: the three NQ sleeves are only 1.3-1.7 independent bets, and ~5,000 prior looks put the deflated Sharpe bar (1.0-1.7) above everything found. A basket insures against picking the wrong sleeve (leader: gap-VWAP in dev, noise-area in 2023-24, VWAP in 2025) but adds no edge; chance a further intraday component adds 0.2 Sharpe and survives: 3-5% per idea (judgment).

## Evidence
Computed unless marked (read). Scripts: `/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/council2/strategy/` (`python3 replay.py`, then `s3`-`s20`). Check: my trade-level replay equals the repo sleeves' P&L and intraday lows bit-for-bit; my 3+2 farm reproduces FINDINGS 28/30b ($12.7k/$2.9k as modelled, $38.8k/$22.8k/$9.9k re-based).

1. **Baskets in one Combine** (R = 9,600, exact joint intraday low; the repo's sum-of-lows bound understates pass ~2 pts). Pass / EV per try:
   - Dev (800 starts): noise 21.2%/$289, VWAP 14.6%/-$19, gap 29.9%/$1,086; equal-weight three 20.1%/$196; inverse-vol 20.4%/$287; noise+gap 24.8%/$492; coin 13.0%/-$51 (SD across seeds $66).
   - 2023-24 (169): noise 32.5%/$1,035, VWAP 12.4%, gap 21.9%/$73; three 21.9%/$357; inverse-vol 24.9%; noise+gap 30.8%/$690; coin 15.6%/+$70 (SD $139).
   - Since 2024-12 (93 starts = 12-14 independent Combines, description only): 6.5/16.1/9.7%; three 19.4%; noise+gap 14.0%; coin 16-21% by seed.
   - A basket sits at the average of its parts (paired year-block SE 2-3 pts; dev: three minus noise -1.1, minus gap -9.8, minus coin +3.8). Re-based, same order (three 24.8%/$440, noise 25.0%/$593, gap 34.4%/$2,155). Sleeve pass rates by start year correlate 0.54-0.71 and the basket's year-to-year SD (0.14) equals a sleeve's: one regime factor. 54-95% of Combines touching +$9,000 are blocked by the 55% rule, basket or not.
   - **Independent bets:** 1.3 (variance, equal risk) to 1.7 (eigenvalue) for the three. With ES and GC 2.1-2.9, but ES legs cut the equal-risk Sharpe 1.16 to 1.04; GC (dev Sharpe 0.13) only lowers volatility (NQ 3/4 + GC 1/4: pass 19.6%, paid 82% vs 53%). Engine sensitivity (biased-coin proxy, `s20`): +0.25 Sharpe is about +1 pt pass, +$100 EV; a basket's +10% Sharpe is ~0.4 pt.
   - Farm (247 dev farms, re-based): 5 x three-sleeve basket $21.7k/$20.9k/p25 $2.6k vs 3+2 $38.8k/$22.8k/$9.9k (mean -$17k [-32k,-5k]); 5 x (noise+gap) $32.1k/$30.4k/$16.0k, worst -$1.0k, not distinguishable (median +$3.7k [-7.2k,+10.8k]). Hindsight: starts 2024-12..07, first year, 3+2 -$4.4k vs $0.0k (VWAP carried it); 2023-24 starts $7.6k vs $2.0k.
2. **D1 re-score** (70 NQ trials of the 144-list, R_rec = G - f(G-N)): median Sharpe 0.00 to 0.22; >=0.5: 5 to 19; >=1.0: 2 (both noise-area gates, equal to the base 1.01). ES/GC re-score (Council 1, read): max 0.73. Cross-trial SD 0.47 gives best-of-N noise bars 1.27 (N=164) and 1.74 (N=5,000); sampling-only 1.04. Nothing new clears (the sleeves are 1.01/0.86/1.30). Best new: ML gap30 0.95 (hold-out 0.48).
3. **D3** (all 12 configs, untuned): add at +0.75 ATR: 180 dev add-ons (14/yr), mean -0.017 R (SE 0.036); at +1.25 ATR: 41, -0.009 (SE 0.095); unconditional-timing placebo 0.00. Deflated t = 3.4 needs +0.12 / +0.32 R each. 15/10-minute decisions: dev Sharpe 0.46 to 0.30/0.17 as modelled, 1.01/1.01/0.98 re-based. No variant gains more than 0.01.
4. **Volatility/event.** After a prior range >= 1.5 ATR (12% of days) the three-basket earns -0.010 R/day vs +0.023 (t -2.0; hold-out -0.031 vs +0.025): opposite to the clustering prior, and the top-quintile sign flips between 2010-16 and 2017-22. 08:30 shock days (10%): +0.032 vs +0.017 (t 0.8; hold-out +0.016 vs +0.019). Trading only those days stretches a Combine to 23-145 sessions, fees $0.3-1.5k, EV -$300 to -$1,090 (one dev exception, +$818, is -$469 in hold-out). Priors (memory, unverified): clustering moves the range ATR sizing absorbs; Gao et al. 2018: stronger momentum on news days; pre-FOMC drift (Lucca-Moench 2015) ~8 days a year, reported faded.

## Recommended next steps (ranked by plausibility x testability x cost)
1. **Keep 3+2; no new sleeve research.** Cost 0; chance a further idea helps 3-5%.
2. **Shadow columns** (daily R of VWAP and of a frozen equal-weight three-basket, no orders) in the forward harness: ~20 lines, no money; the only way to learn whether baskets insure.
3. **Only if the chair wants insurance:** pre-register 5 accounts x 50/50 noise+gap on the split menu. P(truly better than 3+2) ~40%, median gain <$5k. Drop on the first PR18 STOP.

Forward reality: Sharpe SE ~1.2/sqrt(years), so two years give +-0.85; telling 20% from 15% pass needs ~340 independent Combines; PR18 shows no STOP by session 250 in 34% of zero-edge paths (read): survival says almost nothing.

## What not to do
More NQ variants or grids; ES ports (corr 0.4-0.7); event-only Combines (fees bill per 21 sessions); adding a sleeve because 2025 favoured it.

## Biggest thing the chair may be wrong about
That more sleeves mean more independent bets, or that re-costing strengthens the evidence: it lifts all candidates equally, and after deflation only gap-VWAP (1.30) clears even the sampling-only bar. The 2025 record fits zero edge and full edge alike.

## Questions for the user
1. Are free shadow columns allowed in the forward harness?
2. Should 5 x (noise+gap) join the frozen split menu?
3. Do you accept "stop on the first PR18 STOP" as the kill rule?
