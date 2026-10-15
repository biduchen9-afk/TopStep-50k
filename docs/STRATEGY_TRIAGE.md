# Triage of the user's 60-strategy / 50-model list (2026-10-14)

Constraints in force: nothing is traded or held outside 09:30-15:55 ET (prop-firm rule); data = 1-minute NQ, ES, GC (2010-2026); costs $1.22 per micro round turn + 1 tick; selection on 2010-2022 only; every trial is counted.
Note: the NQ VWAP+ATR sleeve was not from a paper (agreed); it is one of ~100 NQ configurations tried, and the dev-only selection already treats it as a searched parameter, not as evidence.

## Strategies
**A. Intraday-native, testable as written (flat by 15:55 ET):** 11 ORB + gap + trend filter; 17 level + candle + stochastic (prior-day low, intraday bars); 18 gap fade; 22 midpoint magnet; 30 first-15-minute momentum; 31 open-drive continuation; 32 lunch fade;
33 close-only momentum (exit 15:55 instead of settlement); 34 volatility-window filter; 35 scheduled-time stand-down (overlay; only the 09:55-10:10 and 13:55-14:15 windows are inside regular hours).
**B. Daily-bar signals that need a swing hold; testable only as (i) a regime filter on an intraday sleeve or (ii) an intraday-only adaptation (signal at the daily close, open to 15:55 trade next session):**
1-7 trend combinations; 8 squeeze ignition; 9 NR7 + momentum; 10 ATR-percentile Donchian; 12 two-day sweep reclaim; 13 prior-day break continuation; 14 consolidation box break; 15-16 and 19-21 mean-reversion combinations; 23 range-expansion continuation; 24 HV-percentile router;
25 BB-width expansion; 26 post-expansion re-entry; 27 ATR-adaptive channel; 28 vol-targeted trend; 36-39 and 41 regime routing / equity-curve gate. Expect the intraday-only versions to be much weaker than the written swing versions.
**C. Not allowed or not testable here:** 29 Asia-range London break (trades the overnight session); 40 weekly/daily Donchian multi-day holds; 42-47 cross-market pairs, cointegration and cross-sectional ranking (multi-day legs; most of the instrument universe and sector proxies are not in our data); 54 time stop beyond the session.
**D. Overlays / evaluation layer (engine features, not signals):** 48 fixed-fractional sizing; 49 daily-loss breaker (the no_activation_dll path already liquidates at -$3,000); 50 consistency cap; 51 trailing-drawdown governor (cushion-proportional sizing already does this); 52 breakeven + chandelier ladder; 53 trade-count limiter;
55 end-of-day flat (already 15:55); 56 pass-probability optimizer (the R grid); 57 minimum-day scheduler; 58 buffer lock; 59 scaling-plan mapper (done); 60 multi-account parameter diversification (partly the three lists).

## Models
Already used or equivalent: 1 ATR, 33 volatility targeting, 36-39 rule state machine (the engine itself), 44 walk-forward (done at farm level), 13-14 ER/ADX features (partly).
Worth adding as VALIDATION tools: 45 deflated Sharpe, 46 block-bootstrap significance, 41 bootstrap expectancy intervals, 36-38 Monte Carlo trade reshuffling through the rule state machine (pass / breach probability), 42 loss-streak distribution.
Worth adding as FEATURES/GATES (cheap): 13 ADX, 14 efficiency ratio, 15 Hurst, 16 variance ratio, 10/31 time-of-day profiles, 2-6 range-based volatility estimators, 11-12 vol-of-vol and term structure.
ML (47-49) and the regime-switching models (17-20) are heavy on data-mining risk; they would need the same counted-trial discipline.
