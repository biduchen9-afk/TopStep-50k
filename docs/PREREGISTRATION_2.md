# Pre-registration 2 (written and committed BEFORE the tests below were run)

Data: existing NQ/ES/GC 1-minute data only (no paper trading, no new data). Selection uses sessions up to 2022-12-31 only. 2023+ is reported afterwards
and is already seen, so it is descriptive. Costs and rules as in `final_strategy.py` (NQ micro basket, R=9,600, hyst refill, no_activation_dll path).

## Part A - shorten the time to first payout (user goal: under 90 calendar days)
Metric (assumption, stated up front): calendar days from the day a Combine is started to the day the first payout becomes eligible (Topstep's 1-3 business-day
approval and transfer time is NOT included). Also reported: days from funded activation to first payout.
Grid (40 combinations, nothing else is tried): path in {standard, consistency}, first-payout trigger t1 in {500, 1000, 1500, 2000, 3000}, funded pre-payout
size a1 in {0.4, 0.8, 1.2, 2.0}; a2 = 1.6, w = 0.5 fixed.
Measured on dev Combine starts every 3 sessions (2011-2022): P(first payout | funded), median days Combine-start -> first payout among accounts that get paid,
share of ALL Combine starts that reach a payout within 90 calendar days, and the farm dev objective (0.5*(mean+p10) of net over 12 three-year dev windows).
Decision rule: a configuration is feasible if its dev objective >= $5,000 (about half of the current $10.2k). Among feasible ones choose the smallest median days.
Success = median <= 90 calendar days. If not reached, report the best feasible and say the goal is not achievable on this strategy without hurting expected value.

## Part B - one new idea: turn-of-month long sleeve
Spec (no free parameters): on the last trading session of each month and the first 3 sessions of the next month, hold 1 long micro from the 09:30 ET open to 15:55 ET,
no stop. NQ and ES only (2 tests). Costs as the other sleeves.
Accept into the portfolio only if ALL hold on dev data: sleeve Sharpe >= 0.5 (ATR-normalised daily R), correlation of its daily R with the final basket's daily R < 0.3,
and adding it with dev-only weights does not lower the dev farm objective. Otherwise reject. Hold-out numbers reported either way. No variants will be tried after seeing results.
