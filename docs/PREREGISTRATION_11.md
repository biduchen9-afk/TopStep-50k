# Pre-registration 11 (committed BEFORE the runs): why fewer funded accounts are paid now, and whether > 80% is reachable without losing much profit

Question (user): "What happened to the accounts paid? It was around 80% previous. Can we still get it above that without losing much profit?"
Known before running (FINDINGS section 15): the ~80% was the "safe" preset (size 0.4 x cushion / ATR before the first payout, first payout at balance $4,000): 82% of funded accounts ever paid, but 258 median days per account. The "balanced" preset the user chose as the default (size 1.2, first payout at $2,000) had 63% (64 days). The default split into single-strategy accounts is more volatile per account, and the farm-level share in section 20/21 (54%) counts accounts that were still running when a 3-year farm ended.

## Part A - where the drop came from (descriptive, no selection)
Per-account paid share (funded accounts that ever receive a payout / funded accounts; sequential Combine starts every 3 sessions, 2010-2022, dev only; pooled by slots: noise 3, gap 2) for: same basket x5 at safe; same basket x5 at balanced; default split (3 noise + 2 gap) at balanced; and each sleeve alone at balanced. Also the farm-level share (ten 3-year dev farms) for the same cases, to show how much of the 54% is the farm-window truncation.

## Part B - the grid (30 cells, all declared here)
Default split, one Combine at a time, all five accounts use the same sizing:
- size before the first payout a1 in {0.4, 0.6, 0.8, 1.0, 1.2} x cushion / ATR;
- first-payout balance t1 in {$2,000, $3,000, $4,000};
- size after the first payout a2 in {0.8, 1.2}.
It contains the current setting (1.2 / $2,000 / 0.8) and the old safe setting (0.4 / $4,000 / 0.8). No other cell is run for selection. Combine sizing (R = 9,600), costs, RTH-only trading and the dev-only rule are unchanged.

Metrics per cell. Account level (dev starts, same as Part A): pass rate, share of funded accounts ever paid (the headline), median calendar days start -> first payout, paid share per sleeve. Farm level (ten 3-year dev farms, start years 2011..2020): mean net, 25th percentile, worst farm, green years, fees, payouts per paid account, median days from the first Combine to the farm's first payout. Seen starts (2021, 2023, 2025-01, 2025-07) and full-history net are reported, not used for selection.

## Selection rule
Qualifying = account-level paid share >= 80% AND dev mean >= 85% of the current setting's dev mean AND dev p25 not lower than the current p25 minus $1,000 AND worst farm not more than 25% deeper. Among qualifying cells the highest dev mean is chosen; speed is NOT a gate but is reported next to the choice (a cell that gets there by waiting much longer must say so). If no cell qualifies, the answer is the frontier: the best paid share at each profit-retention level (100%, 90%, 85%, 75%).
"Above 80%" is read as >= 80% on the dev starts; with about 400 funded accounts per cell on overlapping starts the sampling error of a paid share is about +-3 points, so 80-83% is "about 80%".

## After the run (descriptive; do not select anything)
Edge-removed placebo for the chosen cell (or the closest if none qualifies); the chosen cell on seen starts and full history; stability of the paid share by sub-period (2010-2016 vs 2017-2022).

## Honest limits
30 cells of a grid on overlapping dev windows; the paid share of a cell is a dev estimate; recent years (2023+) have been seen many times; a lower pre-payout size lengthens the wait for the first payout (section 15) and that is the price to report, not hide.
