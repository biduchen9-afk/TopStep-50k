# Pre-registration 12 (committed BEFORE the runs): scaling the funded risk size before the first payout

User decisions (answers to the questions asked first): scale the size in the funded account BEFORE the first payout; triggers to test = profit milestones, recent performance, time ramp (the drawdown guard was not chosen); judge by "accounts paid >= 80% AND profit near current" (the open problem of pre-registration 11, where no constant size reached it).

## What is already scaling, and what is new
Today the funded size is k = round(a x cushion / ATR) with cushion = balance - MLL floor and a = 1.2 before the first payout, 0.8 after it. Because the Topstep MLL trails the PEAK balance (it locks at $0 only when the peak passes $4,500, which is above every first-payout balance tested), the cushion before the first payout is $4,500 minus the current drawdown from the peak: it already shrinks in a drawdown but does NOT grow with profit. So the size does not currently react to profit, to recent results or to time. The schemes below make `a` a function of the account's own state (causal, using only the previous sessions' results), for the stage before the first payout only. After the first payout the size is the unchanged 0.8 x cushion / ATR; Combine sizing (R = 9,600), the first payout rule ($2,000 balance, 50%), costs, RTH-only, and the dev-only rule are unchanged. Default split (3 noise-area + 2 gap-VWAP accounts, one Combine open at a time); the same scheme on all five accounts.

## Engine
`FundedPlan.scale` (default None = today's behaviour, tested to be identical) with three families, all applied only while `stage == 1`:
- `("ladder", ((m1, a1'), ...))`: start at `a1`; while the CURRENT balance is >= m1 use a1', and so on (not sticky: if the balance falls back below a milestone the size steps back down).
- `("perf", L, f_up, f_down)`: a = a1 x f_up if the account's realised P&L over its last L sessions is > 0, a1 x f_down otherwise; a1 x 1 until L sessions exist.
- `("ramp", d, a_start)`: a rises linearly from a_start to a1 over the first d sessions after activation.

## Grid (36 cells, all declared; a2 = 0.8, first payout $2,000 x 50%)
A. Profit ladder, 16 cells: low size in {0.4, 0.8} x high size in {1.2, 2.0} x milestone balance in {$500, $1,000} x direction in {up: start low, step up to high at the milestone; down: start high, step down to low at the milestone}.
B. Recent performance, 8 cells: base a1 in {0.8, 1.2} x L in {5, 10} x (f_up, f_down) in {(1.5, 0.5), (1.25, 0.75)}.
C. Time ramp, 12 cells: a_start in {0.4, 0.6} x end size a1 in {1.2, 1.6} x ramp length d in {10, 20, 40} sessions.
The current setting (constant 1.2) is the baseline and is re-run in the same script (must reproduce $17,486 mean, 61.5% paid).

## Metrics and selection rule (same as pre-registration 11)
Account level (sequential Combine starts every 3 sessions, 2010-2022, pooled noise x3 + gap x2): share of funded accounts ever paid (headline), median days start -> first payout; per group. Farm level (ten 3-year dev farms, 2011-2020 starts): mean net, p25, worst, green years, payouts per paid account, median days to the farm's first payout. Seen starts (2021, 2023, 2025-01, 2025-07) and full history are reported, not used.
Qualifying = account-level paid share >= 80% AND dev mean >= 85% of the baseline's AND dev p25 not lower than baseline minus $1,000 AND worst farm not more than 25% deeper. Highest dev mean among qualifying is chosen. If none qualifies, the answer is the frontier (best paid share at each profit-retention level 100/90/85/75/60%) and the closest cell is named. Speed is not a gate but is reported (a cell that gets paid share by waiting longer must show it).

## After the run (descriptive)
Edge-removed placebo for the chosen or closest cell; paid share for 2010-16 vs 2017-22 starts; a one-line mechanism check (how many sessions the account spends at each size and what share of deaths happen at which size) if a cell comes close.

## Honest limits
36 more looks on the same overlapping dev windows (96 declared looks for the "paid share vs profit" question in total with pre-registration 11); the family choices were suggested by the user, the parameter values by me. Even a qualifying cell is "plausible, not proven" and would need a frozen forward test.

## Round 2 - added AFTER round 1 was seen (declared here, committed before it is run)
Round 1 (36 cells + baseline) result in one line: no cell qualified. Families B (recent performance) and C (time ramp) did not raise the paid share without cutting profit (best B: 74% paid but 31% of the profit; C: at most 67%). The profit ladder that STARTS SMALL and steps UP at a balance milestone ("up") was the only family with both a high paid share and a high profit: up 0.4 -> 1.2 at $1,000: 82.4% paid, mean farm $10.6k (61% of baseline), 192 days per account; up 0.4 -> 2.0 at $1,000: 75.7% paid, $32.1k (184%), but p25 -$3.5k (baseline -$1.6k) and 235 days per account; up 0.8 -> 2.0 at $500: 62% paid, $30.7k, 105 days. The "down" ladders (start big, step down) lowered profit with little paid-share gain. Round 2 therefore refines the UP ladder only, around the frontier between those cells.
Grid (30 cells, all declared; a2 = 0.8, first payout $2,000 x 50%, same on all five accounts):
- L1, two-step up ladder, 24 cells: start size in {0.4, 0.5} x top size in {1.6, 2.0, 2.4} x milestone balance in {$750, $1,000, $1,250, $1,500}.
- L2, three-step up ladder, 6 cells (start -> middle at m1 -> top at m2): (0.4 -> 1.2 at $500 -> 2.0 at $1,000), (0.4 -> 1.2 at $750 -> 2.0 at $1,250), (0.4 -> 1.0 at $500 -> 1.6 at $1,000), (0.4 -> 1.2 at $500 -> 1.6 at $1,000), (0.5 -> 1.0 at $500 -> 2.0 at $1,250), (0.5 -> 1.2 at $750 -> 2.4 at $1,500).
Same metrics and the SAME selection rule (qualifying = paid >= 80% AND mean >= 85% of baseline AND p25 >= baseline - $1,000 AND worst not > 25% deeper; highest mean among qualifying; otherwise the frontier). Looks so far for this question: 60 (pre-registration 11) + 36 + 30 = 126.
Because a winner would be picked from this many cells, any cell that qualifies gets extra checks before it is called anything: (a) it must also hold in both halves of the dev farms (start years 2011-2015 and 2016-2020: mean above the baseline's in each half) and in both halves of the account-level starts (2010-16 and 2017-22: paid share >= 75% in each); (b) the edge-removed placebo must stay negative; (c) the seen starts (2021, 2023, 2025) are reported next to the baseline; (d) the days to the first payout are stated next to the headline, since the starts-small ladders reach their paid share by waiting longer.
