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
