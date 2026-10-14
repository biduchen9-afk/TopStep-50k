# Pre-registration 3 (written and committed BEFORE the grid below was run)

Question from the user: "wait longer -> safer and bigger payout; scale it; change the first-payout balance freely; best end value, green in all years, no leakage or false numbers."
Data: existing NQ/ES/GC data only. Final NQ basket (dev-only weights), Combine R=9,600 (fixed, fixed because R>=16,000 is a cliff), hyst refill, no_activation_dll path,
standard payout path, w=0.5. Selection uses ONLY farms that end on or before 2022-12-31.

Grid (84 combinations, nothing else is tried): first-payout balance t1 in {1500, 2000, 3000, 4000, 5000, 6000, 8000}; pre-payout size a1 in {0.2, 0.4, 0.8};
post-payout size a2 in {0.4, 0.8, 1.6, 2.4}.
Evaluation: ten 3-year farms started on the first session of each year 2011..2020 (all end by 2022-12-31). For each: net = payouts - all fees.
Descriptive columns: P(first payout | funded), mean size of the first payout, mean total paid per funded account, share of paid accounts with a 2nd payout, median days start->first payout.
Selection rule ("green in all years" as a robust objective, since a literal all-green requirement may be infeasible): maximise the 25th percentile of net across the ten start years,
subject to mean net >= $10,000 and at least 6 of 10 start years positive; ties -> higher mean. If no configuration is feasible, say so and report the best by p25.
Reported afterwards, descriptive only (already seen): farm starts 2021-01, 2022-01, 2023-01, 2024-01, 2025-01, 2025-07 through the end of the data, and a coin-flip no-edge control.
Leakage checks to run on the winner: (1) an independent reviewer re-derives the net from the reference engine, (2) shift the stream by one session (a deliberately broken alignment) and confirm the result degrades,
(3) confirm all weights/scales were fitted on <= 2022-12-31.
