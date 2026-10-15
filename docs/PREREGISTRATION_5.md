# Pre-registration 5 (committed BEFORE the runs): shortening the wait for the first (small) payout

User complaint: the wait for a first, small payout is too long. Existing data only; final NQ basket; Topstep rules/fees as before (no_activation_dll, standard path); costs unchanged. Selection on dev only (farms ending <= 2022-12-31).

## Part A - Pareto frontier of speed vs value (Topstep)
Grid (60 configs, nothing else): pre-payout size a1 in {0.4, 0.8, 1.2, 1.6, 2.4}; first-payout balance t1 in {500, 1000, 1500, 2000, 3000, 4000}; fraction withdrawn at the FIRST payout w1 in {0.25, 0.5}.
Fixed: Combine R=9600, post-payout size a2=0.8, later payouts w2=0.5.
Metrics on dev Combine starts every 3 sessions (2011-2022): median calendar days Combine start -> first payout among paid accounts, share of ALL starts paid within 90/120/180 days, P(funded account ever paid),
mean first payout size; and the farm value: mean and 25th percentile of net over ten 3-year dev farms (starts 2011..2020).
Frontier: for each median-days budget B in {90, 120, 150, 210}, the configuration with the highest 25th-percentile net among those with median days <= B and dev mean net >= $5,000.
Reported as descriptive afterwards: 2023+ starts, and a by-start-year table (median days, P(paid)) for the unconstrained optimum and for the B=120 point, to show how much of the wait is regime.

## Part B - which firm should lead (the one funded out of pocket)
Lead firm in {Topstep 150K, FTMO Growth 150K, FundedNext Flex 150K, Take Profit Trader PRO 150K}, each with its Part-A (pre-registration 4) configuration, run alone under the self-funding rule
(one account at a time and out-of-pocket money until the first payout, then cash-gated). Metrics: calendar days from the first Combine to the first payout, out-of-pocket total, net,
over dev 3-year farms (starts 2012..2020) and seen starts (2021, 2023, 2025). No tuning in this part.
