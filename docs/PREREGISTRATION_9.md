# Pre-registration 9 (committed BEFORE the runs): splits at FTMO and FundedNext, scaling pipeline, Kelly-style sizing, days to payout

User decisions: the Topstep default becomes "one sleeve per account group" (3 noise-area + 2 gap-VWAP accounts, one Combine open at a time, balanced sizing); check the best split at FTMO and FundedNext (FundedNext rules ignored by the user, who takes responsibility);
then scale into more accounts with the self-funded ladder and put the full pipeline on the dashboard; test whether Kelly-style sizing can raise payout profit without being greedy; report days to payout again. Existing data only; costs, RTH-only rule and dev-only selection (farms ending <= 2022-12-31) as before.

## Part A - best split per firm (dev selection)
Streams: noise-area only, VWAP+ATR only, gap-VWAP only, NQ basket (dev-only weights). Each firm uses its pre-registration-4 configuration (risk R, pre/post-payout size, first-payout balance) and its own caps (Topstep 5 funded, FundedNext 5, FTMO Growth 3), ONE Combine open at a time within the firm.
Candidate splits, 5-slot firms: same basket x5; 3 noise + 2 gap; 2 noise + 1 VWAP + 2 gap; 2 noise + 3 gap; 5 gap; 5 noise. 3-slot firm (FTMO): same basket x3; 2 noise + 1 gap; 1 noise + 2 gap; 3 gap; 3 noise; 1 noise + 1 VWAP + 1 gap. (Topstep's default is fixed by the user to 3 noise + 2 gap; the other five Topstep splits are reported for comparison.)
Rule: highest 25th-percentile net over ten 3-year dev farms (start years 2011..2020) among splits with mean net > 0; ties -> higher mean.

## Part B - the scaling pipeline
Order Topstep -> FundedNext -> FTMO Growth (dev net per fee dollar, pre-registration 4). Self-funding rule as before: out-of-pocket only for the first Topstep account (one account at a time until its first payout), then every purchase needs cash >= price + 3 billing cycles of monthly fees, a further firm unlocks only when cash >= threshold + price + reserve, one Combine open at a time across everything (it goes to the group with most empty slots). Threshold in {3,000, 6,000, 10,000}; chosen by dev mean over four long dev runs (starts 2012, 2014, 2016, 2018 to 2022-12) and nine 3-year farms, fewest ruins first. Seen starts (2021, 2023, 2025) and the full history are descriptive.

## Part C - Kelly-style sizing in the funded stage
Full-Kelly contracts on the cushion = cushion x mean / variance of the sleeve's daily dollar P&L per micro, with mean and variance estimated causally over the previous 120 sessions (size 1 micro when the trailing mean is not positive). f x Kelly is run for pre-payout f1 in {2, 3, 4} and post-payout f2 in {1, 2, 3} (9 configs), versus the current volatility rule (1.2 / 0.8 x cushion/ATR) and its neighbours 1.8 / 0.8 and 1.2 / 1.2, on the Topstep default split. Combine sizing is unchanged.
"Not greedy" acceptance: a configuration replaces the current one only if dev mean net AND dev p25 rise, the worst 3-year dev farm is not more than 25% deeper, and the share of funded accounts that are paid does not drop by more than 5 points.

## Part D - days to payout
For the new default and the old same-basket version: calendar days from the first Combine to the first payout of the farm (dev farms and seen starts), and per account (start to pass, activation to first payout) on the sleeves' sequential starts.
