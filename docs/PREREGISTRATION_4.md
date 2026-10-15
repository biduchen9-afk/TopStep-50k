# Pre-registration 4 (committed BEFORE the multi-firm runs)

Data: existing NQ/ES/GC data only; the same final NQ basket stream (Topstep cost model) is used at every firm (assumption). Selection uses farms ending on or before 2022-12-31.

## Part A - one standalone farm per firm variant (7 variants: Topstep 150K, FTMO Growth/Pro 150K, FundedNext Flex 150K, Alpha Advanced/Standard 150K, Take Profit Trader PRO 150K)
Grid per firm (identical structure for all): Combine/evaluation risk R in {6400, 9600, 12800}; pre-payout size a1 in {0.4, 0.8}; post-payout size a2 in {0.4, 0.8, 1.6};
first-payout balance t1 from a small firm-specific list (3 values; see `firm_specs.py`). Farm = hyst refill, p_est 1.0, firm's own funded-account cap.
Evaluation: ten 3-year farms started on the first session of each year 2011..2020 (all end by 2022-12-31). Selection rule per firm: highest 25th percentile of net across the ten
start years among configurations with mean net > 0; ties -> higher mean. A firm whose best configuration has mean net <= 0 is "not viable on this strategy".
Reported afterwards: Combine/evaluation pass rate, calendar days to pass and calendar days from activation to first payout (shown separately), P(funded account ever paid),
dev mean / p25 / green years, full-history net, seen starts 2021, 2023, 2025, and an edge-removed placebo (the dev-fitted mean edge subtracted) on dev.
Every firm rule is taken from web-search snippets and is UNVERIFIED; flagged rule risks (bot bans, copy-trading bans, counterparty risk) are reported next to the numbers and are NOT modelled.

## Part B - cash-gated ladder (user's rule)
Out-of-pocket money only for the lead firm (Topstep) until the FIRST payout from any firm is received, with k0 = 1 evaluation at a time. After that no new money is injected:
a purchase needs cash >= price + 3 billing cycles of the monthly fees of all open evaluations including the new one, where cash = payouts received (after a 2-session delay) minus all fees.
A firm may start only after the previous firm in the order has received a payout. Order: Topstep first, then the viable firms by dev mean net per fee dollar (computed in Part A).
Strategy mix: (A) the same basket on every account; (B) different sleeve groups per firm (NQ-trend, NQ gap-VWAP, ES+GC) assigned round-robin by firm order.
Reported over farm starts 2012-01..2020-01 (dev) and 2021/2023/2025 (seen): out-of-pocket total, min cash, shortfall days, days to first payout from the first Combine,
net, payouts and fees by firm, and the green-year count.
