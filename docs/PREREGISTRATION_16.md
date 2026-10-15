# Pre-registration 16: re-test on corrected foundations (written 15 October 2026, before any selection run)

## Why
FINDINGS section 27 (audit) showed that the headline farm numbers rested on (a) a hindsight roll-skip filter and unadjusted splices, (b) accounts-paid and days-to-payout pooled 3:2 by attempts instead of at the farm's real account mix, (c) ten overlapping farms that contain only 51 distinct funded accounts, and (d) rule assumptions that had not been checked against Topstep's own pages. `docs/TOPSTEP_RULES.md` settles (d) except the XFA scaling ladder, the Tier 1 news-window cap, payout processing time and how a daily-loss-limit liquidation fills. This document fixes how the re-test is run before any config is scored.

## Foundations (fixed, not tested)
* **Data:** the repaired every-session NQ series (`repair_splices.py` -> `/home/user/data_raw_live`, splices back-adjusted, no sessions skipped for rolls); `TOPSTEP_RAW_DIR=/home/user/data_raw_live` for every run.
* **Rules v2** (`TOPSTEP_RULES=v2`, the default): everything in the engine as before plus "every payout after the first needs a positive net profit since the last payout".
* **Selection data:** 2010 to 2022-12-31 only. 2023-01-01 to 2026-07-02 is seen data: reported, never used to select. The forward test (PR15) is the only fresh data.
* **Strategy streams:** the three frozen sleeves, unchanged (nq_noise, nq_vwap, nq_gap30; the farm uses nq_noise and nq_gap30). Nothing about their entries or exits is re-tuned here.
* **Already seen before this document (measurement, not selection):** the baseline farm (3 noise-area + 2 gap-VWAP accounts, a1 1.2, first payout at a $2,000 balance, a2 0.8, R 9,600) on 247 overlapping 3-year dev farms (every 10 sessions): rules v1 mean $9,265, median $2,016, p25 -$2,840, p90 $36,270, 64% positive; rules v2 mean $12,719, median $2,859, p25 -$4,675, p90 $52,637, 60% positive. A rule change that only delays some payouts moved the mean by $3.5k, which shows how large the path noise is: **any difference in farm mean smaller than about $4k is not distinguishable from noise**, which is why farm means are not the selection metric.

## Measures
**Account level (primary).** One sleeve, Combine started every 3 sessions from the first tradable session + 260 to the end of dev - 400 (as `run_paid_share.Acc`), funded stage run to its end. Per start: Combine fees = $199 x ceil(sessions used / 21), payouts after the 90/10 split, `EV = payouts - fees`. Reported per group: pass rate, funded accounts, paid share, median days from Combine start to first payout among paid accounts, share of funded accounts paid within 90 days, payouts per funded account, mean EV per Combine start. Standard errors from a year-block bootstrap (blocks = calendar year of the Combine start, 2,000 resamples; paired for differences against the baseline). Farm-mix numbers weight the two groups by the funded accounts the baseline farm actually produces (measured once, then fixed), not 3:2.
**Farm level (safeguard and report).** 247 overlapping 3-year dev farms (every 10 sessions): mean, median, p10, p25, p75, p90, share positive, worst. Seen-period farms (starts 2021-01-04, 2023-01-02, 2025-01-02, 2025-07-01) reported only.

## Candidates (the whole list; nothing else is scored)
* **S1 sizing and trigger, per group (noise-area and gap-VWAP separately):** a1 in {0.6, 0.8, 1.0, 1.2, 1.5, 2.0} (size before the first payout), first-payout balance trigger t1 in {$1,500, $2,000, $3,000, $4,000}, a2 in {0.4, 0.8, 1.2} (size after), Combine risk R in {6,400, 9,600, 12,800}. 216 cells per group, 432 in all. Combine runs are computed once per (group, R).
* **S2 ex-ante risk filters, on the best S1 cell of each group:** skip the session (no trade) when (i) |09:30 open - previous close| > g x ATR14 with g in {1.0, 1.5, 2.0}; (ii) previous session range > r x ATR14 with r in {1.5, 2.0, 2.5}; (iii) 5-session / 60-session mean range ratio > v with v in {1.5, 2.0, 2.5}. Nine cells per group, 18 in all. Everything is known before the first possible entry.
* **S3 account split, with the S1/S2 winners:** noise-area x5, 4 + 1, 3 + 2 (baseline), 2 + 3, gap-VWAP x5. Farm level only.
Total scored: 455 configurations plus the baseline.

## Acceptance (all required, evaluated on dev only)
1. Mean EV per Combine start beats the baseline's by more than 2 paired standard errors, **and** by more than the largest gain the identical grid finds on the edge-removed placebo streams (the selection-noise allowance; placebo streams are the dev-fitted per-ATR mean removed, as in earlier pre-registrations).
2. The EV gain has the same sign in both dev halves (starts <= 2016 and 2017 to 2022).
3. Farm median and p25 are not lower than the baseline's, and the worst farm is not more than 25% deeper.
4. Funded accounts paid at the farm mix are not more than 5 points below the baseline's.
If several pass, the one with the highest p25 farm outcome is chosen; if none passes, the baseline stays. The result is reported as it falls out, whether or not it clears the user's targets.

## The user's two targets, reported separately
(a) The best EV subject to accounts paid >= 80% at the farm mix and median days to first payout <= 90; (b) the same with "within 90 days" as the share of funded accounts paid within 90 days. These are reported as a frontier from the same 455 scored configurations; they are not extra selection looks.

## Stress (on the baseline and the chosen config; not used to select)
Daily-loss-limit liquidation booked $100 and $250 beyond the limit (`TOPSTEP_DLL_SLIP`); XFA ladder flat at 30 micros up to a $3,000 balance; payout processing 5 days; price $229 without the daily-loss-limit discount; new entries blocked in the Tier 1 windows on FOMC days (13:55 to 14:05 and 14:25 to 14:35 ET; the date list is built from the data's own largest-range sessions only as a conservative proxy and is reported as such).

## Deliverables
`retest.py` (evaluator, account and farm level), `run_retest.py`, `results/retest_*.csv`, FINDINGS section 28, dashboard section, and an independent audit of `retest.py` by a separate reviewer before any number from it is quoted.

## Amendment A (15 October 2026, after the baseline measurement and before any grid cell was scored)
The baseline run showed that "EV per Combine start" is the wrong primary metric for a farm that buys one Combine at a time: groups and sizes differ in how long a Combine occupies the slot (noise-area 14.5 sessions on average, gap-VWAP 19.8). By the renewal-reward argument the farm's long-run profit is (sessions) x (EV per session of Combine), so **the primary account-level metric is `rate` = sum(payouts - fees) / sum(Combine sessions used) over the starts**, with the paired year-block bootstrap taken on the difference of rates. Everything else in the acceptance rules is unchanged, with `rate` in place of EV per start. Farm-mix numbers weight the groups by their share of Combine sessions in the baseline farm (starts share x mean Combine length: 75.8% noise-area, 24.2% gap-VWAP) and by their share of funded accounts for paid share and days (82.5% / 17.5%).
Also reported: `mature 3-year value` = rate x 756 sessions, i.e. what a farm that has been running long enough to be in steady state earns per 3 years. It is higher than the realised 3-year farm number because a farm that stops at 756 sessions drops payouts of accounts that are still alive (the window truncation of FINDINGS section 27, item 6); it is a reading of the economics, not a forecast, and the realised 3-year distribution stays the safeguard.
Baseline on the repaired data, rules v2 (measured, not selected): account level noise-area rate $24.8/session (EV $359 per Combine start, paid 51%, median 53 days), gap-VWAP rate $43.4/session (EV $860, paid 65%, median 106 days); farm mix rate $29.3/session = $22.1k per mature 3 years; funded accounts paid at the farm mix 53.8% (48.2% at farm level with unfinished accounts counted unpaid), median days to first payout 59, 37.7% of funded accounts paid within 90 days. The realised 3-year farm: mean $12.7k, median $2.9k.
