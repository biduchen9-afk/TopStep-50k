# TopStep $50K Trading Combine — Rule Sources

The numeric parameters in `src/topstep50k/rules/topstep.py` come from
Topstep's published help-center articles and the topstep.com blog.
This document is the source of truth: if any of these articles change,
the corresponding constants in `topstep.py` and the test fixtures in
`tests/unit/test_rules.py` must be updated in a single commit titled
`rules: sync to <date>`.

## Authoritative URLs

| Topic | URL |
|---|---|
| Combine parameters | https://help.topstep.com/en/articles/8284197-trading-combine-parameters |
| Maximum Loss Limit | https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit |
| Daily Loss Limit | https://help.topstep.com/en/articles/10490293-daily-loss-limit-in-the-trading-combine-and-express-funded-account |
| Scaling Plan | https://help.topstep.com/en/articles/8284223-what-is-the-scaling-plan |
| Consistency | https://help.topstep.com/en/articles/8284208-consistency-at-topstep |
| Drawdown blog | https://www.topstep.com/blog/prop-firm-drawdown-rules/ |

## $50K Combine — current values (verified 2026-05)

| Parameter | Value | Notes |
|---|---|---|
| Starting balance | $50,000 | — |
| Profit target | $3,000 | EOD equity at start + 3,000 |
| Max Loss Limit distance | $2,000 (initial line = $48,000) | See "MLL mechanic" below |
| MLL trail mechanic | **End-of-day**, ratchets on max EOD balance, locks at starting balance | topstep.com blog explicitly says "End-of-Day Drawdown" |
| Daily Loss Limit | **Not enforced on the Combine** | See "DLL on the Combine" below |
| Max contracts | 5 standard or 50 micros (account-wide) | 10:1 micro:standard ratio; mixed positions allowed |
| Consistency rule | Best winning day <= 50% of total cycle PnL | A violation HOLDS the pass, doesn't fail it -- see "Consistency mechanic" below |
| Minimum trading days | None on the Combine | XFA / Funded payout has different rules |

## MLL mechanic — clarification

Topstep's own blog (`topstep.com/blog/prop-firm-drawdown-rules`) states
that the firm "uses End-of-Day Drawdown, which mirrors real market
conditions." On the Combine and the Express Funded Account, the MLL:

  1. Starts at `starting_balance - distance` ($48,000 for $50K).
  2. Trails the highest END-OF-DAY balance, never decreasing.
  3. Locks at the starting balance once the trail reaches it.

This matches the `TrailingMLL` state machine in `topstep.py`. Some
third-party guides confuse the Combine MLL with intraday-trailing
mechanics used by other prop firms; we follow the topstep.com primary
source.

## Consistency mechanic — clarification (corrected 2026-08-26)

**A consistency violation does NOT fail the Combine.** This project's
simulator originally treated hitting the profit target with one day
over 50% of total profit as a terminal failure (`consistency_fail`
ending the attempt right there) -- that was wrong, and it fed every
pass30/pass45 number computed before this date.

Per Topstep's own guidance (corroborated by several independent 2026
third-party breakdowns of the same help-center article): "Topstep does
not fail you when you violate consistency. Instead, your effective
profit target increases so that your best day represents less than 50%
of the new total. You must then earn additional profits to bring the
ratio into compliance... the remaining profit must be earned on a
separate trading day, after the market close of the session in which
your best day was set." In other words: the pass is *held*, not
*denied* -- you keep trading (still fully exposed to MLL risk, which
does not pause just because you nominally hit $3,000) until a later
day's profit dilutes the best-day share back to <=50%.

`simulate_combine_window` and `simulate_sequential_accounts`
(`analysis/passrate.py`) now model this correctly: reaching the target
does not stop the day-by-day loop unless consistency is ALSO satisfied
that same day. The `consistency_fail` outcome label is kept (for
backward compatibility with existing dict/report consumers) but its
meaning changed: it now means "the window/account ran out of time while
still pending dilution" -- not "disqualified". A longer window, or
continuing the same account past a fixed 30/45-day observation window,
would very plausibly resolve many of these into real passes.

## DLL on the Combine

**The TopStep Trading Combine does NOT enforce a Daily Loss Limit**
(verified with the account owner, 2026-05). The `combine_50k()`
preset therefore sets `daily_loss_limit = None`, and the engine's
`check_daily_loss` returns `None` whenever the limit is unset.

The field is retained on `TopstepRules` so callers can opt in to DLL
enforcement for two real scenarios:

  1. A **personal DLL** configured by the trader as a self-imposed
     risk cap (Topstep supports this via Trailing Personal DLL).
  2. Some Funded-account configurations and certain front-end
     platforms (NinjaTrader / Tradovate / Quantower / TradingView)
     can still impose a DLL at the platform layer.

To model either, override the preset:

    from dataclasses import replace
    rules = replace(combine_50k(), daily_loss_limit=Decimal("1000"))

The pass-rate measurement and ORB strategy operate against the
default (no-DLL) Combine rules; the configurable path is exercised
in tests so the code path doesn't bit-rot.

## Scaling Plan vs Combine

The Scaling Plan (2 -> 3 -> 5 contracts at $1,500 / $2,000 profit
thresholds) is an **Express Funded Account** rule, NOT a Combine rule.
The $50K Combine has the flat 5-standard / 50-micro cap from day one.
Our preset `combine_50k()` encodes the Combine cap; a separate `xfa_*`
preset would be needed for the funded account.

## What is intentionally NOT enforced yet

* **News/event blackout** — not a Combine rule. Funded accounts apply
  flatten-by-news rules separately.
* **$150 winning day** — counts toward minimum trading days for
  Express Funded payouts. Not relevant to passing the Combine itself.
* **Dynamic Live Risk Expansion** — a topstep.com feature article
  exists; not part of the standard $50K Combine evaluation.

## Overnight / swing holding — CORRECTED 2026-08-26 (was wrong)

**No overnight or weekend holding is allowed at ANY stage** — Combine,
Express Funded Account, or Live Funded. All positions must be flat by
**3:10 PM CT every weekday** (risk managers start flattening at 3:08 PM
CT; Topstep's automated Trade Desk force-flattens ~10 seconds before
3:10 PM CT if a position is still open; staying flat in time is the
trader's own responsibility). Trading resumes at 5:00 PM CT weekdays,
Sundays after the weekend break. Rules confirmed current as of a
July 1, 2026 update, applied uniformly across all three account
stages.

This directly overturns what this file previously said in the
"intentionally not enforced" list below ("Weekend / overnight holding
— allowed on Combine") — that line was based on May 2026 research and
was simply wrong, or went stale with the same rules refresh that
changed the payout caps. **`OvernightDrift` (the `OD` strategy stream)
is therefore not viable for this project at all** — not just
unmodeled, actually disallowed — and has been retired from the live
ensemble. See `src/topstep50k/strategy/overnight_drift.py`'s module
docstring and `scripts/evaluate_ensemble_databento_recent_evgate.py`
for the retirement note.

| Topic | URL |
|---|---|
| Live Funded Account Rules | https://www.topstep.com/live-funded-account-rules |
| Express Funded Account Parameters | https://help.topstep.com/en/articles/8284215-express-funded-account-parameters |

## Express Funded Account (XFA) — added 2026-08-26

`src/topstep50k/rules/topstep_xfa.py` and `src/topstep50k/rules/rebill.py`.
**help.topstep.com is blocked from this sandbox's network policy** (same
constraint as the Combine rules above) -- these numbers come from
WebSearch snippets of Topstep's help-center articles and cross-checked
against 4-5 independent third-party 2026 aggregator sites, NOT a direct
primary-source fetch. Confirm directly with Topstep before relying on
any of this for a real payout decision, especially the two flagged
below.

| Topic | URL |
|---|---|
| Payout Policy | https://help.topstep.com/en/articles/8284233-topstep-payout-policy |
| Consistency at Topstep | https://help.topstep.com/en/articles/8284208-consistency-at-topstep |
| What is a Reset? | https://help.topstep.com/en/articles/8284128-what-is-a-reset |
| Express Funded Account consistency (blog) | https://www.topstep.com/blog/topstep-express-funded-account-consistency |
| Scaling Plan | https://help.topstep.com/en/articles/8284223-what-is-the-scaling-plan |

### $50K XFA — current values (as reported post-April-28-2026 change)

| Parameter | Value | Notes |
|---|---|---|
| Trailing MLL | Same mechanic as Combine: $2,000 distance, locks at starting balance | Reused conceptually; see PostPayoutDrawdown for the payout-event divergence |
| Profit split | 90% trader / 10% Topstep | Consistent across multiple sources |
| Standard payout path | >= 5 winning days (net PnL >= $150 each, non-consecutive OK) | Cap $2,000/request |
| Consistency payout path | >= 3 trading days AND largest-day / total profit <= 40% | Cap $3,000/request |
| Payout cap (either path) | min(path cap, 50% of current balance) | Per-request, not per-account/month |
| **⚠ Grandfathering** | Accounts purchased BEFORE 2026-04-28 keep their OLD (higher, ~$5-6K) caps on rebill and on Reset Credit | This module's `xfa_50k()` preset assumes a NEW (post-4/28) account. Pass explicit caps if the real account predates the change. |
| **⚠ Post-payout drawdown reset** | Reported: MLL cushion collapses to $0 the instant a payout clears (floor jumps to equal the post-payout balance) | Modeled conservatively in `PostPayoutDrawdown` -- see that class's docstring for exactly what was assumed vs. sourced. **This is the single highest-stakes unverified number in this codebase; confirm with Topstep before trusting it for a live payout-timing decision.** |

### Reset Credits / rebill lifecycle

| Parameter | Value |
|---|---|
| Rebill interval | ~30 days |
| Reset Credit earned per rebill | 1, added to a per-account-size-and-type bank |
| Redeeming a credit | Free; performs a Reset (wipes Combine progress to starting conditions); also pushes the rebill date out 30 days |
| Credit expiry | Credits issued before 2025-12-11: never expire. Issued 2025-12-11 or later: expire 1 year after issue. |
| Redemption limit | 2 per account per day |
| Reset-while-in-profit | If the account was in profit before a Reset and no trades were placed after, Topstep Support MAY (discretionary, not automatic) adjust the new profit target downward. No fixed dollar figure found for this in any source checked. |

### What this explicitly does NOT model

* The discretionary post-reset profit-target adjustment (no formula
  found; would need a direct answer from Topstep support to encode).
* Whether the Standard/Consistency payout PATH is a one-time account-
  level choice made at funding, or freely re-evaluated per payout
  request. Sources were ambiguous; `XFARules.eligible_paths()` reports
  both if both currently qualify and lets the caller decide.

## Verification protocol

1. WebFetch each URL above. (If the sandbox network policy blocks the
   help.topstep.com host, use WebSearch with a query targeting
   topstep.com to retrieve excerpts; cross-check at least two
   independent third-party sources from the same year.)
2. For each parameter in the table above, confirm the article still
   asserts the same value.
3. If any value changed, update `topstep.py` constants, this doc, and
   tests in a single commit titled `rules: sync to <date>`.

Last verification: 2026-05-17 (via WebSearch; help.topstep.com is
denied at the sandbox network policy level so direct WebFetch was not
possible -- the verification used the topstep.com blog and two
independent third-party 2026 references as cross-checks).
