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
| Consistency rule | Best winning day <= 50% of total cycle PnL | Evaluated at end-of-combine only |
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
* **Weekend / overnight holding** — allowed on Combine. Express Funded
  has additional flatten-by-time-X rules; out of scope here.
* **$150 winning day** — counts toward minimum trading days for
  Express Funded payouts. Not relevant to passing the Combine itself.
* **Dynamic Live Risk Expansion** — a topstep.com feature article
  exists; not part of the standard $50K Combine evaluation.

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
