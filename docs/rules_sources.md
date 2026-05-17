# TopStep Rule Sources

The numeric parameters in `src/topstep50k/rules/topstep.py` come from
Topstep's published help-center articles. If any of these articles change,
the corresponding constants in `topstep.py` must be updated and the test
fixtures in `tests/unit/test_rules.py` regenerated.

## $50K Trading Combine — sources

| Parameter | Value | Source |
|---|---|---|
| Starting balance | $50,000 | Topstep Help: "Trading Combine Parameters" |
| Profit target | $3,000 | Topstep Help: "Trading Combine Parameters" |
| Max Loss Limit distance | $2,000 (line = $48,000 at start) | Topstep Help: "What is the Maximum Loss Limit?" |
| MLL trail mechanic | End-of-day, locks at starting balance | Topstep Help: "What is the Maximum Loss Limit?" |
| Daily Loss Limit | $1,000 | Topstep Help: "Daily Loss Limit in the Trading Combine" |
| DLL semantics | Auto-liquidate for session, NOT a Combine fail | Topstep Help: "Daily Loss Limit in the Trading Combine" |
| Max contracts (standard) | 5 per symbol | Topstep Help: "What is the Scaling Plan?" / max contracts breakdowns |
| Max contracts (micro) | 50 per symbol | Topstep Help: per-account contract limits |
| Consistency rule | Best winning day <= 50% of total cycle PnL | Topstep Help: consistency / "$150 Winning Day" article |

URLs (https://help.topstep.com):
* /en/articles/8284197-trading-combine-parameters
* /en/articles/8284204-what-is-the-maximum-loss-limit
* /en/articles/8284207-what-is-the-daily-loss-limit-and-what-happens-if-i-exceed-it
* /en/articles/8284223-what-is-the-scaling-plan

## What is intentionally NOT enforced yet

* **Minimum trading days for Combine pass** — Topstep documents this for
  Express Funded payouts; the Combine itself doesn't currently mandate a
  minimum count. Add only when confirmed.
* **News/event blackout** — not a Combine rule. Funded-account rules differ
  and should be encoded in a separate class when we add the funded sim.
* **Weekend / overnight holding** — allowed on Combine. Funded accounts
  have additional flatten-by-time-X rules; out of scope here.

## Verification protocol

1. WebFetch each URL above (or open in browser if Cloudflare blocks the
   automated fetch).
2. For each parameter in the table, confirm the article still asserts
   the same value.
3. If any value changed, update `topstep.py` constants, this doc, and
   tests in a single commit titled `rules: sync to <date>`.
