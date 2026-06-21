# Audit findings and bug fix — 2026-06-21

## Summary

Full look-ahead audit + rules correctness audit conducted across all
strategy, regime, engine, and rules files. One real bug found and fixed.
No look-ahead bias found anywhere.

---

## Bug fixed: passrate.py — simulation continued past profit target

**File:** `src/topstep50k/analysis/passrate.py`, line 127

**The bug:** `simulate_combine_window` did not `break` out of the
day-loop when the profit target was first reached. The simulation
continued processing post-target days, which:
1. Could turn a legitimate `pass` into `mll_breach` if a large loss
   occurred after the target was already hit.
2. Inflated the denominator of the consistency check (best_day / total)
   by including post-target PnL, making consistency easier to pass.

**The fix:** Added `break` immediately after recording `passed_target_on`.
A real Combine terminates the moment the target is reached — post-target
days must not be processed.

**Impact on reported results:**

| metric | old (buggy) | new (correct) | direction |
|:--|---:|---:|:--|
| $50K pass45 multi-asset | 44.2% | **49.0%** | ↑ +4.8pp |
| $50K pass30 multi-asset | 30.6% | **33.3%** | ↑ +2.7pp |
| $25K pass45 multi-asset | 45.2% | **28.5%** | ↓ -16.7pp |
| $25K pass30 multi-asset | 36.2% | **25.3%** | ↓ -10.9pp |

**Why $50K improved:** Post-target MLL breaches (wrongly counted as
failures) now correctly counted as passes. 37 fewer false MLL breaches.

**Why $25K dropped dramatically:** With a $1,500 target and ensemble
stdev of $796/day, many windows hit the target in 3-7 days. Pre-fix,
the loop continued through all 30 days, diluting best-day / total
(making consistency easy). Post-fix, the check uses only days through
the target day — and with a high-variance strategy, one early day
frequently exceeds $750 (50% of $1,500). The $25K results in all prior
docs were inflated by this bug. The pipeline is significantly better
suited to the $50K Combine than the $25K Combine.

**Regression tests added:**
- `test_post_target_mll_breach_does_not_override_pass`
- `test_post_target_days_excluded_from_consistency`

---

## Bug fixed: overnight_drift.py — entry_filter not marked as rejected

**File:** `src/topstep50k/strategy/overnight_drift.py`, lines 116-124

**The bug:** When `entry_filter` returned False, the code returned
`None` without marking the state as "this cycle is skipped." On every
subsequent bar in the entry window (15:55-16:00 ET, ~5 bars for 1-min
data), the filter was re-queried unnecessarily. Since the filter is
deterministic (same date → same answer), there was no incorrect behavior
— just 4 redundant filter calls per skipped overnight cycle.

**The fix:** Added `entry_filter_rejected: bool = False` to `_ODState`.
Set to `True` when filter rejects. Reset to `False` on each day-roll.
The entry block now short-circuits when `entry_filter_rejected` is True.

---

## Look-ahead audit: CLEAN (no violations found)

All three strategy implementations (ORB, MeanRev, OvernightDrift),
all three regime gates, the backtesting engine, and the data loaders
were audited. No temporal look-ahead bias found anywhere.

### Known limitations (NOT bugs, NOT look-ahead)

1. **Stop/TP levels reference signal-bar close, not actual fill price.**
   `orb.py:234` and `mean_reversion.py:227` set `entry_price = bar.close`
   for stop/TP distance calculation. The actual fill occurs at the NEXT
   bar's open. If a gap exists between bar.close and next-bar.open, the
   effective stop distance differs from the coded value. This is a
   price-proxy approximation — no future data is used — but it means
   stop levels can be slightly mis-calibrated. Fixing this requires a
   fill callback to the strategy, which is an architectural change.
   Tagged for future implementation.

2. **Engine fill timestamps are one-bar late in the audit log.**
   `engine/backtest.py`: fills execute at `bar.open` before
   `clock.advance_to(bar.ts)` is called. So `fill_ts = clock.now()`
   records the previous bar's close time rather than the current bar's
   open time. **P&L is computed correctly (fill price = bar.open is
   correct).** Only the audit log timestamp is off. Tagged for future
   fix.

---

## Rules correctness audit: CLEAN

- `TrailingMLL.update_end_of_day()`: correct trail-and-lock logic.
- `simulate_combine_window()`: MLL check ordering (check then update)
  is correct — see above for the separate `break` bug.
- Consistency check with zero/negative total: correctly returns `None`.
- `combine_25k()` parameter values: all correct.
- Sliding window indexing: no off-by-one errors.

---

## Corrected baseline result

The corrected headline for the multi-asset ensemble (9 streams,
pass-rate-aware weights, proper TopStep rules):

  **$50K Combine: pass30 = 33.3%, pass45 = 49.0%**
  **$25K Combine: pass30 = 25.3%, pass45 = 28.5%**
  Sharpe +1.66, MaxDD -$12,840 (unchanged — no strategy code changed)

All prior docs reporting $25K pass-rates above 35% are retroactively
incorrect due to the passrate.py bug. The $50K numbers in prior docs
are slightly conservative (actual is higher than reported).
