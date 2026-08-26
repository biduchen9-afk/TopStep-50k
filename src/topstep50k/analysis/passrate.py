"""Realized Combine pass-rate over a long backtest.

The acceptance criterion the user set: a "challenge-ready" strategy
must show > 50% Combine pass-rate when its 4-year daily-PnL series is
diced into 30-45 day windows (the typical pass-window). This module is
the instrument that produces that number.

Two related primitives:

  * `simulate_combine_window(daily_pnl_window, rules, starting_balance)`
    -- runs the strategy's daily PnL series through the Combine rule
    machine (trailing MLL, profit target, consistency) and reports
    pass / fail / breach-reason.
  * `realized_pass_rate(daily_pnl, rules, starting_balance, ...)` --
    slides a fixed-length window across the daily-PnL series with a
    configurable stride and counts how many windows passed.

The simulator is deliberately separate from `analysis/bootstrap.py`
(which resamples blocks). Here we want the LITERAL historical
pass-rate, not a resampled estimate -- "if a trader had started a
Combine on day t, would they have passed by day t+W?"

No look-ahead: each window only consumes daily PnLs whose dates fall
strictly within [start, end]. The MLL state is initialised fresh per
window (each Combine starts from $0 drawdown). Days outside the window
are ignored.

References: TopStep rules per docs/rules_sources.md.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Iterable, Sequence

from topstep50k.rules.topstep import TopstepRules, TrailingMLL


@dataclass(frozen=True)
class CombineWindowResult:
    """Outcome of running one Combine cycle over a daily-PnL window."""

    start_day: date
    end_day: date
    n_days: int
    outcome: str  # 'pass' | 'mll_breach' | 'consistency_fail' | 'no_target'
    final_pnl: Decimal
    days_to_outcome: int  # -1 if outcome is 'no_target' (ran to end)
    breach_day: date | None = None


@dataclass(frozen=True)
class RealizedPassRate:
    n_windows: int
    n_passed: int
    pass_rate: float
    outcomes: dict[str, int]  # outcome -> count
    window_results: list[CombineWindowResult] = field(default_factory=list)
    window_days: int = 0
    stride_days: int = 0

    @property
    def ci_95_low(self) -> float:
        """Wilson 95% lower bound on the pass-rate."""
        return _wilson_ci(self.n_passed, self.n_windows, 1.96)[0]

    @property
    def ci_95_high(self) -> float:
        return _wilson_ci(self.n_passed, self.n_windows, 1.96)[1]


def _wilson_ci(k: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    rad_sq = p * (1 - p) / n + z * z / (4 * n * n)
    rad = (rad_sq ** 0.5) * z / denom
    return max(0.0, centre - rad), min(1.0, centre + rad)


def simulate_combine_window(
    daily_pnl: Sequence[tuple[date, Decimal]],
    *,
    rules: TopstepRules,
    starting_balance: Decimal,
) -> CombineWindowResult:
    """Run one Combine cycle over an ordered daily-PnL series.

    Stops on the first MLL breach -- that's a genuine, immediate failure.
    A consistency violation is NOT a failure: per Topstep's own guidance
    ("Consistency at Topstep" help article; corroborated by third-party
    2026 breakdowns -- see docs/rules_sources.md), hitting the profit
    target with one day representing more than 50% of total profit does
    not fail the Combine. It just means the pass is held: the effective
    target rises (to 2x the current best day) until later days dilute the
    best-day share back under 50%. So this does NOT stop the day loop the
    moment the raw target is first crossed -- it keeps evaluating day by
    day (still fully subject to MLL risk, which does not go away just
    because you nominally hit $3,000) until EITHER an MLL breach occurs,
    OR a day's end finds target reached AND consistency satisfied
    simultaneously (a real pass), OR the window runs out first.

    Outcome 'consistency_fail' (legacy name, kept for backward
    compatibility with existing outcome-key consumers) means: the window
    ended with the profit target reached at some point, but the running
    best-day share never resolved under 50% before the window ran out --
    NOT that the account was disqualified. A longer window very plausibly
    resolves it into a clean pass; this is a "ran out of window", not a
    "failed the Combine".

    `daily_pnl` is a sequence of (date, decimal_pnl) tuples, ascending.
    """
    if not daily_pnl:
        raise ValueError("empty daily_pnl window")
    days_sorted = sorted(daily_pnl, key=lambda x: x[0])
    start_day = days_sorted[0][0]
    end_day = days_sorted[-1][0]
    n_days = len(days_sorted)

    mll = TrailingMLL(starting_balance, rules.max_loss_limit_distance)
    eq = starting_balance
    cumulative: dict[date, Decimal] = {}
    ever_reached_target = False

    for i, (day, pnl) in enumerate(days_sorted):
        eq = eq + pnl
        cumulative[day] = pnl
        # MLL is evaluated against the EOD balance after the trail has
        # ratcheted from the PREVIOUS day. We update the trail at EOD
        # AFTER the breach check, matching the engine's per-day order.
        breach = rules.check_max_loss(eq, mll.anchor)
        if breach is not None:
            return CombineWindowResult(
                start_day=start_day, end_day=end_day, n_days=n_days,
                outcome="mll_breach", final_pnl=eq - starting_balance,
                days_to_outcome=i + 1, breach_day=day,
            )
        mll.update_end_of_day(eq)
        if rules.reached_profit_target(eq):
            ever_reached_target = True
            cons = rules.check_consistency(cumulative)
            if cons is None:
                return CombineWindowResult(
                    start_day=start_day, end_day=end_day, n_days=n_days,
                    outcome="pass", final_pnl=eq - starting_balance,
                    days_to_outcome=i + 1, breach_day=None,
                )
            # Target reached but one day is still >50% of total profit --
            # not a failure, keep trading (loop continues; still subject
            # to MLL breach on subsequent days).

    outcome = "consistency_fail" if ever_reached_target else "no_target"
    return CombineWindowResult(
        start_day=start_day, end_day=end_day, n_days=n_days,
        outcome=outcome, final_pnl=eq - starting_balance,
        days_to_outcome=-1, breach_day=None,
    )


@dataclass(frozen=True)
class SequentialAccountResult:
    """Outcome of ONE account in a sequential (non-overlapping) simulation:
    trade until pass/breach, then the next account starts the following
    trading day. Unlike CombineWindowResult, there's no fixed window
    length -- an account runs as long as it takes to resolve, or until the
    data runs out (outcome='no_target' then means "still open, unresolved
    as of the end of this data", not "timed out at some fixed day count").
    """
    account_no: int
    start_day: date
    end_day: date
    n_days: int
    outcome: str  # 'pass' | 'mll_breach' | 'consistency_fail' | 'no_target'
    final_pnl: Decimal
    peak_profit: Decimal          # highest cumulative profit reached, ever
    reached_checkpoint: bool      # peak_profit >= the checkpoint threshold passed in


@dataclass(frozen=True)
class SequentialAccountsSummary:
    accounts: list[SequentialAccountResult]
    checkpoint: Decimal

    @property
    def n_accounts(self) -> int:
        return len(self.accounts)

    def count(self, outcome: str) -> int:
        return sum(1 for a in self.accounts if a.outcome == outcome)

    @property
    def pass_rate(self) -> float:
        return self.count("pass") / self.n_accounts if self.n_accounts else 0.0

    @property
    def checkpoint_accounts(self) -> list[SequentialAccountResult]:
        """Accounts that reached the checkpoint profit at some point."""
        return [a for a in self.accounts if a.reached_checkpoint]

    @property
    def checkpoint_pass_rate(self) -> float:
        """Of the accounts that reached the checkpoint, what fraction went
        on to actually pass (vs. giving it back to a breach, or still
        being open/unresolved)?"""
        cp = self.checkpoint_accounts
        if not cp:
            return 0.0
        return sum(1 for a in cp if a.outcome == "pass") / len(cp)


def simulate_sequential_accounts(
    daily_pnl: dict[date, Decimal],
    *,
    rules: TopstepRules,
    starting_balance: Decimal,
    checkpoint: Decimal = Decimal("1500"),
) -> SequentialAccountsSummary:
    """Simulate opening ONE Combine account at a time against a long
    daily-PnL series -- never two accounts trading at once. Each account
    starts the day after the previous one resolved (passed or breached)
    and runs until IT resolves or the data runs out. This answers "if I
    kept rebilling/continuing the same account while it's alive, and only
    started a fresh one after a breach or a pass, how many account
    attempts would this have been, and how many passed?" -- as opposed to
    `realized_pass_rate`'s sliding window, which starts a new hypothetical
    attempt on literally every day regardless of what's already running.

    `checkpoint` (default $1,500, i.e. half the $50K Combine's $3,000
    target) is tracked per-account: did this account's cumulative profit
    ever reach that level before its final outcome? `checkpoint_pass_rate`
    on the returned summary answers "given an account got at least this
    far ahead at some point, what fraction of the time did it go on to
    actually pass?" -- separate from a fresh account's raw pass rate.

    Rebilling itself doesn't change any of this: it doesn't reset an
    account's progress (see docs/rules_sources.md), so a rebill event
    mid-account is a no-op here -- the SAME account just keeps running.

    A consistency violation does NOT end the account (see the equivalent
    note on `simulate_combine_window` -- same correction applies here):
    hitting the target with one day over 50% of total profit just means
    the account keeps trading, still fully exposed to MLL risk, until a
    later day dilutes the best-day share back under 50%. Outcome
    'consistency_fail' here means the account ran out of OOS/backtest
    data while in that pending state -- not that it was disqualified.
    """
    days_sorted = sorted(daily_pnl.keys())
    accounts: list[SequentialAccountResult] = []
    i = 0
    n = len(days_sorted)
    account_no = 0
    while i < n:
        account_no += 1
        start_day = days_sorted[i]
        mll = TrailingMLL(starting_balance, rules.max_loss_limit_distance)
        eq = starting_balance
        peak_profit = Decimal(0)
        cumulative: dict[date, Decimal] = {}
        outcome: str | None = None
        ever_reached_target = False
        j = i
        while j < n:
            day = days_sorted[j]
            pnl = daily_pnl.get(day, Decimal(0))
            eq += pnl
            cumulative[day] = pnl
            profit = eq - starting_balance
            if profit > peak_profit:
                peak_profit = profit
            breach = rules.check_max_loss(eq, mll.anchor)
            if breach is not None:
                outcome = "mll_breach"
                break
            mll.update_end_of_day(eq)
            if rules.reached_profit_target(eq):
                ever_reached_target = True
                cons = rules.check_consistency(cumulative)
                if cons is None:
                    outcome = "pass"
                    break
                # Not yet consistency-compliant -- keep trading this SAME
                # account (not a failure); fall through to the next day.
            j += 1
        else:
            j = n - 1  # ran out of data before resolving

        if outcome is None:
            outcome = "consistency_fail" if ever_reached_target else "no_target"

        end_day = days_sorted[j]
        accounts.append(SequentialAccountResult(
            account_no=account_no, start_day=start_day, end_day=end_day,
            n_days=j - i + 1, outcome=outcome, final_pnl=eq - starting_balance,
            peak_profit=peak_profit, reached_checkpoint=peak_profit >= checkpoint,
        ))
        i = j + 1  # next account starts the day AFTER this one resolved

    return SequentialAccountsSummary(accounts=accounts, checkpoint=checkpoint)


def realized_pass_rate(
    daily_pnl: dict[date, Decimal] | Iterable[tuple[date, Decimal]],
    *,
    rules: TopstepRules,
    starting_balance: Decimal,
    window_days: int = 30,
    stride_days: int = 1,
) -> RealizedPassRate:
    """Slide a `window_days`-long Combine window over the daily-PnL
    series with the given stride. Returns aggregated pass-rate plus
    every window's outcome.

    Default stride=1 means every starting trading day is a separate
    Combine attempt (a sliding-window pass-rate). Set stride_days=window_days
    for strictly non-overlapping windows.
    """
    if window_days < 2:
        raise ValueError("window_days must be >= 2")
    if stride_days < 1:
        raise ValueError("stride_days must be >= 1")

    if isinstance(daily_pnl, dict):
        items = sorted(daily_pnl.items(), key=lambda x: x[0])
    else:
        items = sorted(daily_pnl, key=lambda x: x[0])
    if len(items) < window_days:
        raise ValueError(
            f"need at least {window_days} days for one window; have {len(items)}"
        )

    results: list[CombineWindowResult] = []
    for start_idx in range(0, len(items) - window_days + 1, stride_days):
        window = items[start_idx : start_idx + window_days]
        results.append(simulate_combine_window(
            window, rules=rules, starting_balance=starting_balance,
        ))
    n_passed = sum(1 for r in results if r.outcome == "pass")
    counts = Counter(r.outcome for r in results)
    return RealizedPassRate(
        n_windows=len(results),
        n_passed=n_passed,
        pass_rate=n_passed / len(results) if results else 0.0,
        outcomes=dict(counts),
        window_results=results,
        window_days=window_days,
        stride_days=stride_days,
    )
