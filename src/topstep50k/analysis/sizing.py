"""Position-sizing overlays on the sequential-account simulator.

A different lever than alpha search: this session tried eleven-plus
distinct entry-signal candidates to replace OD and found nothing that
survives DSR deflation + real OOS verification. This explores whether
a smarter bet-sizing POLICY on the SAME already-real edge (ORB+MeanRev)
can improve the discrete pass/fail outcome against the Combine's
specific risk structure (a $2,000 trailing MLL, a $3,000 target, no
benefit to profit beyond the target) -- not a new signal, a different
way to spend the existing one.

Two concrete ideas, both causal (decided using only state known as of
the START of that trading day, never today's own P&L):

  target_proximity_scaling  -- once cumulative profit crosses some
    fraction of the way to target, cut size. Rationale: the marginal
    value of additional profit falls once you're close to the target
    (extra profit past it is not credited toward passing sooner), while
    the marginal cost of a bad day is unchanged (still a full breach
    risk against the trailing MLL) -- so the risk/reward of full size
    gets worse specifically once you're ahead, not better.

  drawdown_responsive_scaling -- cut size for a few days after a
    losing day, on the theory that a string of losses is more likely
    to continue trading through/into an MLL breach at full size than
    to reverse -- so temporarily reducing size buys the account more
    days to recover, at the cost of a slower recovery if the edge
    reasserts immediately.

Both are DESCRIBED, not proven, by this module -- see
scripts/evaluate_sizing_overlays.py for whether either actually
improves the sequential pass rate on real OOS data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Callable

from topstep50k.analysis.passrate import SequentialAccountResult, SequentialAccountsSummary
from topstep50k.rules.topstep import TopstepRules, TrailingMLL


@dataclass(frozen=True)
class AccountState:
    """State visible to a sizing function at the START of a trading
    day -- before that day's own P&L is known. Never includes anything
    from the current day itself (no look-ahead)."""
    profit_so_far: Decimal          # cumulative profit BEFORE today
    peak_profit: Decimal            # highest profit_so_far ever reached, this account
    days_elapsed: int               # trading days completed so far (0 on day 1)
    yesterday_pnl: Decimal | None   # None on day 1


SizingFn = Callable[[AccountState], float]


def full_size(state: AccountState) -> float:
    """Baseline: no scaling, qty=1 throughout (matches every prior
    result this session)."""
    return 1.0


def target_proximity_scaling(
    checkpoint: Decimal = Decimal("1500"),
    scale_after: float = 0.5,
) -> SizingFn:
    """Cut size to `scale_after` once cumulative profit >= checkpoint."""
    def fn(state: AccountState) -> float:
        return scale_after if state.profit_so_far >= checkpoint else 1.0
    return fn


def drawdown_responsive_scaling(
    scale_after_loss: float = 0.5,
) -> SizingFn:
    """Cut size to `scale_after_loss` for the day immediately following
    a losing day. Reverts to full size the day after that (not a
    multi-day cooldown -- kept simple and easy to reason about)."""
    def fn(state: AccountState) -> float:
        if state.yesterday_pnl is not None and state.yesterday_pnl < 0:
            return scale_after_loss
        return 1.0
    return fn


def combined_scaling(*fns: SizingFn) -> SizingFn:
    """Apply the MINIMUM of several sizing functions' outputs (most
    conservative wins) -- for testing whether combining rules helps or
    just over-shrinks."""
    def fn(state: AccountState) -> float:
        return min(f(state) for f in fns)
    return fn


def simulate_sequential_accounts_sized(
    daily_pnl: dict[date, Decimal],
    *,
    rules: TopstepRules,
    starting_balance: Decimal,
    sizing_fn: SizingFn = full_size,
    checkpoint: Decimal = Decimal("1500"),
) -> SequentialAccountsSummary:
    """Same account-lifecycle logic as
    analysis.passrate.simulate_sequential_accounts, but each day's raw
    P&L is scaled by `sizing_fn(state)` -- evaluated on state known
    strictly BEFORE that day -- before being applied to equity. With
    sizing_fn=full_size this reproduces simulate_sequential_accounts
    exactly (see the parity test).
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
        yesterday_pnl: Decimal | None = None
        j = i
        while j < n:
            day = days_sorted[j]
            raw_pnl = daily_pnl.get(day, Decimal(0))
            profit_so_far = eq - starting_balance
            state = AccountState(
                profit_so_far=profit_so_far, peak_profit=peak_profit,
                days_elapsed=j - i, yesterday_pnl=yesterday_pnl,
            )
            scale = sizing_fn(state)
            pnl = raw_pnl * Decimal(str(scale))

            eq += pnl
            cumulative[day] = pnl
            yesterday_pnl = pnl
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
            j += 1
        else:
            j = n - 1

        if outcome is None:
            outcome = "consistency_fail" if ever_reached_target else "no_target"

        end_day = days_sorted[j]
        accounts.append(SequentialAccountResult(
            account_no=account_no, start_day=start_day, end_day=end_day,
            n_days=j - i + 1, outcome=outcome, final_pnl=eq - starting_balance,
            peak_profit=peak_profit, reached_checkpoint=peak_profit >= checkpoint,
        ))
        i = j + 1

    return SequentialAccountsSummary(accounts=accounts, checkpoint=checkpoint)
