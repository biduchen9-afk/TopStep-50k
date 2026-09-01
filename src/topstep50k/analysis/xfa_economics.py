"""Express Funded Account (XFA) lifecycle simulation and economics.

Everything built earlier this session about the funded phase
(XFARules, PostPayoutDrawdown) was defined but never actually
exercised against a real daily-P&L series. This closes that gap: given
one daily-P&L path (real or resampled), step day by day through a
funded account's life -- track the trailing floor, check payout
eligibility, take payouts when eligible, apply the post-payout floor
reset -- and report what actually happened. Monte-Carlo-resample many
paths from the empirical daily-return distribution to get a
distribution of outcomes, not one lucky/unlucky draw.

Modeling choice flagged explicitly (real ambiguity, see
docs/rules_sources.md "What this explicitly does NOT model"): payout
eligibility (winning days / consistency %) is tracked SINCE THE LAST
PAYOUT, resetting each time one is taken -- not cumulative since
funding. This is the conservative, standard prop-firm reading (each
request must independently qualify) but Topstep's own wording on this
specific point was not found. If the true rule is cumulative-since-
funding, actual payout frequency would be higher than this model shows.

Also inherits PostPayoutDrawdown's own flagged conservatism: the floor
collapses to zero cushion at each payout (sourced from a third-party
worked example, not Topstep's own wording -- see that class's
docstring). Both assumptions push toward UNDER-stating payout
frequency/amount, not over-stating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Callable

import numpy as np

from topstep50k.rules.topstep_xfa import PostPayoutDrawdown, XFARules


@dataclass(frozen=True)
class XFAAccountState:
    """State visible to an XFA sizing function at the START of a trading
    day -- before that day's own P&L is known (no look-ahead, same
    discipline as analysis.sizing.AccountState on the Combine side).

    `cushion` and `locked` are the two structurally important fields:
    pre-first-payout, once cushion reaches the full mll_distance the
    floor LOCKS at the original starting balance and stops trailing --
    a genuinely safe state. The lock does not survive a payout: the
    instant a payout is taken, cushion resets to exactly zero AND the
    floor resumes trailing indefinitely with no further lock (see
    PostPayoutDrawdown docstring) -- so a sizing policy that treats
    "just funded" and "just paid out" the same way is missing the
    single biggest risk concentration in the whole lifecycle.
    """
    balance: Decimal
    floor: Decimal                    # current PostPayoutDrawdown.line
    cushion: Decimal                  # balance - floor, always >= 0
    locked: bool                      # True once the floor has locked (pre-payout only)
    days_since_funding: int
    days_since_last_payout: int | None  # None before the first payout
    n_payouts_so_far: int
    yesterday_pnl: Decimal | None


XFASizingFn = Callable[[XFAAccountState], float]


def xfa_full_size(state: XFAAccountState) -> float:
    """Baseline: no scaling, qty=1 throughout (matches the economics
    run logged this session -- the one that found 100% breach risk)."""
    return 1.0


@dataclass(frozen=True)
class PayoutEvent:
    day_index: int
    path: str                # "standard" | "consistency"
    payout_amount: Decimal   # gross, leaves the account
    trader_take: Decimal     # net to trader after the 90/10 split
    balance_after: Decimal


@dataclass(frozen=True)
class XFALifecycleResult:
    days_run: int
    breached: bool           # True if the account was lost to an MLL breach
    final_balance: Decimal
    payouts: list[PayoutEvent]

    @property
    def n_payouts(self) -> int:
        return len(self.payouts)

    @property
    def total_trader_income(self) -> Decimal:
        return sum((p.trader_take for p in self.payouts), start=Decimal("0"))

    @property
    def days_to_first_payout(self) -> int | None:
        return self.payouts[0].day_index if self.payouts else None


def simulate_xfa_lifecycle(
    daily_pnl: list[Decimal],
    *,
    xfa: XFARules,
    horizon_days: int | None = None,
    sizing_fn: XFASizingFn = xfa_full_size,
) -> XFALifecycleResult:
    """Step through `daily_pnl` (an ORDERED list of UNSCALED daily $ P&L
    -- the raw edge, qty=1) as a funded account's trading life. Each
    day's raw P&L is scaled by `sizing_fn(state)`, evaluated on state
    known strictly BEFORE that day (no look-ahead), before being applied
    to balance -- same pattern as
    analysis.sizing.simulate_sequential_accounts_sized on the Combine
    side. With sizing_fn=xfa_full_size this reproduces the original
    unscaled behavior exactly.

    Stops early on an MLL breach. `horizon_days` caps the run (default:
    the full length of `daily_pnl`).
    """
    n = len(daily_pnl) if horizon_days is None else min(horizon_days, len(daily_pnl))
    pdd = PostPayoutDrawdown(xfa.starting_balance, xfa.mll_distance)
    balance = xfa.starting_balance
    since_last_payout: dict[date, Decimal] = {}
    payouts: list[PayoutEvent] = []
    breached = False
    yesterday_pnl: Decimal | None = None
    days_since_last_payout: int | None = None

    days_completed = 0
    for i in range(n):
        state = XFAAccountState(
            balance=balance, floor=pdd.line, cushion=balance - pdd.line,
            locked=pdd.locked, days_since_funding=i,
            days_since_last_payout=days_since_last_payout,
            n_payouts_so_far=len(payouts), yesterday_pnl=yesterday_pnl,
        )
        scale = sizing_fn(state)
        pnl = daily_pnl[i] * Decimal(str(scale))

        balance += pnl
        # Synthetic date key -- only used for XFARules' Mapping-keyed
        # eligibility checks, which don't care about real calendar dates.
        day_key = date(2000, 1, 1).fromordinal(1 + i)
        since_last_payout[day_key] = pnl
        yesterday_pnl = pnl
        days_since_last_payout = (0 if days_since_last_payout is None
                                   else days_since_last_payout + 1)
        days_completed = i + 1

        if balance <= pdd.line:
            breached = True
            break
        pdd.update_end_of_day(balance)

        paths = xfa.eligible_paths(since_last_payout)
        if paths:
            best_path = max(paths, key=lambda p: xfa.max_payout(p, balance))
            payout = xfa.max_payout(best_path, balance)
            if payout > 0:
                trader_take = xfa.trader_take(payout)
                balance -= payout
                pdd.apply_payout(payout, balance)
                payouts.append(PayoutEvent(
                    day_index=i, path=best_path, payout_amount=payout,
                    trader_take=trader_take, balance_after=balance,
                ))
                since_last_payout = {}
                days_since_last_payout = 0
                # NOTE: no breach check here. apply_payout() re-anchors
                # the line to EXACTLY the post-payout balance (the
                # documented zero-cushion assumption), so balance ==
                # pdd.line is true by construction the instant a payout
                # clears -- checking `balance <= pdd.line` here was a
                # tautology that flagged a "breach" on literally every
                # payout, regardless of any actual subsequent loss. The
                # real breach exposure from zero cushion shows up
                # correctly on the NEXT iteration's ordinary top-of-loop
                # check, once a real day's P&L can actually move balance
                # relative to the (now fixed) line.

    return XFALifecycleResult(days_run=days_completed, breached=breached,
                               final_balance=balance, payouts=payouts)


@dataclass(frozen=True)
class XFAMonteCarloResult:
    n_sims: int
    horizon_days: int
    total_income: np.ndarray        # one draw per sim: total trader $ over the horizon
    mean_income: float
    median_income: float
    p05_income: float
    p95_income: float
    prob_breach: float              # P(account lost to MLL breach within horizon)
    mean_n_payouts: float
    mean_days_to_first_payout: float | None  # over sims that got a first payout

    @property
    def prob_survive(self) -> float:
        return 1.0 - self.prob_breach


def monte_carlo_xfa_economics(
    empirical_daily_pnl: list[Decimal],
    *,
    xfa: XFARules,
    horizon_days: int = 252,
    n_sims: int = 1000,
    block_len: int = 10,
    seed: int = 42,
    sizing_fn: XFASizingFn = xfa_full_size,
) -> XFAMonteCarloResult:
    """Block-bootstrap-resample `empirical_daily_pnl` (same block_len=10
    convention as everywhere else this session) into `n_sims` synthetic
    paths of length `horizon_days`, run simulate_xfa_lifecycle on each,
    and report the distribution of trader income -- not just its mean.
    """
    values = np.array([float(v) for v in empirical_daily_pnl])
    m = len(values)
    if m < block_len:
        raise ValueError(f"need at least block_len={block_len} days, have {m}")
    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon_days // block_len)

    incomes = np.empty(n_sims)
    breaches = np.empty(n_sims, dtype=bool)
    n_payouts_arr = np.empty(n_sims)
    first_payout_days: list[int] = []

    for i in range(n_sims):
        starts = rng.integers(0, m, size=n_blocks)
        blocks = []
        for s in starts:
            if s + block_len <= m:
                blocks.append(values[s : s + block_len])
            else:
                blocks.append(np.concatenate([values[s:], values[: block_len - (m - s)]]))
        path = np.concatenate(blocks)[:horizon_days]
        daily_pnl = [Decimal(str(round(float(v), 2))) for v in path]

        result = simulate_xfa_lifecycle(daily_pnl, xfa=xfa, horizon_days=horizon_days,
                                         sizing_fn=sizing_fn)
        incomes[i] = float(result.total_trader_income)
        breaches[i] = result.breached
        n_payouts_arr[i] = result.n_payouts
        if result.days_to_first_payout is not None:
            first_payout_days.append(result.days_to_first_payout)

    return XFAMonteCarloResult(
        n_sims=n_sims, horizon_days=horizon_days,
        total_income=incomes,
        mean_income=float(incomes.mean()),
        median_income=float(np.median(incomes)),
        p05_income=float(np.percentile(incomes, 5)),
        p95_income=float(np.percentile(incomes, 95)),
        prob_breach=float(breaches.mean()),
        mean_n_payouts=float(n_payouts_arr.mean()),
        mean_days_to_first_payout=(float(np.mean(first_payout_days))
                                    if first_payout_days else None),
    )
