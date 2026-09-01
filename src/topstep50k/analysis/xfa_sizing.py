"""Candidate risk-adjustment (position-sizing) policies for the XFA
lifecycle -- the funded-phase analogue of analysis/sizing.py's
Combine-phase overlays.

Structural starting point (read PostPayoutDrawdown's docstring in
rules/topstep_xfa.py first): pre-first-payout, once cumulative profit
reaches the full mll_distance ($2,000), the trailing floor LOCKS at the
account's original starting balance and stops moving -- a genuinely
safe state; a full-size losing streak can only breach by giving back
the entire $2,000 cushion. The lock does NOT survive a payout: the
instant a payout is taken, cushion resets to exactly zero AND the floor
resumes trailing indefinitely with no further lock. That means the
single riskiest moment in the whole funded lifecycle is the day right
after a payout clears -- one full-size losing day, at zero cushion,
can end the account outright. This is why the 100% breach figure
logged earlier this session isn't really "the edge isn't good enough";
it's "full size, forever, includes a payout-triggered reset to the
single most fragile state a trailing-floor account can be in, repeated
every ~10 days (the measured mean days-to-first-payout)."

Two policy families follow directly from that:

  cushion_proportional_scaling -- scale size by how much of the
    mll_distance is currently rebuilt as cushion (0 at a fresh floor,
    1.0 once fully rebuilt or locked). This is the general-purpose
    lever: thin cushion == small size, everywhere in the lifecycle, not
    just right after a payout.

  post_payout_cooldown -- a blunter, easier-to-reason-about version:
    cut size hard for a fixed number of trading days immediately after
    EVERY payout (when cushion is known to be exactly zero), then
    revert to full size. Cheaper to explain to a live trader than a
    continuous ramp; a natural baseline before trying the smoother
    version above.

Both are DESCRIBED, not proven, here -- see
scripts/evaluate_xfa_sizing_overlays.py for whether either actually
moves the survival probability, under the same select-on-IS /
touch-OOS-once discipline used for the Combine-phase sizing overlay.

A real tension neither policy escapes: shrinking size to survive
also shrinks the P&L that makes the account eligible for the NEXT
payout in the first place (winning-day $ thresholds don't move) -- so
the objective is never "maximize survival" alone, it's "survival
subject to still generating a meaningfully positive income stream."
See the script for how that trade-off is reported, not just the
survival number in isolation.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Callable

from topstep50k.analysis.xfa_economics import XFAAccountState, XFASizingFn, xfa_full_size


def cushion_proportional_scaling(
    mll_distance: Decimal,
    floor_frac: float = 0.2,
    ramp_frac: float = 1.0,
) -> XFASizingFn:
    """Scale size linearly with cushion/mll_distance: `floor_frac` at
    zero cushion (never fully flat -- a fully flat account can never
    become eligible for the next payout), ramping to full size once
    cushion reaches `ramp_frac` of mll_distance. Once the floor is
    LOCKED (pre-payout, cushion already maxed and safe), always full
    size regardless of the nominal ratio.
    """
    def fn(state: XFAAccountState) -> float:
        if state.locked:
            return 1.0
        ratio = float(state.cushion / mll_distance) / ramp_frac if mll_distance else 1.0
        ratio = max(0.0, min(1.0, ratio))
        return floor_frac + (1.0 - floor_frac) * ratio
    return fn


def post_payout_cooldown(
    cooldown_days: int = 10,
    scale_during: float = 0.3,
) -> XFASizingFn:
    """Cut size to `scale_during` for the first `cooldown_days` trading
    days after EACH payout (days_since_last_payout in
    [0, cooldown_days)); full size otherwise, including the entire
    pre-first-payout run.
    """
    def fn(state: XFAAccountState) -> float:
        if (state.days_since_last_payout is not None
                and state.days_since_last_payout < cooldown_days):
            return scale_during
        return 1.0
    return fn


def hard_stop_after(max_days: int, scale_after: float = 0.0) -> XFASizingFn:
    """Cut size to `scale_after` (default 0 = flat) once
    days_since_funding >= max_days. Unlike the two policies above, this
    reacts to TENURE, not account state -- it bounds total time-at-risk
    instead of reacting to cushion or a recent payout. Tests a
    different hypothesis: that the risk here isn't concentrated in an
    identifiable account STATE at all, but simply accumulates with every
    additional day spent exposed to a trailing (record-chasing) floor,
    so no state-reactive policy can fix it -- only bounding the horizon
    can.
    """
    def fn(state: XFAAccountState) -> float:
        return scale_after if state.days_since_funding >= max_days else 1.0
    return fn


def time_decay_scaling(half_life_days: float, floor_frac: float = 0.1) -> XFASizingFn:
    """Exponentially decay size with tenure (half-life in trading days),
    floored at `floor_frac` rather than decaying all the way to zero --
    a smoother version of hard_stop_after's same tenure-based
    hypothesis."""
    def fn(state: XFAAccountState) -> float:
        decay = 0.5 ** (state.days_since_funding / half_life_days)
        return floor_frac + (1.0 - floor_frac) * decay
    return fn


def combined_xfa_scaling(*fns: XFASizingFn) -> XFASizingFn:
    """Apply the MINIMUM of several XFA sizing functions (most
    conservative wins) -- e.g. combine a post-payout cooldown with a
    continuous cushion ramp."""
    def fn(state: XFAAccountState) -> float:
        return min(f(state) for f in fns)
    return fn
