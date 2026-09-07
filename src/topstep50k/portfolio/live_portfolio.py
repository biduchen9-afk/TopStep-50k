"""Live daily-signal portfolio for the 6-stream regime-gated ensemble.

*** READ BEFORE TRADING THIS (updated 2026-08-26) ***
OD (OvernightDrift) is PERMANENTLY REMOVED: TopStep does not allow
overnight or weekend holding at ANY account stage -- Combine, XFA, or
Live Funded -- confirmed current per a July 1, 2026 rules update (see
docs/rules_sources.md).

This ensemble (ORB + MeanRev x ES/NQ/GC, no OD) is the best-available
result of an extensive session-long search for a legal third leg to
replace OD's contribution -- eleven distinct strategy/gate combinations
were tried (GapFill, VolumeProfileMeanReversion incl. a real stop-sign
bugfix, VwapStdFractal, IntradayMomentum + a take-profit variant raw
and volatility-gated, ThreeDayReversal low-vol-gated, an ORB variant
that holds to close instead of a fixed target, PowerHourContinuation
-- a new end-of-session strategy purpose-built to be time-disjoint from
everything else) and NONE added genuine, non-redundant, OOS-surviving
edge; several looked promising in-sample and were confirmed via DSR
deflation and/or direct OOS computation to be overfit. See
docs/rules_sources.md and the evaluate_*_databento*.py / screen_*.py
scripts for the individual writeups.

HONEST CAVEAT: this ensemble does NOT cleanly clear its own Gate 2/3
promotion checklist either -- t-stat=1.32 vs the 1.5 hard threshold
required (evaluate_ensemble_databento_v10_no_overnight.py). Every
other check passes (EV, profit factor, Sharpe, IS/OOS pass30 ratio,
OOS Sharpe ratio all OK) and OOS actually performs in line with IS
(Sharpe 0.74 both), so this reads as "not yet enough data to be
statistically certain" rather than "no edge" -- but it is a real,
acknowledged gap, not a clean pass. Sequential (non-overlapping)
account simulation on the OOS period: 34 accounts, 10 pass = 29.4%
unconditional pass rate; Monte Carlo block-bootstrap (2000 resamples):
mean 32.4%, 90% interval [20.0%, 46.7%], P(true rate > 50%) = 2.2%.
Expected cost to pass one $50K Combine at this pass rate (TopStep
Standard Path, $49/mo rebill + $149 one-time activation): roughly
$254-$394, central estimate ~$300.

WALK-FORWARD VALIDATION (added 2026-08-26, evaluate_walkforward.py):
the single 70/30 split above is only one draw of "which period was
OOS." Re-ran with an expanding walk-forward (500-day minimum IS, 5
sequential ~133-day OOS folds, weights re-derived each fold from only
prior data) and concatenated the 5 folds' OOS segments into one
~665-day walk-forward-OOS series -- nearly double the original
single-touch OOS length. Result: Monte Carlo mean 33.8%, TIGHTER 90%
interval [24.6%, 44.0%], P(true rate > 50%) = 0.3%. This confirms the
single-split number wasn't a lucky (or unlucky) draw -- the walk-
forward estimate lands almost exactly where the single split did,
just with a narrower, more confident interval. Also surfaced real
fold-to-fold variance worth knowing about operationally: one fold
(2025-06-25 to 2025-12-29) was genuinely negative (Sharpe -0.76,
-$6,387, pass30=13.5%) -- this system has real multi-month losing
stretches, not just steady grinding. GC/ORB's weight also decayed to
0 in the most recent fold, worth monitoring for a possibly-fading edge.

POSITION-SIZING OVERLAY (added 2026-08-26, evaluate_sizing_overlays.py):
a genuine, if modest, improvement was found NOT from a new signal but
from a smarter bet-sizing policy on this same edge -- selected using
IS Monte Carlo mean pass rate only, then validated with exactly one
OOS touch (same one-touch discipline as strategy selection; a first
attempt that picked the policy by comparing OOS numbers directly was
caught and corrected before being reported). Rule: once an account's
CUMULATIVE PROFIT reaches $1,500 (the same checkpoint tracked
elsewhere in this file), cut position size to 0.5x contracts for the
rest of that account's life. Rationale: past that point the marginal
value of more profit is lower (nothing is credited for clearing the
$3,000 target by more), while the marginal cost of a bad day is
unchanged (still full MLL breach risk) -- so full size stops being the
right risk/reward trade specifically once you're already ahead.
Requires the TRADER to track their own account's running profit and
apply the 0.5x cut manually -- this stateless daily planner has no
account-state to hook it into automatically. OOS result: Monte Carlo
mean pass rate 32.4% -> 35.8%, P(true rate > 50%) 2.2% -> 7.3%. Real,
but still well short of confidently clearing 50% -- treat as a
worthwhile addition on top of the base ensemble, not a fix on its own.

Run `scripts/generate_daily_signals.py` before 9:30 ET with data through
yesterday's close to get today's trading plan.

Weights are frozen from evaluate_ensemble_databento_v10_no_overnight.py
on the corrected Databento dataset (this file's IS_WEIGHTS must match
that script's printed weights exactly):
  Data       : es/nq/gc_databento.txt (GitHub Release "Data list" v1.0.0)
  IS period  : 2021-12-31 to 2025-02-25 (815d, 70%)
  OOS period : 2025-02-26 to 2026-07-03 (350d, 30%, one-touch)
  IS  result : pass30=35.1%, Sharpe=+0.74 (t-stat=1.32, below 1.5 threshold)
  OOS result : pass30=28.7%, pass45=28.8%, Sharpe=+0.74, EV=+$62/day

No new parameters, no re-fitting. Gates are literature-grounded rules with
zero free parameters tuned to our data (see regime/conditioners.py).

IMPORTANT: this whole result depends on THREE engine-level fixes:
(1, 2026-08-26) data/loaders.py was tagging raw NinjaTrader/SierraChart
timestamps as UTC when they are actually US/Eastern local time --
verified against the CME daily maintenance-halt gap position and the
NYSE open/close volume spikes; (2, 2026-08-26) engine/ledger.py's
trading_day() now rolls at 16:00 CT (the CME session boundary) instead
of raw calendar date; (3, 2026-08-26) analysis/passrate.py's consistency
check no longer treats a >50%-best-day violation as a terminal failure
-- per Topstep's own published mechanic, the effective profit target
rises to 2x the best day and trading continues, which raised both IS
and OOS pass30/pass45 materially and reshuffled the EV-gated weights
below (this fix changed the WEIGHTS themselves, not just the reported
pass rate, because per-stream IS-pass30 feeds the weight derivation).
Re-running any of this on a DIFFERENT data source must re-verify the
source file's actual timezone convention before trusting the result --
do not assume it's UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from topstep50k.engine.types import Bar

# ---------------------------------------------------------------------------
# Frozen IS pass-rate-aware weights (do not change without re-evaluation)
# Derived from IS evaluation in
# scripts/evaluate_ensemble_databento_v10_no_overnight.py -- EV-gated: any
# stream with IS EV <= 0 gets zero weight (GC/MeanRev here).
# ---------------------------------------------------------------------------

IS_WEIGHTS: dict[tuple[str, str], float] = {
    ("ES", "ORB"):     0.357,
    ("NQ", "ORB"):     0.333,
    ("GC", "ORB"):     0.247,
    ("ES", "MeanRev"): 0.049,
    ("NQ", "MeanRev"): 0.015,
    ("GC", "MeanRev"): 0.000,
}

# Pre-committed strategy parameters (must match evaluate_ensemble_*.py scripts)
ORB_PARAMS = dict(
    qty=1,
    or_minutes=30,
    direction="both",
    stop_mode="opposite_range",
    stop_ticks=40,
    tp_multiple=1.0,
    flat_before_close_minutes=15,
)
MR_PARAMS = dict(
    qty=1,
    lookback=60,
    sigma_mult=2.0,
    stop_ticks=15,
    time_stop_minutes=45,
)

_ORB_EXIT_TIME = time(15, 45)
_MR_EXIT_TIME  = time(15, 45)


@dataclass(frozen=True)
class StreamStatus:
    """Status of one strategy-asset stream for a given trading date."""
    asset: str
    strategy: str
    active: bool
    gate_name: str
    gate_detail: str   # human-readable gate value / reason
    weight: float      # IS pass-rate-aware weight (frozen)

    @property
    def key(self) -> tuple[str, str]:
        return (self.asset, self.strategy)


@dataclass
class DailyPlan:
    """Trading plan for one session, produced before market open."""
    for_date: date
    streams: list[StreamStatus]

    @property
    def active(self) -> list[StreamStatus]:
        return [s for s in self.streams if s.active]

    @property
    def inactive(self) -> list[StreamStatus]:
        return [s for s in self.streams if not s.active]

    @property
    def total_active_weight(self) -> float:
        return sum(s.weight for s in self.active)

    def print_plan(self) -> None:
        """Print a human-readable pre-market plan to stdout."""
        divider = "─" * 68
        print(f"\n{'='*68}")
        print(f"  TOPSTEP $50K TRADING PLAN — {self.for_date}")
        print(f"{'='*68}")

        active = sorted(self.active, key=lambda s: -s.weight)
        inactive = sorted(self.inactive, key=lambda s: -s.weight)

        if not active:
            print("\n  ⚑  NO ACTIVE STREAMS TODAY — STAND ASIDE\n")
        else:
            print(f"\n  ACTIVE ({len(active)} streams, combined weight = "
                  f"{self.total_active_weight:.3f} / 1.000)\n")
            for s in active:
                print(f"  ✓  {s.asset}/{s.strategy:<8}  w={s.weight:.3f}  "
                      f"[{s.gate_name}: {s.gate_detail}]")

        if inactive:
            print(f"\n  INACTIVE ({len(inactive)} streams)\n")
            for s in inactive:
                print(f"  ✗  {s.asset}/{s.strategy:<8}  w={s.weight:.3f}  "
                      f"[{s.gate_name}: {s.gate_detail}]")

        if active:
            print(f"\n{divider}")
            print("  EXECUTION NOTES (active streams only)")
            print(divider)
            for s in active:
                _print_execution_note(s)

        print(f"\n{'='*68}\n")


def _print_execution_note(s: StreamStatus) -> None:
    prefix = f"  {s.asset}/{s.strategy}"
    if s.strategy == "ORB":
        print(f"{prefix}:")
        print(f"    WATCH 09:30–10:00 ET to establish the opening range")
        print(f"    At 10:00 ET, trade in the direction of the 30-min return:")
        print(f"      Positive 30-min return → BUY;  stop = range LOW")
        print(f"      Negative 30-min return → SELL; stop = range HIGH")
        print(f"    Take-profit = 1.0× range distance beyond entry")
        print(f"    Hard flat at {_ORB_EXIT_TIME} ET if neither TP nor stop hit")
    elif s.strategy == "MeanRev":
        print(f"{prefix}:")
        print(f"    Fade moves that extend ≥ 2σ outside the 60-min Bollinger band")
        print(f"    Stop = 15 ticks; flatten after 45 min if not exited")
        print(f"    Hard flat at {_MR_EXIT_TIME} ET")
    print()


# ---------------------------------------------------------------------------
# Core: build today's plan from historical bars
# ---------------------------------------------------------------------------

def build_daily_plan(
    bars_by_asset: dict[str, list["Bar"]],
    for_date: date,
) -> DailyPlan:
    """Compute today's trading plan from historical bars.

    Parameters
    ----------
    bars_by_asset : dict mapping asset symbol → list of Bar objects, all
        with ts <= end-of-day yesterday. The gate computation uses only
        prior-day data so this is strictly causal.
    for_date : the trading date you want signals for (typically today).

    Returns
    -------
    DailyPlan with gate status, weights, and execution notes for each
    of the 6 strategy-asset streams.
    """
    from topstep50k.regime.conditioners import (
        per_day_session_stats,
        rolling_vol,
        trailing_median,
    )

    streams: list[StreamStatus] = []

    for asset in sorted(bars_by_asset.keys()):
        bars = bars_by_asset[asset]
        if not bars:
            continue
        stats = per_day_session_stats(bars)

        # Locate the prior trading day (gate is always on prior-day data).
        # We find the last known trading day that is strictly before for_date.
        # This handles both in-sample dates and future dates beyond data end.
        days_sorted = sorted(stats.keys())
        prior_day: date | None = None
        for d in reversed(days_sorted):
            if d < for_date:
                prior_day = d
                break

        rv5 = rolling_vol(stats, 5)
        rv20 = rolling_vol(stats, 20)
        rv20_median = trailing_median(rv20, 60)

        # -- ORB gate: rv5_prior / rv20_prior >= 0.8 --
        if prior_day is not None:
            v5 = rv5.get(prior_day, 0.0)
            v20 = rv20.get(prior_day, 0.0)
            if v5 > 0 and v20 > 0:
                ratio = v5 / v20
                orb_active = ratio >= 0.8
                cmp = "≥" if orb_active else "<"
                orb_detail = f"rv5/rv20 = {ratio:.2f} {cmp} 0.80"
            else:
                orb_active = False
                orb_detail = "insufficient history (vol=0)"
        else:
            orb_active = False
            orb_detail = "insufficient history"
        streams.append(StreamStatus(
            asset=asset, strategy="ORB",
            active=orb_active, gate_name="expansion_gate",
            gate_detail=orb_detail,
            weight=IS_WEIGHTS.get((asset, "ORB"), 0.0),
        ))

        # -- MeanRev gate: rv20_prior < median(rv20, 60d) --
        if prior_day is not None:
            v20_p = rv20.get(prior_day, 0.0)
            med = rv20_median.get(prior_day, 0.0)
            if v20_p > 0 and med > 0:
                mr_active = v20_p < med
                cmp = "<" if mr_active else "≥"
                mr_detail = f"rv20 = {v20_p:.4f} {cmp} median {med:.4f}"
            else:
                mr_active = False
                mr_detail = "insufficient history (vol=0)"
        else:
            mr_active = False
            mr_detail = "insufficient history"
        streams.append(StreamStatus(
            asset=asset, strategy="MeanRev",
            active=mr_active, gate_name="low_vol_gate",
            gate_detail=mr_detail,
            weight=IS_WEIGHTS.get((asset, "MeanRev"), 0.0),
        ))

    return DailyPlan(for_date=for_date, streams=streams)
