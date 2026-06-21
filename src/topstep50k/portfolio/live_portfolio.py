"""Live daily-signal portfolio for the 9-stream regime-gated ensemble.

Run `scripts/generate_daily_signals.py` before 9:30 ET with data through
yesterday's close to get today's trading plan.

All gate logic and weights are frozen from the OOS-promoted IS evaluation:
  IS period  : 2022-01 to 2025-01 (70%)
  OOS period : 2025-01 to 2026-04 (30%)
  OOS result : +$91/day, Sharpe=+1.44, pass30=37.2% — OOS_PROMOTED

No new parameters, no re-fitting. Gates are literature-grounded rules with
zero free parameters tuned to our data (see regime/conditioners.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from topstep50k.engine.types import Bar

# ---------------------------------------------------------------------------
# Frozen IS pass-rate-aware weights (do not change without re-evaluation)
# Derived from IS evaluation in scripts/evaluate_ensemble_is.py
# ---------------------------------------------------------------------------

IS_WEIGHTS: dict[tuple[str, str], float] = {
    ("ES", "ORB"):     0.289,
    ("GC", "OD"):      0.210,
    ("GC", "ORB"):     0.166,
    ("ES", "OD"):      0.147,
    ("NQ", "ORB"):     0.118,
    ("NQ", "OD"):      0.052,
    ("ES", "MeanRev"): 0.010,
    ("NQ", "MeanRev"): 0.009,
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
OD_PARAMS = dict(
    qty=1,
    entry_offset_minutes=5,
    exit_offset_minutes=5,
)

_ORB_EXIT_TIME = time(15, 45)
_MR_EXIT_TIME  = time(15, 45)
_OD_ENTRY_TIME = time(9, 35)
_OD_EXIT_TIME  = time(9, 55)


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
    if s.strategy == "OD":
        print(f"{prefix}:")
        print(f"    BUY 1 contract at {_OD_ENTRY_TIME} ET open")
        print(f"    EXIT at {_OD_EXIT_TIME} ET (hard flat)")
        print(f"    No stop — intraday carry only, exit is time-based")
    elif s.strategy == "ORB":
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
    of the 9 strategy-asset streams.
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

        # -- OD gate: prior RTH return < 0 --
        if prior_day is not None and prior_day in stats:
            rth_r = stats[prior_day].rth_return
            od_active = rth_r < 0.0
            cmp = "<" if od_active else "≥"
            od_detail = f"prior RTH = {rth_r*100:+.2f}% {cmp} 0%"
        else:
            od_active = False
            od_detail = "insufficient history"
        streams.append(StreamStatus(
            asset=asset, strategy="OD",
            active=od_active, gate_name="post_selloff_gate",
            gate_detail=od_detail,
            weight=IS_WEIGHTS.get((asset, "OD"), 0.0),
        ))

    return DailyPlan(for_date=for_date, streams=streams)
