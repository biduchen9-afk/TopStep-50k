"""Strategy #4: Gap Fill — promotion checklist evaluation.

Pre-committed before any data is viewed:
  Predicted edge direction : FADE (long gap-down, short gap-up)
  Expected IS pass30 range : 15-30%
  Regime placement         : MEAN-REVERTING (avoid trending / high-vol days)
  IS trade freq expected   : ~3-5/month per asset

Parameters (pre-committed, literature-derived):
  gap_threshold_pct = 0.0015  (0.15%)
  max_gap_pct       = 0.0150  (1.50%)
  stop_multiple     = 1.0     (1:1 RR)
  time_stop_minutes = 90      (11:00 ET if open at 09:30)

Evaluated on ES only first (fastest). If OOS-promoted, fan out to NQ/GC.
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.calendar import is_high_impact_day
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.evaluation.harness import (
    evaluate_strategy,
    is_oos_split,
)
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.gap_fill import GapFill


IS_FRACTION = 0.70
ASSETS = {
    "ES": {
        "instrument": Instrument(symbol="ES", point_value=Decimal("50"),
                                  tick_size=Decimal("0.25"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.25,
        "data_path": ROOT / "data" / "raw" / "es_cleaned.txt",
    },
    "NQ": {
        "instrument": Instrument(symbol="NQ", point_value=Decimal("20"),
                                  tick_size=Decimal("0.25"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.25,
        "data_path": ROOT / "data" / "raw" / "nq_cleaned.txt",
    },
    "GC": {
        "instrument": Instrument(symbol="GC", point_value=Decimal("100"),
                                  tick_size=Decimal("0.10"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.10,
        "data_path": ROOT / "data" / "raw" / "gc_cleaned.txt",
    },
}
RULES = combine_50k()

GAP_PARAMS = dict(
    qty=1,
    gap_threshold_pct=0.0015,
    max_gap_pct=0.0150,
    stop_multiple=1.0,
    time_stop_minutes=90,
    flat_before_close_minutes=15,
)


def run_asset(asset_key: str, bars, with_filter: bool = False) -> dict[date, Decimal]:
    asset = ASSETS[asset_key]
    filt = (lambda d: not is_high_impact_day(d)) if with_filter else None
    strat = GapFill(symbol=asset_key,
                     tick_size=asset["tick_size_f"],
                     daily_filter=filt,
                     **GAP_PARAMS)
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: asset["instrument"]},
                    strategy=pf, audit=InMemoryAuditLog(),
                    combine_enforcement=False)
    result = bt.run(clk, src)
    return result.daily_pnl


def to_series(daily_pnl: dict, day_index: list[date]) -> np.ndarray:
    out = np.zeros(len(day_index), dtype=float)
    for i, d in enumerate(day_index):
        if d in daily_pnl:
            out[i] = float(daily_pnl[d])
    return out


def print_quick_stats(label: str, arr: np.ndarray, days: list[date]) -> None:
    nz = int((arr != 0).sum())
    if arr.size < 2:
        print(f"  {label}: insufficient data")
        return
    sh = (arr.mean() / arr.std(ddof=1)) * np.sqrt(252) if arr.std(ddof=1) > 0 else 0
    pf_v = (arr[arr > 0].sum() / -arr[arr < 0].sum()
             if (arr < 0).any() else float("inf"))
    eq = np.cumsum(arr)
    mdd = float((eq - np.maximum.accumulate(eq)).min())
    print(f"  {label:<30}: n_days={arr.size} nz={nz:>4} "
          f"total=${arr.sum():>+9,.0f} Sharpe={sh:>+5.2f} "
          f"PF={pf_v:>4.2f} MaxDD=${mdd:>+8,.0f}")
    pnl_d = {d: Decimal(str(round(float(v), 2))) for d, v in zip(days, arr)}
    rr30 = realized_pass_rate(pnl_d, rules=RULES,
                               starting_balance=RULES.starting_balance,
                               window_days=30, stride_days=1)
    rr45 = realized_pass_rate(pnl_d, rules=RULES,
                               starting_balance=RULES.starting_balance,
                               window_days=45, stride_days=1)
    print(f"  {'':>30}  pass30={rr30.pass_rate:.1%} ({rr30.n_passed}/{rr30.n_windows})  "
          f"pass45={rr45.pass_rate:.1%} ({rr45.n_passed}/{rr45.n_windows})")


def main():
    print("=" * 72)
    print("STRATEGY #4: GAP FILL — PROMOTION CHECKLIST")
    print("Pre-committed prediction: long gap-down / short gap-up, IS pass30 15-30%")
    print("=" * 72)

    # ── Load ES bars and run both variants ─────────────────────────────────
    print("\nLoading ES bars...", flush=True)
    t0 = _time.time()
    es_bars = list(load_bars_csv(ASSETS["ES"]["data_path"]))
    print(f"  {len(es_bars):,} bars in {_time.time() - t0:.1f}s", flush=True)

    print("\nRunning ES GapFill (no filter)...", flush=True)
    t0 = _time.time()
    es_pnl_raw = run_asset("ES", es_bars, with_filter=False)
    print(f"  done in {_time.time() - t0:.1f}s  ({len(es_pnl_raw)} days)", flush=True)

    print("Running ES GapFill (skip FOMC + NFP days)...", flush=True)
    t0 = _time.time()
    es_pnl_filt = run_asset("ES", es_bars, with_filter=True)
    print(f"  done in {_time.time() - t0:.1f}s  ({len(es_pnl_filt)} days)", flush=True)

    # ── Build day index and IS/OOS split ──────────────────────────────────
    union_days = sorted(set(es_pnl_raw) | set(es_pnl_filt))
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
    print(f"\nIS : {is_days[0]} -> {is_days[-1]} ({len(is_days)} d)")
    print(f"OOS: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d)")

    raw_full  = to_series(es_pnl_raw,  union_days)
    filt_full = to_series(es_pnl_filt, union_days)

    # ── Quick stats overview ───────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("QUICK STATS (IS only, for Gate 2/3 preview)")
    print(f"{'─'*72}")
    print_quick_stats("ES GapFill raw",      raw_full[is_mask],  is_days)
    print_quick_stats("ES GapFill +filter",  filt_full[is_mask], is_days)

    # ── Full IS/OOS checklist for the BEST variant ─────────────────────────
    print(f"\n{'─'*72}")
    print("GATE-BY-GATE CHECKLIST (raw variant first)")
    print(f"{'─'*72}")
    result_raw = evaluate_strategy(
        label="GapFill-ES-raw",
        daily_full=raw_full,
        is_mask=is_mask, oos_mask=oos_mask,
        is_days=is_days, oos_days=oos_days,
        rules=RULES,
        best_existing_is_pass30=0.33,  # reference: multi-asset IS pass30
        run_oos=True,
    )
    result_raw.print_report()

    print(f"\n{'─'*72}")
    print("GATE-BY-GATE CHECKLIST (+FOMC/NFP filter)")
    print(f"{'─'*72}")
    result_filt = evaluate_strategy(
        label="GapFill-ES-filtered",
        daily_full=filt_full,
        is_mask=is_mask, oos_mask=oos_mask,
        is_days=is_days, oos_days=oos_days,
        rules=RULES,
        best_existing_is_pass30=0.33,
        run_oos=True,
    )
    result_filt.print_report()

    # ── If OOS-promoted, fan out to NQ and GC ─────────────────────────────
    best = (result_filt
             if result_filt.promoted in ("oos_promoted", "fully_promoted")
             else result_raw)
    use_filter = best.label.endswith("filtered")

    if best.promoted in ("oos_promoted", "fully_promoted"):
        print(f"\n{'='*72}")
        print(f"OOS-PROMOTED → fanning out to NQ and GC")
        print(f"{'='*72}")
        for ak in ("NQ", "GC"):
            print(f"\nLoading {ak} bars...", flush=True)
            bars = list(load_bars_csv(ASSETS[ak]["data_path"]))
            print(f"  {len(bars):,} bars", flush=True)
            pnl = run_asset(ak, bars, with_filter=use_filter)
            days_k = sorted(set(pnl.keys()))
            arr_full = to_series(pnl, days_k)
            # Quick IS/OOS split on this asset's own day index
            is_n = int(len(days_k) * IS_FRACTION)
            is_d = days_k[:is_n]; oos_d = days_k[is_n:]
            is_a = arr_full[:is_n]; oos_a = arr_full[is_n:]
            print(f"\n  {ak} IS:", end=" "); print_quick_stats("IS", is_a, is_d)
            print(f"  {ak} OOS:", end=" "); print_quick_stats("OOS", oos_a, oos_d)

    # ── OOS standalone stats (always print for reference) ─────────────────
    print(f"\n{'─'*72}")
    print("OOS STANDALONE STATS (for reference)")
    print(f"{'─'*72}")
    print_quick_stats("ES GapFill raw OOS",     raw_full[oos_mask],  oos_days)
    print_quick_stats("ES GapFill +filt OOS",   filt_full[oos_mask], oos_days)

    # ── DSR trial count update ─────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("DSR TRIAL ACCOUNTING")
    print(f"{'─'*72}")
    print("  Trials added this evaluation: 2 (raw + filtered)")
    print("  Cumulative project trials: ~27 + 2 = 29")
    print("  Note: gap-fill OOS is counted regardless of promotion outcome.")

    print(f"\n{'='*72}")
    print(f"VERDICT")
    print(f"{'='*72}")
    print(f"  raw variant    : {result_raw.promoted.upper()}")
    print(f"  filter variant : {result_filt.promoted.upper()}")


if __name__ == "__main__":
    main()
