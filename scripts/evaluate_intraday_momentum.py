"""Strategy #5: Intraday Momentum — promotion checklist evaluation.

Pre-committed before any data is viewed:
  Predicted edge direction : MOMENTUM (long on up-open, short on down-open)
  Expected IS pass30 range : 15-30%
  Regime placement         : TRENDING / directional sessions
  IS trade freq expected   : ~3-5/month per asset (triggers on min 0.10% signal)

Parameters (pre-committed, literature-derived from Gao et al. 2018):
  signal_minutes   = 30     (first half-hour, as in paper)
  min_signal_pct   = 0.0010 (0.10% minimum move — noise filter)
  stop_multiple    = 2.0    (2× signal range, literature-wide stop)
  time_stop_minutes= 180    (abandon after 3 h if no stop hit)
  exit_time_local  = 15:45  (flatten before close auction noise)

Evaluated on ES only first. If OOS-promoted, fan out to NQ/GC.
DSR accounting: +2 trials (raw + filtered) → cumulative ~31.
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date
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
from topstep50k.evaluation.harness import evaluate_strategy, is_oos_split
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.intraday_momentum import IntradayMomentum


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

IM_PARAMS = dict(
    qty=1,
    signal_minutes=30,
    min_signal_pct=0.0010,
    stop_multiple=2.0,
    time_stop_minutes=180,
)


def run_asset(asset_key: str, bars, with_filter: bool = False) -> dict[date, Decimal]:
    asset = ASSETS[asset_key]
    filt = (lambda d: not is_high_impact_day(d)) if with_filter else None
    strat = IntradayMomentum(symbol=asset_key,
                              tick_size=asset["tick_size_f"],
                              daily_filter=filt,
                              **IM_PARAMS)
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
    print("STRATEGY #5: INTRADAY MOMENTUM — PROMOTION CHECKLIST")
    print("Pre-committed: long up-open / short down-open, IS pass30 15-30%")
    print("=" * 72)

    print("\nLoading ES bars...", flush=True)
    t0 = _time.time()
    es_bars = list(load_bars_csv(ASSETS["ES"]["data_path"]))
    print(f"  {len(es_bars):,} bars in {_time.time() - t0:.1f}s", flush=True)

    print("\nRunning ES IntradayMomentum (no filter)...", flush=True)
    t0 = _time.time()
    es_pnl_raw = run_asset("ES", es_bars, with_filter=False)
    print(f"  done in {_time.time() - t0:.1f}s  ({len(es_pnl_raw)} days)", flush=True)

    print("Running ES IntradayMomentum (skip FOMC+NFP days)...", flush=True)
    t0 = _time.time()
    es_pnl_filt = run_asset("ES", es_bars, with_filter=True)
    print(f"  done in {_time.time() - t0:.1f}s  ({len(es_pnl_filt)} days)", flush=True)

    union_days = sorted(set(es_pnl_raw) | set(es_pnl_filt))
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
    print(f"\nIS : {is_days[0]} -> {is_days[-1]} ({len(is_days)} d)")
    print(f"OOS: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d)")

    raw_full  = to_series(es_pnl_raw,  union_days)
    filt_full = to_series(es_pnl_filt, union_days)

    print(f"\n{'─'*72}")
    print("QUICK STATS (IS only)")
    print(f"{'─'*72}")
    print_quick_stats("ES IntradayMomentum raw",    raw_full[is_mask],  is_days)
    print_quick_stats("ES IntradayMomentum +filter", filt_full[is_mask], is_days)

    print(f"\n{'─'*72}")
    print("GATE-BY-GATE CHECKLIST (raw variant)")
    print(f"{'─'*72}")
    result_raw = evaluate_strategy(
        label="IntradayMomentum-ES-raw",
        daily_full=raw_full,
        is_mask=is_mask, oos_mask=oos_mask,
        is_days=is_days, oos_days=oos_days,
        rules=RULES,
        best_existing_is_pass30=0.33,
        run_oos=True,
    )
    result_raw.print_report()

    print(f"\n{'─'*72}")
    print("GATE-BY-GATE CHECKLIST (+FOMC/NFP filter)")
    print(f"{'─'*72}")
    result_filt = evaluate_strategy(
        label="IntradayMomentum-ES-filtered",
        daily_full=filt_full,
        is_mask=is_mask, oos_mask=oos_mask,
        is_days=is_days, oos_days=oos_days,
        rules=RULES,
        best_existing_is_pass30=0.33,
        run_oos=True,
    )
    result_filt.print_report()

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
            is_n = int(len(days_k) * IS_FRACTION)
            is_d = days_k[:is_n]; oos_d = days_k[is_n:]
            is_a = arr_full[:is_n]; oos_a = arr_full[is_n:]
            print(f"\n  {ak} IS:", end=" "); print_quick_stats("IS", is_a, is_d)
            print(f"  {ak} OOS:", end=" "); print_quick_stats("OOS", oos_a, oos_d)

    print(f"\n{'─'*72}")
    print("OOS STANDALONE STATS (reference)")
    print(f"{'─'*72}")
    print_quick_stats("ES IntradayMom raw OOS",   raw_full[oos_mask],  oos_days)
    print_quick_stats("ES IntradayMom +filt OOS", filt_full[oos_mask], oos_days)

    print(f"\n{'─'*72}")
    print("DSR TRIAL ACCOUNTING")
    print(f"{'─'*72}")
    print("  Trials added this evaluation: 2 (raw + filtered)")
    print("  Cumulative project trials: ~29 + 2 = 31")

    print(f"\n{'='*72}")
    print("VERDICT")
    print(f"{'='*72}")
    print(f"  raw variant    : {result_raw.promoted.upper()}")
    print(f"  filter variant : {result_filt.promoted.upper()}")


if __name__ == "__main__":
    main()
