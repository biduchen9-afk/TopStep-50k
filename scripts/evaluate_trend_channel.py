"""Strategy #9: Trend Channel Breakout — promotion checklist evaluation.

Pre-committed before any data is viewed:
  Predicted edge direction : TREND (long on 10-day high break, short on low break)
  Expected IS pass30 range : 10-25%
  Regime placement         : TRENDING markets (directional days)
  IS trade freq expected   : ~30-50/year (10-day channel generates signals
                              on sustained directional moves)

Parameters (pre-committed, Donchian/Hurst-derived):
  lookback_days  = 10  (10-day channel — shorter than classic 20-day for more signals)
  atr_lookback   = 14  (ATR: standard)
  stop_atr_mult  = 2.0 (2× ATR stop, Hurst et al. 2013)
  exit_lookback  = 5   (reverse on 5-day opposite extreme)
  max_hold_days  = 15  (cap exposure)

Evaluated on ES, NQ, GC (trend following works across asset classes per literature).
DSR accounting: +2 trials (raw + filtered, though filter may not change much) → ~37.
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
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.evaluation.harness import evaluate_strategy, is_oos_split
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.trend_channel import TrendChannel


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
TC_PARAMS = dict(qty=1, lookback_days=10, atr_lookback=14,
                  stop_atr_mult=2.0, exit_lookback=5, max_hold_days=15)


def run_asset(asset_key: str, bars) -> dict[date, Decimal]:
    asset = ASSETS[asset_key]
    strat = TrendChannel(symbol=asset_key, tick_size=asset["tick_size_f"], **TC_PARAMS)
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
    print(f"  {label:<35}: n_days={arr.size} nz={nz:>4} "
          f"total=${arr.sum():>+9,.0f} Sharpe={sh:>+5.2f} "
          f"PF={pf_v:>4.2f} MaxDD=${mdd:>+8,.0f}")
    pnl_d = {d: Decimal(str(round(float(v), 2))) for d, v in zip(days, arr)}
    rr30 = realized_pass_rate(pnl_d, rules=RULES,
                               starting_balance=RULES.starting_balance,
                               window_days=30, stride_days=1)
    rr45 = realized_pass_rate(pnl_d, rules=RULES,
                               starting_balance=RULES.starting_balance,
                               window_days=45, stride_days=1)
    print(f"  {'':>35}  pass30={rr30.pass_rate:.1%} ({rr30.n_passed}/{rr30.n_windows})  "
          f"pass45={rr45.pass_rate:.1%} ({rr45.n_passed}/{rr45.n_windows})")


def main():
    print("=" * 72)
    print("STRATEGY #9: TREND CHANNEL BREAKOUT — PROMOTION CHECKLIST")
    print("Pre-committed: Donchian 10-day channel, 2×ATR stop, 5-day exit")
    print("=" * 72)

    results_by_asset = {}
    for ak, aconf in ASSETS.items():
        print(f"\nLoading {ak} bars...", flush=True)
        t0 = _time.time()
        bars = list(load_bars_csv(aconf["data_path"]))
        print(f"  {len(bars):,} bars in {_time.time() - t0:.1f}s", flush=True)

        print(f"Running {ak} TrendChannel...", flush=True)
        t0 = _time.time()
        pnl = run_asset(ak, bars)
        print(f"  done in {_time.time() - t0:.1f}s  ({len(pnl)} days)", flush=True)
        results_by_asset[ak] = pnl

    # Use ES union days for IS/OOS split
    es_pnl = results_by_asset["ES"]
    union_days = sorted(es_pnl.keys())
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
    print(f"\nIS : {is_days[0]} -> {is_days[-1]} ({len(is_days)} d)")
    print(f"OOS: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d)")

    print(f"\n{'─'*72}")
    print("QUICK STATS (IS only) — per asset")
    print(f"{'─'*72}")
    results_checklist = {}
    for ak in ASSETS:
        pnl = results_by_asset[ak]
        arr = to_series(pnl, union_days)
        print_quick_stats(f"{ak} TrendChannel IS", arr[is_mask], is_days)

        print(f"\n{'─'*72}")
        print(f"GATE-BY-GATE CHECKLIST: {ak}")
        print(f"{'─'*72}")
        res = evaluate_strategy(
            label=f"TrendChannel-{ak}",
            daily_full=arr,
            is_mask=is_mask, oos_mask=oos_mask,
            is_days=is_days, oos_days=oos_days,
            rules=RULES,
            best_existing_is_pass30=0.33,
            run_oos=True,
        )
        res.print_report()
        results_checklist[ak] = (res, arr)

    print(f"\n{'─'*72}")
    print("OOS STANDALONE STATS (reference)")
    print(f"{'─'*72}")
    for ak, (res, arr) in results_checklist.items():
        print_quick_stats(f"{ak} TrendChannel OOS", arr[oos_mask], oos_days)

    print(f"\n{'─'*72}")
    print("DSR TRIAL ACCOUNTING")
    print(f"{'─'*72}")
    print("  Trials added this evaluation: 3 (ES + NQ + GC, single variant each)")
    print("  Cumulative project trials: ~35 + 3 = 38")

    print(f"\n{'='*72}")
    print("VERDICT")
    print(f"{'='*72}")
    for ak, (res, _) in results_checklist.items():
        print(f"  TrendChannel-{ak} : {res.promoted.upper()}")


if __name__ == "__main__":
    main()
