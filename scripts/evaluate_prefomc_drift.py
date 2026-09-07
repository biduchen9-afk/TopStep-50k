"""Strategy #6: Pre-FOMC Drift — promotion checklist evaluation.

Pre-committed before any data is viewed:
  Predicted edge direction : LONG ONLY (overnight FOMC eve → FOMC morning)
  Expected IS pass30 range : 5-20% (only 8 FOMC meetings/year → ~40 IS events)
  Regime placement         : MACRO EVENT — only on specific calendar dates
  IS trade freq expected   : ~8/year (one per FOMC meeting)

Parameters (pre-committed, directly from Lucca & Moench 2015):
  entry_time : 15:55 ET on FOMC eve (near RTH close)
  exit_time  : 09:35 ET on FOMC announcement day (shortly after RTH open)
  direction  : LONG ONLY (documented 49bp average; short shows no robust edge)
  stop       : None (hold overnight as paper documents)

Note: Very low trade frequency (~8/year). Pass-rate windows may have high
variance due to sparse signal. Evaluated on ES only (equity-index focus of paper).
DSR accounting: +1 trial → cumulative ~32.
(Single variant — no filter variant because the FOMC filter IS the strategy.)
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
from topstep50k.strategy.prefomc_drift import PreFOMCDrift


RULES = combine_50k()

ES_INSTRUMENT = Instrument(
    symbol="ES", point_value=Decimal("50"),
    tick_size=Decimal("0.25"),
    commission_per_side=Decimal("2.50"),
)
DATA_PATH = ROOT / "data" / "raw" / "es_cleaned.txt"


def run_es(bars) -> dict[date, Decimal]:
    strat = PreFOMCDrift(symbol="ES", qty=1)
    pf = PortfolioStrategy(components={"ES": strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({"ES": bars}, clk)
    bt = Backtester(rules=RULES, instruments={"ES": ES_INSTRUMENT},
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
    print("STRATEGY #6: PRE-FOMC DRIFT — PROMOTION CHECKLIST")
    print("Pre-committed: long FOMC-eve close → FOMC-day open, Lucca & Moench 2015")
    print("=" * 72)

    print("\nLoading ES bars...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(DATA_PATH))
    print(f"  {len(bars):,} bars in {_time.time() - t0:.1f}s", flush=True)

    print("\nRunning ES PreFOMCDrift...", flush=True)
    t0 = _time.time()
    pnl = run_es(bars)
    print(f"  done in {_time.time() - t0:.1f}s  ({len(pnl)} active days)", flush=True)

    union_days = sorted(pnl.keys())
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
    print(f"\nIS : {is_days[0]} -> {is_days[-1]} ({len(is_days)} d)")
    print(f"OOS: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d)")

    arr_full = to_series(pnl, union_days)

    # Count actual FOMC trades in IS and OOS
    is_nz = int((arr_full[is_mask] != 0).sum())
    oos_nz = int((arr_full[oos_mask] != 0).sum())
    print(f"\nFOMC events in IS: {is_nz} | OOS: {oos_nz}")

    print(f"\n{'─'*72}")
    print("QUICK STATS (IS only)")
    print(f"{'─'*72}")
    print_quick_stats("ES PreFOMCDrift", arr_full[is_mask], is_days)

    print(f"\n{'─'*72}")
    print("GATE-BY-GATE CHECKLIST")
    print(f"{'─'*72}")
    result = evaluate_strategy(
        label="PreFOMCDrift-ES",
        daily_full=arr_full,
        is_mask=is_mask, oos_mask=oos_mask,
        is_days=is_days, oos_days=oos_days,
        rules=RULES,
        best_existing_is_pass30=0.33,
        run_oos=True,
    )
    result.print_report()

    print(f"\n{'─'*72}")
    print("OOS STANDALONE STATS (reference)")
    print(f"{'─'*72}")
    print_quick_stats("ES PreFOMCDrift OOS", arr_full[oos_mask], oos_days)

    print(f"\n{'─'*72}")
    print("DSR TRIAL ACCOUNTING")
    print(f"{'─'*72}")
    print("  Trials added this evaluation: 1 (single variant — FOMC filter IS the strategy)")
    print("  Cumulative project trials: ~31 + 1 = 32")

    print(f"\n{'='*72}")
    print("VERDICT")
    print(f"{'='*72}")
    print(f"  PreFOMCDrift-ES : {result.promoted.upper()}")
    print(f"\nNote: Low trade frequency (~8/year) means t_stat and pass30 may")
    print(f"be volatile. Academic effect well-documented but sparse in any")
    print(f"single instrument's time series.")


if __name__ == "__main__":
    main()
