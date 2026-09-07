"""Strategy #11: Sweep → MSS/Impulse — Pine Script port evaluation on GC.

Pre-committed configuration (matches user's TradingView inputs):
  Trade session     : BOTH (London + NY)
  Session timezone  : Asia/Bangkok
  London window     : 12:00-14:00 Bangkok (= 05:00-07:00 UTC)
  NY window         : 18:00-19:30 Bangkok (= 11:00-12:30 UTC)
  Risk:Reward       : 2.0
  5m ATR length     : 14
  5m impulse body   : >= ATR × 0.65   (user-tuned vs default 0.40)
  5m close-in-extr. : 0.65            (user-tuned vs default 0.55)
  5m MSS lookback   : 5               (user-tuned vs default 6)
  Max wait after sweep: 24 (5m bars)
  Force close       : 23:59 Bangkok

Asset: GC futures (TopStep tradable). Note: original TradingView result
was on XAUUSD spot CFD — instrument differences (tick, commission,
overnight gaps) make exact reproduction impossible.

Sizing:
  IS  : 1 contract (apples-to-apples vs existing 9-stream ensemble)
  OOS : two reports — 1 contract AND 3 contracts (≈ "100% of equity"
        scaled for TopStep $50K)

DSR trial accounting: +3 trials (IS-1c, OOS-1c, OOS-3c) — note that the
session windows and impulse params were user-tuned on XAUUSD, which is
itself an implicit set of free parameters. Cumulative: ~40 + 3 = 43.
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
from topstep50k.strategy.sweep_mss import SweepMSS


GC_INSTRUMENT = Instrument(
    symbol="GC", point_value=Decimal("100"),
    tick_size=Decimal("0.10"),
    commission_per_side=Decimal("2.50"),
)
GC_TICK = 0.10
GC_DATA = ROOT / "data" / "raw" / "gc_cleaned.txt"
RULES = combine_50k()


def run_strategy(bars, qty: int) -> dict[date, Decimal]:
    strat = SweepMSS(symbol="GC", qty=qty, tick_size=GC_TICK)
    pf = PortfolioStrategy(components={"GC": strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({"GC": bars}, clk)
    bt = Backtester(
        rules=RULES, instruments={"GC": GC_INSTRUMENT},
        strategy=pf, audit=InMemoryAuditLog(),
        combine_enforcement=False,
    )
    return bt.run(clk, src).daily_pnl


def to_series(daily_pnl: dict, day_index: list[date]) -> np.ndarray:
    out = np.zeros(len(day_index), dtype=float)
    for i, d in enumerate(day_index):
        if d in daily_pnl:
            out[i] = float(daily_pnl[d])
    return out


def quick_stats(label: str, arr: np.ndarray, days: list[date]) -> None:
    nz = int((arr != 0).sum())
    if arr.size < 2:
        print(f"  {label}: insufficient data")
        return
    sh = (arr.mean() / arr.std(ddof=1)) * np.sqrt(252) if arr.std(ddof=1) > 0 else 0
    pf_v = (arr[arr > 0].sum() / -arr[arr < 0].sum()
             if (arr < 0).any() else float("inf"))
    eq = np.cumsum(arr)
    mdd = float((eq - np.maximum.accumulate(eq)).min())
    ev = float(arr.sum() / max(1, arr.size))
    print(f"  {label:<35}: n_days={arr.size} nz={nz:>4} "
          f"total=${arr.sum():>+10,.0f} EV=${ev:>+6.1f}/day Sharpe={sh:>+5.2f} "
          f"PF={pf_v:>4.2f} MaxDD=${mdd:>+9,.0f}")
    pnl_d = {d: Decimal(str(round(float(v), 2))) for d, v in zip(days, arr)}
    rr30 = realized_pass_rate(pnl_d, rules=RULES,
                               starting_balance=RULES.starting_balance,
                               window_days=30, stride_days=1)
    rr45 = realized_pass_rate(pnl_d, rules=RULES,
                               starting_balance=RULES.starting_balance,
                               window_days=45, stride_days=1)
    print(f"  {'':>35}  pass30={rr30.pass_rate:.1%} ({rr30.n_passed}/{rr30.n_windows})  "
          f"pass45={rr45.pass_rate:.1%} ({rr45.n_passed}/{rr45.n_windows})")
    # Outcome breakdown for the 30-day windows (pass / MLL breach / consistency / no target)
    oc = rr30.outcomes
    print(f"  {'':>35}  30d outcomes: pass={oc.get('pass',0)} "
          f"mll={oc.get('mll_breach',0)} cons={oc.get('consistency_fail',0)} "
          f"no_target={oc.get('no_target',0)}")


def main():
    print("=" * 72)
    print("STRATEGY #11: SWEEP → MSS/IMPULSE (Pine Script port) — GC")
    print("Pre-committed: Bangkok windows 12-14 / 18-19:30, ATR×0.65, MSS=5")
    print("=" * 72)

    print(f"\nLoading GC bars from {GC_DATA.name}...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(GC_DATA))
    print(f"  {len(bars):,} bars in {_time.time()-t0:.1f}s", flush=True)

    # ── Run IS with 1 contract ─────────────────────────────────────────
    print("\nRunning SweepMSS on GC (1 contract, full history)...", flush=True)
    t0 = _time.time()
    pnl_1c = run_strategy(bars, qty=1)
    print(f"  done in {_time.time()-t0:.1f}s ({len(pnl_1c)} days)", flush=True)

    union_days = sorted(pnl_1c.keys())
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
    print(f"\nIS : {is_days[0]} -> {is_days[-1]} ({len(is_days)} d)")
    print(f"OOS: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d)")

    arr_1c = to_series(pnl_1c, union_days)
    is_arr_1c  = arr_1c[is_mask]
    oos_arr_1c = arr_1c[oos_mask]

    # ── IS report (1 contract) ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("IS QUICK STATS — 1 contract")
    print(f"{'─'*72}")
    quick_stats("SweepMSS GC IS (1c)", is_arr_1c, is_days)

    print(f"\n{'─'*72}")
    print("IS GATE-BY-GATE CHECKLIST (1 contract)")
    print(f"{'─'*72}")
    res_is = evaluate_strategy(
        label="SweepMSS-GC-1c",
        daily_full=arr_1c,
        is_mask=is_mask, oos_mask=oos_mask,
        is_days=is_days, oos_days=oos_days,
        rules=RULES,
        best_existing_is_pass30=0.156,  # 9-stream IS pass30
        run_oos=False,  # we'll do OOS separately with both sizes
    )
    res_is.print_report()

    # ── OOS 1 contract ─────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("OOS QUICK STATS — 1 contract")
    print(f"{'─'*72}")
    quick_stats("SweepMSS GC OOS (1c)", oos_arr_1c, oos_days)

    # ── OOS 3 contracts (≈ "100% equity" scaled) ───────────────────────
    print("\nRunning SweepMSS on GC OOS (3 contracts)...", flush=True)
    t0 = _time.time()
    pnl_3c = run_strategy(bars, qty=3)
    arr_3c = to_series(pnl_3c, union_days)
    oos_arr_3c = arr_3c[oos_mask]
    print(f"  done in {_time.time()-t0:.1f}s", flush=True)

    print(f"\n{'─'*72}")
    print("OOS QUICK STATS — 3 contracts (scaled, '100% of equity' analog)")
    print(f"{'─'*72}")
    quick_stats("SweepMSS GC OOS (3c)", oos_arr_3c, oos_days)

    print(f"\n{'─'*72}")
    print("OOS DRAWDOWN & MLL ANALYSIS")
    print(f"{'─'*72}")
    for label, oos_arr in [("1 contract", oos_arr_1c), ("3 contracts", oos_arr_3c)]:
        if oos_arr.size == 0:
            continue
        eq = np.cumsum(oos_arr)
        peaks = np.maximum.accumulate(eq)
        dd = eq - peaks
        # Approximate TopStep $2000 trailing MLL breach count: any time
        # equity drops more than $2000 from its peak.
        breach_days = int((dd <= -2000).sum())
        worst_dd = float(dd.min())
        worst_day_loss = float(oos_arr.min())
        max_day_gain = float(oos_arr.max())
        print(f"  {label}:")
        print(f"    Worst running drawdown vs peak: ${worst_dd:+,.0f}")
        print(f"    Days deeper than -$2,000 (MLL proxy): {breach_days} of {oos_arr.size}")
        print(f"    Worst single-day PnL: ${worst_day_loss:+,.0f}")
        print(f"    Best single-day PnL: ${max_day_gain:+,.0f}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  IS (1c) Sharpe={(is_arr_1c.mean()/is_arr_1c.std(ddof=1)*np.sqrt(252)) if is_arr_1c.std(ddof=1)>0 else 0:+.2f}, "
          f"total=${is_arr_1c.sum():+,.0f}, EV=${is_arr_1c.sum()/max(1,is_arr_1c.size):+.2f}/day")
    print(f"  OOS (1c) Sharpe={(oos_arr_1c.mean()/oos_arr_1c.std(ddof=1)*np.sqrt(252)) if oos_arr_1c.std(ddof=1)>0 else 0:+.2f}, "
          f"total=${oos_arr_1c.sum():+,.0f}, EV=${oos_arr_1c.sum()/max(1,oos_arr_1c.size):+.2f}/day")
    print(f"  OOS (3c) Sharpe={(oos_arr_3c.mean()/oos_arr_3c.std(ddof=1)*np.sqrt(252)) if oos_arr_3c.std(ddof=1)>0 else 0:+.2f}, "
          f"total=${oos_arr_3c.sum():+,.0f}, EV=${oos_arr_3c.sum()/max(1,oos_arr_3c.size):+.2f}/day")

    print(f"\n  DSR trials added: 3 (IS-1c, OOS-1c, OOS-3c)")
    print(f"  Cumulative project trials: ~40 + 3 = 43")


if __name__ == "__main__":
    main()
