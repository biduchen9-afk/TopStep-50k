"""Does adding an Asia-session ORB alongside the NY-session ORB help or
dilute performance?

Runs three variants per asset on the timezone-corrected Databento data,
recent window (>= 2021-12-31): NY-only, Asia-only, and NY+Asia combined
(summed daily P&L -- valid because the two sessions never overlap in
time: NY RTH is 09:30-16:00 ET, Asia is 19:00-04:00 ET, so running them
as one portfolio is equivalent to summing two independent single-
strategy backtests).

Both use the same shape found in the R:R sweep (stop_mode=fixed_ticks,
stop_ticks=40, tp_ticks=40, or_minutes=30) and the same vol-expansion
gate (orb_expansion_gate), so this isolates the effect of the SESSION
choice, not a different strategy.

"Asia session" here = the Tokyo/Hong Kong window, 19:00-04:00 US/Eastern
(a standard FX/futures session convention) -- adjust if you had a
different window in mind.
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date, datetime, time as dtime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.analysis.stats import performance
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.evaluation.harness import evaluate_strategy, is_oos_split
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import orb_expansion_gate, per_day_session_stats
from topstep50k.rules import combine_50k
from topstep50k.strategy.orb import OpeningRangeBreakout


ASSETS = {
    "ES": dict(instrument=Instrument(symbol="ES", point_value=Decimal("50"),
                                      tick_size=Decimal("0.25"),
                                      commission_per_side=Decimal("2.50")),
               tick_size_f=0.25, data_path=ROOT / "data" / "raw" / "es_databento.txt"),
    "NQ": dict(instrument=Instrument(symbol="NQ", point_value=Decimal("20"),
                                      tick_size=Decimal("0.25"),
                                      commission_per_side=Decimal("2.50")),
               tick_size_f=0.25, data_path=ROOT / "data" / "raw" / "nq_databento.txt"),
    "GC": dict(instrument=Instrument(symbol="GC", point_value=Decimal("100"),
                                      tick_size=Decimal("0.10"),
                                      commission_per_side=Decimal("2.50")),
               tick_size_f=0.10, data_path=ROOT / "data" / "raw" / "gc_databento.txt"),
}
RULES = combine_50k()
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)

# Winning shape from sweep_orb_rr_databento.py
BASE = dict(or_minutes=30, stop_mode="fixed_ticks", stop_ticks=40,
            tp_ticks=40, tp_multiple=None, flat_before_close_minutes=15)


def run_single(bars, asset_key, tick_size_f, instrument, gate, *, session_open, session_close):
    strat = OpeningRangeBreakout(
        symbol=asset_key, qty=1, direction="both", tick_size=tick_size_f,
        daily_filter=gate, session_open_local=session_open,
        session_close_local=session_close, **BASE,
    )
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: instrument}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, src).daily_pnl


def to_series(daily_pnl: dict, day_index: list[date]) -> np.ndarray:
    out = np.zeros(len(day_index), dtype=float)
    for i, d in enumerate(day_index):
        if d in daily_pnl:
            out[i] = float(daily_pnl[d])
    return out


def report(label: str, arr: np.ndarray, days: list[date]) -> dict:
    daily = {d: Decimal(str(round(float(v), 2))) for d, v in zip(days, arr)}
    nz = int((arr != 0).sum())
    sh = float(arr.mean() / arr.std(ddof=1) * np.sqrt(252)) if arr.std(ddof=1) > 0 else 0.0
    pf_v = (arr[arr > 0].sum() / -arr[arr < 0].sum()) if (arr < 0).any() else float("inf")
    win_rate = (arr > 0).sum() / nz if nz else 0.0
    rr30 = realized_pass_rate(daily, rules=RULES, starting_balance=RULES.starting_balance,
                               window_days=30, stride_days=1)
    rr45 = realized_pass_rate(daily, rules=RULES, starting_balance=RULES.starting_balance,
                               window_days=45, stride_days=1)
    print(f"  {label:<16} nz={nz:>4} total=${arr.sum():>+9,.0f} Sharpe={sh:>+5.2f} "
          f"PF={pf_v:>5.2f} win%={win_rate*100:>4.0f}% "
          f"pass30={rr30.pass_rate:>5.1%} pass45={rr45.pass_rate:>5.1%}")
    return {"sharpe": sh, "pf": pf_v, "pass30": rr30.pass_rate, "pass45": rr45.pass_rate}


def main():
    print("=" * 78)
    print("NY vs ASIA vs NY+ASIA -- ORB SESSION COMPARISON (TZ-CORRECTED)")
    print(f"Shape: {BASE}, vol-expansion gated")
    print("NY = 09:30-16:00 ET  |  Asia = 19:00-04:00 ET (Tokyo/HK window)")
    print("=" * 78)

    all_checklists = []
    for ak in ASSETS:
        cfg = ASSETS[ak]
        print(f"\nLoading {ak}...", flush=True)
        t0 = _time.time()
        bars = [b for b in load_bars_csv(cfg["data_path"]) if b.ts >= RECENT_START]
        print(f"  {len(bars):,} bars in {_time.time()-t0:.0f}s", flush=True)
        stats = per_day_session_stats(bars)
        gate = orb_expansion_gate(stats)

        t0 = _time.time()
        ny_pnl = run_single(bars, ak, cfg["tick_size_f"], cfg["instrument"], gate,
                             session_open=dtime(9, 30), session_close=dtime(16, 0))
        asia_pnl = run_single(bars, ak, cfg["tick_size_f"], cfg["instrument"], gate,
                               session_open=dtime(19, 0), session_close=dtime(4, 0))
        print(f"  ran NY + Asia in {_time.time()-t0:.0f}s", flush=True)

        union_days = sorted(set(ny_pnl.keys()) | set(asia_pnl.keys()))
        ny_arr = to_series(ny_pnl, union_days)
        asia_arr = to_series(asia_pnl, union_days)
        combo_arr = ny_arr + asia_arr

        print(f"\n{'='*78}\n{ak}\n{'='*78}")
        print(f"  {'variant':<16} {'nz':>4} {'total':>11} {'Sharpe':>7} {'PF':>6} "
              f"{'win%':>5} {'pass30':>7} {'pass45':>7}")
        report("NY only", ny_arr, union_days)
        report("Asia only", asia_arr, union_days)
        report("NY + Asia", combo_arr, union_days)

        # Full 6-gate checklist for each variant
        is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
        print()
        for name, arr in [("NY-only", ny_arr), ("Asia-only", asia_arr), ("NY+Asia", combo_arr)]:
            v = evaluate_strategy(f"{ak}/{name}", arr, is_mask, oos_mask, is_days, oos_days, RULES)
            print(f"  [{name}] -> {v.promoted.upper():<14} IS-pass30={v.is_pass30:.1%} "
                  f"OOS-pass30={v.oos_pass30:.1%} OOS-Sharpe={v.oos_sharpe:+.2f}")
            all_checklists.append((ak, name, v))

    print(f"\n{'='*78}\nSUMMARY: does Asia help or dilute?\n{'='*78}")
    for ak in ASSETS:
        rows = [x for x in all_checklists if x[0] == ak]
        by_name = {name: v for _, name, v in rows}
        print(f"  {ak}: NY-only OOS-pass30={by_name['NY-only'].oos_pass30:.1%}  "
              f"vs  NY+Asia OOS-pass30={by_name['NY+Asia'].oos_pass30:.1%}  "
              f"({'HELPS' if by_name['NY+Asia'].oos_pass30 > by_name['NY-only'].oos_pass30 else 'DILUTES/NO CHANGE'})")


if __name__ == "__main__":
    main()
