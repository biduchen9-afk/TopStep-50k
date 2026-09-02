"""Full IS/OOS Gate 2/3/4 checklist for ORB-TrendHold (same entry +
orb_expansion_gate as v1's ORB, no fixed TP, holds to end-of-day),
across all 3 assets.

screen_orb_trendhold_databento.py's IS-only screen showed the
strongest signal of anything tried this session: ES Sharpe=+0.88
pass30=20.6% GATE2-OK, NQ Sharpe=+1.01 pass30=23.9% GATE2-OK, GC
weaker (Sharpe=+0.24, PF=1.05, fails the light Gate2 proxy on the
0.3 Sharpe threshold but still positive). This runs the real one-touch
harness on all three assets.

Run with: python scripts/evaluate_orb_trendhold_databento.py
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

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

TRENDHOLD_PARAMS = dict(
    qty=1, or_minutes=30, direction="both",
    stop_mode="opposite_range",
    tp_multiple=None, tp_ticks=None,
    flat_before_close_minutes=15,
)


def load_recent_bars(data_path) -> list:
    return [b for b in load_bars_csv(data_path) if b.ts >= RECENT_START]


def run_cell(bars, asset_key, tick_size_f, instrument, gate):
    strat = OpeningRangeBreakout(symbol=asset_key, tick_size=tick_size_f,
                                  daily_filter=gate, **TRENDHOLD_PARAMS)
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: instrument}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, src)


def to_series(daily_pnl: dict, day_index: list[date]) -> np.ndarray:
    out = np.zeros(len(day_index), dtype=float)
    for i, d in enumerate(day_index):
        if d in daily_pnl:
            out[i] = float(daily_pnl[d])
    return out


def main():
    print("=" * 78)
    print("ORB-TrendHold -- FULL IS/OOS 6-GATE CHECKLIST")
    print("=" * 78)

    t0 = _time.time()
    verdicts = []
    for ak in ASSETS:
        cfg = ASSETS[ak]
        print(f"\nLoading {ak} bars...", flush=True)
        tl = _time.time()
        bars = load_recent_bars(cfg["data_path"])
        print(f"  {len(bars):,} bars in {_time.time()-tl:.1f}s", flush=True)

        stats = per_day_session_stats(bars)
        gate = orb_expansion_gate(stats)

        t1 = _time.time()
        result = run_cell(bars, ak, cfg["tick_size_f"], cfg["instrument"], gate)
        days = sorted(result.daily_pnl.keys())
        is_mask, oos_mask, is_days, oos_days = is_oos_split(days, {})
        daily_full = to_series(result.daily_pnl, days)
        label = f"{ak}/ORB_TrendHold"
        verdict = evaluate_strategy(label, daily_full, is_mask, oos_mask,
                                     is_days, oos_days, RULES)
        print(f"  [{_time.time()-t1:.0f}s] {label:<20} -> {verdict.promoted.upper():<14} "
              f"IS-pass30={verdict.is_pass30:.1%} IS-Sharpe={verdict.is_sharpe:+.2f}  "
              f"OOS-pass30={verdict.oos_pass30:.1%} OOS-Sharpe={verdict.oos_sharpe:+.2f}",
              flush=True)
        verdicts.append((ak, verdict))
        del bars

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for ak, v in verdicts:
        print(f"  {ak}/ORB_TrendHold: {v.promoted:<14} IS-pass30={v.is_pass30:.1%} "
              f"OOS-pass30={v.oos_pass30:.1%} IS-Sharpe={v.is_sharpe:+.2f} OOS-Sharpe={v.oos_sharpe:+.2f}")
        if v.gate2:
            failed = [g.name for g in v.gate2 if g.hard and not g.passed]
            failed += [g.name for g in v.gate3 if g.hard and not g.passed]
            failed += [g.name for g in v.gate4 if g.hard and not g.passed]
            if failed:
                print(f"      failed: {', '.join(failed)}")

    survivors = [(ak, v) for ak, v in verdicts if v.promoted == "oos_promoted"]
    print(f"\n  {len(survivors)}/{len(verdicts)} reached OOS_PROMOTED")
    for ak, v in survivors:
        v.print_report()

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
