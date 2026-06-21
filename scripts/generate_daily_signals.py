"""Generate today's trading plan for the OOS-promoted 9-stream ensemble.

Run this before 9:30 ET each morning:

    python scripts/generate_daily_signals.py

Or for a specific date (useful for testing):

    python scripts/generate_daily_signals.py 2026-06-20

The script loads all historical bar data, computes today's regime gates, and
prints an actionable pre-market plan: which of the 9 streams are active,
why, their IS pass-rate-aware weights, and execution notes per strategy type.

Ensemble summary (frozen, from OOS evaluation):
  IS  (2022-01 → 2025-01): pass30=15.6%, Sharpe=+0.73
  OOS (2025-01 → 2026-04): pass30=37.2%, Sharpe=+1.44 — OOS_PROMOTED
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.data.loaders import load_bars_csv
from topstep50k.portfolio.live_portfolio import IS_WEIGHTS, build_daily_plan

EASTERN = ZoneInfo("America/New_York")

DATA_PATHS = {
    "ES": ROOT / "data" / "raw" / "es_cleaned.txt",
    "NQ": ROOT / "data" / "raw" / "nq_cleaned.txt",
    "GC": ROOT / "data" / "raw" / "gc_cleaned.txt",
}


def main() -> None:
    # ── Determine the trading date ─────────────────────────────────────────
    if len(sys.argv) > 1:
        try:
            for_date = date.fromisoformat(sys.argv[1])
        except ValueError:
            print(f"ERROR: cannot parse date {sys.argv[1]!r} — use YYYY-MM-DD")
            sys.exit(1)
    else:
        for_date = datetime.now(tz=EASTERN).date()

    print(f"Loading bar data for signal plan on {for_date}...", flush=True)

    # ── Load historical bars ───────────────────────────────────────────────
    bars_by_asset: dict = {}
    for asset, path in DATA_PATHS.items():
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {asset}")
            continue
        t0 = _time.time()
        bars = list(load_bars_csv(path))
        print(f"  {asset}: {len(bars):,} bars in {_time.time()-t0:.1f}s", flush=True)
        bars_by_asset[asset] = bars

    if not bars_by_asset:
        print("ERROR: no bar data found. Check data/raw/")
        sys.exit(1)

    # ── Build and print today's plan ───────────────────────────────────────
    plan = build_daily_plan(bars_by_asset, for_date)
    plan.print_plan()

    # ── Weight summary ─────────────────────────────────────────────────────
    active = plan.active
    if active:
        print("WEIGHT BREAKDOWN (IS pass-rate-aware, frozen):")
        total = sum(IS_WEIGHTS.values())
        print(f"  {'Stream':<15} {'IS weight':>10}  {'% of total':>10}")
        print(f"  {'─'*15} {'─'*10}  {'─'*10}")
        for s in sorted(IS_WEIGHTS.items(), key=lambda x: -x[1]):
            sym, strat = s[0]
            w = s[1]
            tag = " ← ACTIVE" if (sym, strat) in [(a.asset, a.strategy) for a in active] else ""
            print(f"  {sym}/{strat:<12} {w:>10.3f}  {w/total*100:>9.1f}%{tag}")
        print(f"\n  Active weight today: {plan.total_active_weight:.3f} / {total:.3f} "
              f"({plan.total_active_weight/total*100:.0f}% of full ensemble)\n")


if __name__ == "__main__":
    main()
