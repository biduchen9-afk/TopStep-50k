"""Single-file portfolio runner — drop into VS Code, hit Run.

This is the deployable 9-stream OOS-promoted ensemble in one file. Two modes:

    python run_portfolio.py                  # today's daily plan (default)
    python run_portfolio.py --plan 2026-06-25 # plan for a specific date
    python run_portfolio.py --backtest        # full OOS backtest + monthly stats
    python run_portfolio.py --report          # rebuild HTML monthly report

No new strategies, no parameter tuning. The IS pass-rate-aware weights are
frozen from the IS evaluation that produced OOS Sharpe +1.44, pass30 37.2%.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

EASTERN = ZoneInfo("America/New_York")
DATA_DIR = ROOT / "data" / "raw"

ASSETS_DATA = {
    "ES": DATA_DIR / "es_cleaned.txt",
    "NQ": DATA_DIR / "nq_cleaned.txt",
    "GC": DATA_DIR / "gc_cleaned.txt",
}


def cmd_plan(args):
    """Print today's (or a specific date's) actionable trading plan."""
    from topstep50k.data.loaders import load_bars_csv
    from topstep50k.portfolio.live_portfolio import IS_WEIGHTS, build_daily_plan

    if args.plan:
        try:
            for_date = date.fromisoformat(args.plan)
        except ValueError:
            print(f"ERROR: --plan expects YYYY-MM-DD, got {args.plan!r}")
            sys.exit(1)
    else:
        for_date = datetime.now(tz=EASTERN).date()

    print(f"Loading bars for {for_date}...")
    bars_by_asset = {}
    for asset, path in ASSETS_DATA.items():
        if not path.exists():
            print(f"  WARN: {path.name} not found, skipping {asset}")
            continue
        bars_by_asset[asset] = list(load_bars_csv(path))
        print(f"  {asset}: {len(bars_by_asset[asset]):,} bars")

    plan = build_daily_plan(bars_by_asset, for_date)
    plan.print_plan()

    print("WEIGHT BREAKDOWN (frozen IS pass-rate-aware):")
    total = sum(IS_WEIGHTS.values())
    active_keys = [(a.asset, a.strategy) for a in plan.active]
    for (asset, strat), w in sorted(IS_WEIGHTS.items(), key=lambda x: -x[1]):
        tag = " ← ACTIVE" if (asset, strat) in active_keys else ""
        print(f"  {asset}/{strat:<10} {w:.3f}  ({w/total*100:.1f}%){tag}")
    print(f"\nActive weight: {plan.total_active_weight:.3f} / {total:.3f} "
          f"({plan.total_active_weight/total*100:.0f}%)")


def cmd_backtest(args):
    """Run the full ensemble OOS backtest end-to-end."""
    import subprocess
    print("Running full ensemble OOS evaluation (this takes ~3-5 minutes)...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_ensemble_oos.py")],
        cwd=str(ROOT),
    )
    sys.exit(result.returncode)


def cmd_report(args):
    """Regenerate the HTML monthly performance report."""
    import subprocess
    print("Regenerating monthly HTML report...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_monthly_report.py")],
        cwd=str(ROOT),
    )
    if result.returncode == 0:
        html_path = ROOT / "results" / "ensemble_oos_monthly.html"
        print(f"\nReport: {html_path}")
    sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(
        description="9-stream OOS-promoted portfolio runner.")
    p.add_argument("--plan", metavar="YYYY-MM-DD", nargs="?", const="",
                    help="Generate daily plan (default: today's date)")
    p.add_argument("--backtest", action="store_true",
                    help="Run the full OOS ensemble backtest")
    p.add_argument("--report", action="store_true",
                    help="Regenerate the HTML monthly report")
    args = p.parse_args()

    if args.backtest:
        cmd_backtest(args)
    elif args.report:
        cmd_report(args)
    else:
        # Default: daily plan (today if --plan not given or empty)
        if args.plan == "":
            args.plan = None
        cmd_plan(args)


if __name__ == "__main__":
    main()
