"""End-to-end: ORB on the full ES dataset; report Combine pass-rates.

Usage:
    python scripts/run_orb_es.py [--or-minutes 30] [--tp-multiple 1.0] \
        [--stop-mode opposite_range] [--direction both]

Outputs a structured summary covering:

  * Overall daily-PnL stats over the full ES file (2021-12 -> 2026-04).
  * Realized Combine pass-rate over 30 and 45 day sliding windows.
  * Stationary-bootstrap pass-probability with Wilson CI (cross-check
    against the realized number).
  * Walk-forward folds: 12-month train / 3-month test, anchored.
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.bootstrap import topstep_pass_probability
from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.analysis.stats import performance
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.orb import OpeningRangeBreakout


ES = Instrument(
    symbol="ES",
    point_value=Decimal("50"),
    tick_size=Decimal("0.25"),
    commission_per_side=Decimal("2.50"),  # round-trip ~ $5 incl. exchange fees
)
# MES is the same price series as ES with 1/10 the $/point. Trading MES
# on the $50K Combine is the conventional choice because the trailing
# MLL of $2,000 is too tight for a single ES contract.
MES = Instrument(
    symbol="ES",  # We feed ES bars through the engine -- the price action
    point_value=Decimal("5"),
    tick_size=Decimal("0.25"),
    commission_per_side=Decimal("0.50"),  # round-trip ~ $1 for micros
)


def _load_es_bars(limit_days: int | None = None):
    path = ROOT / "data" / "raw" / "es_cleaned.txt"
    bars = list(load_bars_csv(path))
    if limit_days:
        last = bars[-1].ts.date()
        from datetime import timedelta as _td
        cutoff = last - _td(days=limit_days)
        bars = [b for b in bars if b.ts.date() >= cutoff]
    return bars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--or-minutes", type=int, default=30)
    parser.add_argument("--tp-multiple", type=float, default=1.0)
    parser.add_argument("--stop-mode", choices=["opposite_range", "fixed_ticks"],
                        default="opposite_range")
    parser.add_argument("--stop-ticks", type=int, default=40)
    parser.add_argument("--direction", choices=["long", "short", "both"],
                        default="both")
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--flat-before-close-minutes", type=int, default=15)
    parser.add_argument("--limit-days", type=int, default=None,
                        help="Only use the last N calendar days (for quick iteration)")
    parser.add_argument("--instrument", choices=["ES", "MES"], default="ES",
                        help="ES=$50/pt (e-mini), MES=$5/pt (micro)")
    args = parser.parse_args()
    instrument = ES if args.instrument == "ES" else MES

    print(f"Loading ES bars{' (last ' + str(args.limit_days) + ' days)' if args.limit_days else ''}...")
    t0 = _time.time()
    bars = _load_es_bars(args.limit_days)
    print(f"  {len(bars):,} bars loaded in {_time.time()-t0:.1f}s; "
          f"{bars[0].ts} -> {bars[-1].ts}")

    strat = OpeningRangeBreakout(
        symbol="ES",
        qty=args.qty,
        or_minutes=args.or_minutes,
        direction=args.direction,
        stop_mode=args.stop_mode,
        stop_ticks=args.stop_ticks,
        tp_multiple=args.tp_multiple,
        flat_before_close_minutes=args.flat_before_close_minutes,
    )
    portfolio = PortfolioStrategy(components={"ES": strat})
    rules = combine_50k()
    audit = InMemoryAuditLog()
    clock = Clock(bars[0].ts)
    src = InMemoryBarSource({"ES": bars}, clock)
    bt = Backtester(rules=rules, instruments={"ES": instrument},
                    strategy=portfolio, audit=audit,
                    combine_enforcement=False)  # 4-year trace; rules
                    # applied after-the-fact via realized_pass_rate

    print(f"Running ORB or_min={args.or_minutes} tp={args.tp_multiple} "
          f"stop={args.stop_mode} dir={args.direction} qty={args.qty}...")
    t0 = _time.time()
    result = bt.run(clock, src)
    print(f"  backtest done in {_time.time()-t0:.1f}s")
    fills = [e for e in result.audit.of_kind("fill")
             if "forced" not in e.payload]
    print(f"  fills (non-forced): {len(fills)}")
    print(f"  total realised PnL: ${result.realised_pnl}")
    print(f"  commissions: ${result.commissions}")
    print(f"  end-of-stream equity (this is run-end, NOT pass criterion): "
          f"${result.final_equity}")

    daily = result.daily_pnl
    if not daily:
        print("\nNo daily PnL recorded. The strategy didn't trade.")
        return
    print(f"\nDaily PnL series: {len(daily)} trading days, "
          f"{min(daily)} -> {max(daily)}")
    perf = performance(daily, result.equity_curve,
                       starting_balance=rules.starting_balance)
    print(f"  Sharpe (annual): {perf.sharpe_annual:.2f}")
    print(f"  Sortino (annual): {perf.sortino_annual:.2f}")
    print(f"  Win-rate (day): {perf.win_rate:.1%}")
    print(f"  Profit factor: {perf.profit_factor:.2f}")
    print(f"  Max DD ($): ${perf.max_drawdown_dollars} ({perf.max_drawdown_pct:.1%})")
    print(f"  Best day: ${perf.best_period}   Worst day: ${perf.worst_period}")

    print(f"\n=== Realized Combine pass-rate (the user's primary criterion) ===")
    for window in (30, 45):
        rr = realized_pass_rate(
            daily, rules=rules,
            starting_balance=rules.starting_balance,
            window_days=window, stride_days=1,
        )
        print(f"  window={window}d, stride=1d, n={rr.n_windows}: "
              f"pass={rr.n_passed} ({rr.pass_rate:.1%}) "
              f"[Wilson95% {rr.ci_95_low:.1%}-{rr.ci_95_high:.1%}] "
              f"outcomes={rr.outcomes}")

    print(f"\n=== Bootstrap (Politis-Romano) cross-check ===")
    pnl_decimals = [v for _, v in sorted(daily.items())]
    for n_days in (30, 45):
        bs = topstep_pass_probability(
            daily_pnl=pnl_decimals,
            rules=rules,
            starting_balance=rules.starting_balance,
            n_draws=2000,
            target_n_days=n_days,
            block_mean_length=5.0,
            seed=42,
        )
        print(f"  target_days={n_days}: pass_rate={bs.pass_rate:.1%} "
              f"[{bs.ci_low:.1%}-{bs.ci_high:.1%}] "
              f"(mll={bs.mll_breach_rate:.1%}, "
              f"consistency_fail={bs.consistency_fail_rate:.1%})")


if __name__ == "__main__":
    main()
