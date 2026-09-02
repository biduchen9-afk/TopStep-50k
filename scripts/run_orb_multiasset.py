"""Multi-asset ORB on ES + NQ + GC, sized as micros.

Hypothesis: single-instrument ORB on ES has a payoff distribution
(occasional fat-tail trend days) that fails the consistency rule
(best day <= 50% of total). Trading the same setup across three
uncorrelated instruments should naturally flatten the daily-PnL
distribution because winning days on one rarely coincide with
equally-sized winning days on the others.

Sizing: micros across the board (MES $5/pt, MNQ $2/pt, MGC $10/pt).
Total contracts well under the 50-micro account cap.
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from datetime import timedelta as _td
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


# Micro contracts -- $-per-point divided by 10 from the standard.
MES = Instrument(symbol="ES", point_value=Decimal("5"),
                 tick_size=Decimal("0.25"), commission_per_side=Decimal("0.50"))
MNQ = Instrument(symbol="NQ", point_value=Decimal("2"),
                 tick_size=Decimal("0.25"), commission_per_side=Decimal("0.50"))
MGC = Instrument(symbol="GC", point_value=Decimal("10"),
                 tick_size=Decimal("0.10"), commission_per_side=Decimal("0.50"))

INSTRUMENTS = {"ES": MES, "NQ": MNQ, "GC": MGC}


def _load_aligned(limit_days: int | None = None):
    """Load all three bar streams; clip to common date range."""
    bars: dict[str, list] = {}
    for sym in ("ES", "NQ", "GC"):
        path = ROOT / "data" / "raw" / f"{sym.lower()}_cleaned.txt"
        bars[sym] = list(load_bars_csv(path))
    # Clip to the latest "first" and earliest "last" so every symbol
    # contributes the full common window.
    start = max(b[0].ts for b in bars.values())
    end = min(b[-1].ts for b in bars.values())
    if limit_days:
        start = max(start, end - _td(days=limit_days))
    for s in bars:
        bars[s] = [b for b in bars[s] if start <= b.ts <= end]
    return bars, start, end


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--or-minutes", type=int, default=30)
    p.add_argument("--tp-multiple", type=float, default=2.0)
    p.add_argument("--stop-mode", choices=["opposite_range", "fixed_ticks"],
                   default="fixed_ticks")
    p.add_argument("--stop-ticks", type=int, default=20)
    p.add_argument("--direction", choices=["long", "short", "both"], default="long")
    p.add_argument("--qty-es", type=int, default=3)
    p.add_argument("--qty-nq", type=int, default=3)
    p.add_argument("--qty-gc", type=int, default=2)
    p.add_argument("--flat-before-close-minutes", type=int, default=15)
    p.add_argument("--limit-days", type=int, default=None)
    args = p.parse_args()

    print("Loading ES + NQ + GC...")
    t0 = _time.time()
    bars, start, end = _load_aligned(args.limit_days)
    for s, b in bars.items():
        print(f"  {s}: {len(b):,} bars")
    print(f"  common window {start.date()} -> {end.date()} ({(end-start).days}d)")
    print(f"  loaded in {_time.time()-t0:.1f}s")

    components = {}
    for sym, qty in (("ES", args.qty_es), ("NQ", args.qty_nq), ("GC", args.qty_gc)):
        if qty <= 0:
            continue
        inst = INSTRUMENTS[sym]
        components[sym] = OpeningRangeBreakout(
            symbol=sym, qty=qty,
            or_minutes=args.or_minutes,
            direction=args.direction,
            stop_mode=args.stop_mode,
            stop_ticks=args.stop_ticks,
            tp_multiple=args.tp_multiple,
            flat_before_close_minutes=args.flat_before_close_minutes,
            tick_size=float(inst.tick_size),
        )

    portfolio = PortfolioStrategy(components=components)
    # We're simulating MES/MNQ/MGC trading by feeding ES/NQ/GC price bars
    # through Instruments configured with micro $-per-point. The
    # position-cap arithmetic needs to know these symbols are micros so
    # it scales 1:1 against the 50-micro cap (not 10:1 as for standard).
    from dataclasses import replace
    rules = replace(
        combine_50k(),
        micro_symbols=frozenset({"ES", "NQ", "GC", "MES", "MNQ", "MGC"}),
    )
    audit = InMemoryAuditLog()
    clock = Clock(start)
    src = InMemoryBarSource(bars, clock)
    bt = Backtester(rules=rules, instruments=INSTRUMENTS,
                    strategy=portfolio, audit=audit,
                    combine_enforcement=False)

    print(f"\nRunning multi-asset ORB (or={args.or_minutes} tp={args.tp_multiple} "
          f"stop={args.stop_mode}/{args.stop_ticks}tk dir={args.direction}, "
          f"qty ES={args.qty_es} NQ={args.qty_nq} GC={args.qty_gc} micros)...")
    t0 = _time.time()
    result = bt.run(clock, src)
    print(f"  backtest done in {_time.time()-t0:.1f}s")

    fills = [e for e in result.audit.of_kind("fill") if "forced" not in e.payload]
    by_sym = {}
    for f in fills:
        by_sym.setdefault(f.payload["symbol"], 0)
        by_sym[f.payload["symbol"]] += 1
    print(f"  fills by symbol: {by_sym}  total={len(fills)}")
    print(f"  total realised PnL: ${result.realised_pnl}")
    print(f"  commissions: ${result.commissions}")

    daily = result.daily_pnl
    print(f"\nDaily PnL: {len(daily)} trading days")
    perf = performance(daily, result.equity_curve,
                       starting_balance=rules.starting_balance)
    print(f"  Sharpe: {perf.sharpe_annual:.2f}   Sortino: {perf.sortino_annual:.2f}")
    print(f"  Win-rate (day): {perf.win_rate:.1%}   Profit factor: {perf.profit_factor:.2f}")
    print(f"  Max DD ($): ${perf.max_drawdown_dollars} ({perf.max_drawdown_pct:.1%})")
    print(f"  Best day: ${perf.best_period}   Worst day: ${perf.worst_period}")

    print(f"\n=== Realized Combine pass-rate (primary criterion) ===")
    for window in (30, 45):
        rr = realized_pass_rate(
            daily, rules=rules,
            starting_balance=rules.starting_balance,
            window_days=window, stride_days=1,
        )
        print(f"  window={window}d  n={rr.n_windows}: pass={rr.n_passed} "
              f"({rr.pass_rate:.1%}) [Wilson95 {rr.ci_95_low:.1%}-{rr.ci_95_high:.1%}]")
        print(f"    outcomes={rr.outcomes}")

    print(f"\n=== Bootstrap cross-check ===")
    pnl_dec = [v for _, v in sorted(daily.items())]
    for n in (30, 45):
        bs = topstep_pass_probability(
            daily_pnl=pnl_dec, rules=rules,
            starting_balance=rules.starting_balance,
            n_draws=2000, target_n_days=n, block_mean_length=5.0, seed=42,
        )
        print(f"  target_days={n}: pass={bs.pass_rate:.1%} "
              f"[{bs.ci_low:.1%}-{bs.ci_high:.1%}]  "
              f"(mll={bs.mll_breach_rate:.1%}, "
              f"consistency_fail={bs.consistency_fail_rate:.1%})")


if __name__ == "__main__":
    main()
