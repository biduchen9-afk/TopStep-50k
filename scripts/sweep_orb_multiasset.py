"""Systematic parameter sweep of multi-asset ORB.

Builds a small grid of OR length, TP multiple, stop type, and direction,
runs each combination on the full ES+NQ+GC dataset, and prints a single
ranked table including pass-rate, MaxDD, and the binding-constraint
breakdown. Then runs sensitivity_sweep on those results to compute the
deflated-Sharpe winner and the matrix-level PBO.
"""

from __future__ import annotations

import sys
import time as _time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.analysis.sensitivity import sensitivity_sweep
from topstep50k.analysis.stats import performance
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.orb import OpeningRangeBreakout


MES = Instrument(symbol="ES", point_value=Decimal("5"),
                 tick_size=Decimal("0.25"),
                 commission_per_side=Decimal("0.37"))
MNQ = Instrument(symbol="NQ", point_value=Decimal("2"),
                 tick_size=Decimal("0.25"),
                 commission_per_side=Decimal("0.37"))
MGC = Instrument(symbol="GC", point_value=Decimal("10"),
                 tick_size=Decimal("0.10"),
                 commission_per_side=Decimal("0.74"))
INSTRUMENTS = {"ES": MES, "NQ": MNQ, "GC": MGC}
RULES = replace(combine_50k(),
                micro_symbols=frozenset({"ES", "NQ", "GC", "MES", "MNQ", "MGC"}))


def _load_aligned():
    bars = {}
    for sym in ("ES", "NQ", "GC"):
        bars[sym] = list(load_bars_csv(ROOT / "data" / "raw" / f"{sym.lower()}_cleaned.txt"))
    start = max(b[0].ts for b in bars.values())
    end = min(b[-1].ts for b in bars.values())
    for s in bars:
        bars[s] = [b for b in bars[s] if start <= b.ts <= end]
    return bars, start, end


def run_combo(bars, start, *, or_min, tp, stop_mode, stop_tk, direction,
              qty_es=3, qty_nq=3, qty_gc=2):
    components = {}
    for sym, qty in (("ES", qty_es), ("NQ", qty_nq), ("GC", qty_gc)):
        if qty <= 0:
            continue
        components[sym] = OpeningRangeBreakout(
            symbol=sym, qty=qty, or_minutes=or_min, direction=direction,
            stop_mode=stop_mode, stop_ticks=stop_tk, tp_multiple=tp,
            flat_before_close_minutes=15,
            tick_size=float(INSTRUMENTS[sym].tick_size),
        )
    portfolio = PortfolioStrategy(components=components)
    clock = Clock(start)
    src = InMemoryBarSource(bars, clock)
    bt = Backtester(rules=RULES, instruments=INSTRUMENTS, strategy=portfolio,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clock, src)


def summarise(result):
    daily = result.daily_pnl
    if not daily:
        return None
    perf = performance(daily, result.equity_curve,
                       starting_balance=RULES.starting_balance)
    rr30 = realized_pass_rate(daily, rules=RULES,
                              starting_balance=RULES.starting_balance,
                              window_days=30, stride_days=1)
    rr45 = realized_pass_rate(daily, rules=RULES,
                              starting_balance=RULES.starting_balance,
                              window_days=45, stride_days=1)
    return {
        "sharpe": perf.sharpe_annual,
        "max_dd_dollars": float(perf.max_drawdown_dollars),
        "best": float(perf.best_period),
        "worst": float(perf.worst_period),
        "pf": perf.profit_factor,
        "n_days": perf.n_periods,
        "pass30": rr30.pass_rate,
        "pass45": rr45.pass_rate,
        "outcomes30": rr30.outcomes,
        "outcomes45": rr45.outcomes,
        "daily_returns": [float(v) / float(RULES.starting_balance)
                          for _, v in sorted(daily.items())],
    }


def main():
    print("Loading data...")
    t0 = _time.time()
    bars, start, end = _load_aligned()
    print(f"  loaded in {_time.time()-t0:.1f}s; window {start.date()} -> {end.date()}")

    # Grid
    combos = []
    for or_min in (15, 30, 60):
        for tp in (1.0, 2.0, 3.0):
            for direction in ("long", "short", "both"):
                combos.append({"or_min": or_min, "tp": tp,
                               "stop_mode": "fixed_ticks", "stop_tk": 20,
                               "direction": direction})

    print(f"\nRunning {len(combos)} configs...\n")
    rows = []
    for i, p in enumerate(combos):
        t0 = _time.time()
        result = run_combo(bars, start, **p)
        s = summarise(result)
        s["params"] = p
        s["wall_s"] = _time.time() - t0
        rows.append(s)
        print(f"  [{i+1:>2}/{len(combos)}] or={p['or_min']:>2} tp={p['tp']:.1f} "
              f"dir={p['direction']:>5}: sharpe={s['sharpe']:>5.2f} "
              f"pass30={s['pass30']:>5.1%} pass45={s['pass45']:>5.1%} "
              f"MaxDD=${s['max_dd_dollars']:>7,.0f} pf={s['pf']:.2f} "
              f"({s['wall_s']:.1f}s)")

    print("\n=== Ranked by 30-day realized pass-rate ===")
    rows.sort(key=lambda r: r["pass30"], reverse=True)
    print(f"{'or':>3} {'tp':>4} {'dir':>5} {'pass30':>7} {'pass45':>7} "
          f"{'sharpe':>6} {'MaxDD':>9} {'best':>7} {'worst':>7} "
          f"{'pf':>5} {'outcomes30':}")
    for r in rows[:15]:
        p = r["params"]
        print(f"{p['or_min']:>3} {p['tp']:>4.1f} {p['direction']:>5} "
              f"{r['pass30']:>6.1%} {r['pass45']:>6.1%} "
              f"{r['sharpe']:>6.2f} ${r['max_dd_dollars']:>8,.0f} "
              f"${r['best']:>6,.0f} ${r['worst']:>6,.0f} "
              f"{r['pf']:>5.2f} {r['outcomes30']}")

    # PBO across the matrix
    try:
        T = min(len(r["daily_returns"]) for r in rows)
        if T >= 16:
            def runner(params_dict):
                # Look up the matching row
                for r in rows:
                    if r["params"] == params_dict:
                        return r["daily_returns"][:T]
                raise KeyError(str(params_dict))
            grid = {
                "or_min": sorted({c["or_min"] for c in combos}),
                "tp": sorted({c["tp"] for c in combos}),
                "stop_mode": sorted({c["stop_mode"] for c in combos}),
                "stop_tk": sorted({c["stop_tk"] for c in combos}),
                "direction": sorted({c["direction"] for c in combos}),
            }
            # Sensitivity_sweep iterates combos itself; build a sweep
            # over JUST the dimensions we varied to keep PBO meaningful.
            print(f"\n=== Sensitivity sweep + PBO ===")
            res = sensitivity_sweep(
                grid={k: v for k, v in grid.items() if len(v) > 1},
                runner=runner,
                pbo_slices=16,
            )
            print(f"  winner params: {res.winner.params}")
            print(f"  winner Sharpe: {res.winner.sharpe_annual:.2f}")
            print(f"  PSR of winner: {res.winner.psr:.3f}")
            print(f"  Deflated SR for winner: {res.dsr_for_winner:.3f}")
            print(f"  expected max SR under null: {res.expected_max_sharpe:.2f}")
            print(f"  PBO across grid: {res.pbo:.3f}")
            print(f"    (PBO > 0.5 = winner does not generalize OOS;"
                  f" PBO ~0.5 = no signal)")
    except Exception as e:
        print(f"\nPBO computation skipped: {e}")


if __name__ == "__main__":
    main()
