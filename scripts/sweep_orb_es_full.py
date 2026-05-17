"""Full ORB parameter sweep on single-asset ES (the highest pass-rate
baseline so far). Includes DSR-deflated winner and matrix PBO.

Grid (3 x 3 x 2 x 2 x 4 = 144 cells):

  or_minutes : 15, 30, 60
  tp_multiple : 1.0, 2.0, 3.0
  stop_mode : opposite_range OR fixed_ticks-20
  direction : long, both
  chop filter (min_or_width_vs_median) : 0.0, 0.5, 1.0, 1.5

Each cell runs the full 4-year ES backtest (combine_enforcement=False)
and reports 30/45-day realised pass-rates, Sharpe, MaxDD, and the
outcome distribution. The aggregator then ranks and computes:

  * DSR for the winner deflated for the trial population size
  * PBO across the whole matrix via combinatorially-symmetric CV
"""

from __future__ import annotations

import sys
import time as _time
from dataclasses import replace
from decimal import Decimal
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.dsr import (
    deflated_sharpe,
    probability_of_backtest_overfit,
)
from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.analysis.stats import performance
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.orb import OpeningRangeBreakout


ES = Instrument(symbol="ES", point_value=Decimal("50"),
                tick_size=Decimal("0.25"),
                commission_per_side=Decimal("2.50"))
RULES = combine_50k()


def run_cell(bars, *, or_min, tp, stop_mode, direction, chop):
    strat = OpeningRangeBreakout(
        symbol="ES", qty=1,
        or_minutes=or_min, direction=direction,
        stop_mode=stop_mode,
        stop_ticks=20,
        tp_multiple=tp,
        flat_before_close_minutes=15,
        tick_size=0.25,
        min_or_width_vs_median=chop,
    )
    pf = PortfolioStrategy(components={"ES": strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({"ES": bars}, clk)
    bt = Backtester(rules=RULES, instruments={"ES": ES}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, src)


def main():
    print("Loading ES bars...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(ROOT / "data" / "raw" / "es_cleaned.txt"))
    print(f"  {len(bars):,} bars in {_time.time()-t0:.1f}s", flush=True)

    grid = {
        "or_min":     [15, 30, 60],
        "tp":         [1.0, 2.0, 3.0],
        "stop_mode":  ["opposite_range", "fixed_ticks"],
        "direction":  ["long", "both"],
        "chop":       [0.0, 0.5, 1.0, 1.5],
    }
    combos = [dict(zip(grid.keys(), v, strict=True))
              for v in product(*grid.values())]
    print(f"\nRunning {len(combos)} configs...\n", flush=True)

    rows = []
    for i, params in enumerate(combos):
        t = _time.time()
        result = run_cell(bars, **params)
        daily = result.daily_pnl
        if not daily:
            print(f"  [{i+1:>3}/{len(combos)}] {params}: NO TRADES", flush=True)
            continue
        perf = performance(daily, result.equity_curve,
                           starting_balance=RULES.starting_balance)
        rr30 = realized_pass_rate(daily, rules=RULES,
                                  starting_balance=RULES.starting_balance,
                                  window_days=30, stride_days=1)
        rr45 = realized_pass_rate(daily, rules=RULES,
                                  starting_balance=RULES.starting_balance,
                                  window_days=45, stride_days=1)
        rows.append({
            "params": params,
            "sharpe": perf.sharpe_annual,
            "max_dd": float(perf.max_drawdown_dollars),
            "best": float(perf.best_period),
            "worst": float(perf.worst_period),
            "pf": perf.profit_factor,
            "pass30": rr30.pass_rate,
            "pass45": rr45.pass_rate,
            "outcomes30": rr30.outcomes,
            "daily_returns": [float(v) / float(RULES.starting_balance)
                              for _, v in sorted(daily.items())],
        })
        print(f"  [{i+1:>3}/{len(combos)}] or={params['or_min']:>2} "
              f"tp={params['tp']:.1f} stop={params['stop_mode']:>13} "
              f"dir={params['direction']:>4} chop={params['chop']:.1f}: "
              f"pass30={rr30.pass_rate:>5.1%} pass45={rr45.pass_rate:>5.1%} "
              f"sharpe={perf.sharpe_annual:>+5.2f} "
              f"MaxDD=${perf.max_drawdown_dollars:>7} "
              f"({_time.time()-t:.0f}s)", flush=True)

    rows.sort(key=lambda r: r["pass30"], reverse=True)

    print(f"\n=== Top 15 by 30-day pass-rate ===", flush=True)
    print(f"{'or':>3} {'tp':>4} {'stop':>13} {'dir':>5} {'chop':>5} "
          f"{'pass30':>7} {'pass45':>7} {'sharpe':>7} {'MaxDD':>9} "
          f"{'best':>7} {'worst':>7} {'pf':>5}", flush=True)
    for r in rows[:15]:
        p = r["params"]
        print(f"{p['or_min']:>3} {p['tp']:>4.1f} {p['stop_mode']:>13} "
              f"{p['direction']:>5} {p['chop']:>5.1f} "
              f"{r['pass30']:>6.1%} {r['pass45']:>6.1%} "
              f"{r['sharpe']:>+7.2f} ${r['max_dd']:>8,.0f} "
              f"${r['best']:>6,.0f} ${r['worst']:>6,.0f} "
              f"{r['pf']:>5.2f}", flush=True)

    # DSR + PBO using the actual sweep results.
    if len(rows) >= 4:
        T = min(len(r["daily_returns"]) for r in rows)
        all_sharpes = [r["sharpe"] for r in rows]
        winner = max(rows, key=lambda r: r["sharpe"])
        try:
            dsr = deflated_sharpe(
                winner["daily_returns"],
                all_trial_sharpes_annual=all_sharpes,
                periods_per_year=252,
            )
            print(f"\n=== Deflated Sharpe (winner = best-sharpe cell) ===", flush=True)
            print(f"  winner params: {winner['params']}", flush=True)
            print(f"  raw Sharpe: {dsr.sharpe:.3f}", flush=True)
            print(f"  expected-max Sharpe under null: {dsr.expected_max_sharpe:.3f}", flush=True)
            print(f"  trial-sharpe stdev: {dsr.sharpe_trial_std:.3f}", flush=True)
            print(f"  DSR (prob true SR > deflated bench): {dsr.dsr:.4f}", flush=True)
            if dsr.dsr < 0.95:
                print(f"  -> NOT statistically distinguishable from selection bias.", flush=True)
        except Exception as e:
            print(f"  DSR skipped: {e}", flush=True)

        try:
            mat = np.array([r["daily_returns"][:T] for r in rows]).T  # (T, N)
            pbo = probability_of_backtest_overfit(mat, s=16,
                                                  rng=np.random.default_rng(0))
            print(f"\n=== PBO across the grid ===", flush=True)
            print(f"  N strategies: {len(rows)}, T periods: {T}", flush=True)
            print(f"  PBO: {pbo:.3f}", flush=True)
            print(f"  (PBO > 0.5 = in-sample winner DOES NOT generalise OOS;",
                  flush=True)
            print(f"   PBO ~ 0.5 = noise; PBO < 0.2 = genuine edge)", flush=True)
        except Exception as e:
            print(f"  PBO skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
