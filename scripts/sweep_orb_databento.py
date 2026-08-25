"""ORB parameter sweep on the new Databento dataset (2010-06 -> 2026-07,
~16 years, ES + NQ + GC). Same methodology as sweep_orb_es_full.py,
generalised to all 3 assets and to the 70/30 IS/OOS harness used by the
9-stream ensemble (evaluation/harness.py).

Phase 1 (screen): sweep a parameter grid on IS-only bars per asset, rank
by IS pass30 subject to a light Gate-2 sanity filter. Cheap and fast --
narrows ~100+ combos per asset down to a handful of finalists.

Phase 2 (verdict): run the full 6-gate promotion checklist
(harness.evaluate_strategy, IS+OOS, ONE-TOUCH) for the top finalists per
asset on the complete 16-year bar history.

Run with: python scripts/sweep_orb_databento.py
"""

from __future__ import annotations

import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, time as dtime, timedelta, timezone
from decimal import Decimal
from itertools import product
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

GRID = {
    "or_min":    [15, 30, 60],
    "tp":        [1.0, 1.5, 2.0],
    "stop_mode": ["opposite_range", "fixed_ticks"],
    "chop":      [0.0, 0.5],
}
DIRECTION = "both"   # fixed for phase-1 screen; prior ES sweep showed 'both' > 'long'
N_FINALISTS = 3       # top-N per asset promoted to phase-2 full checklist


def run_cell(bars, asset_key, tick_size_f, instrument, *, or_min, tp, stop_mode, chop,
             direction=DIRECTION):
    strat = OpeningRangeBreakout(
        symbol=asset_key, qty=1, or_minutes=or_min, direction=direction,
        stop_mode=stop_mode, stop_ticks=20, tp_multiple=tp,
        flat_before_close_minutes=15, tick_size=tick_size_f,
        min_or_width_vs_median=chop,
    )
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: instrument}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, src)


def _to_series(daily_pnl: dict, day_index: list[date]) -> np.ndarray:
    out = np.zeros(len(day_index), dtype=float)
    for i, d in enumerate(day_index):
        if d in daily_pnl:
            out[i] = float(daily_pnl[d])
    return out


def find_is_cutoff(bars, asset_key, tick_size_f, instrument) -> date:
    """One full-history baseline run just to locate the IS/OOS day cutoff."""
    result = run_cell(bars, asset_key, tick_size_f, instrument,
                       or_min=30, tp=1.0, stop_mode="opposite_range", chop=0.0)
    days = sorted(result.daily_pnl.keys())
    _, _, is_days, _ = is_oos_split(days, {})
    return is_days[-1]


def sweep_asset(asset_key: str) -> dict:
    """Runs entirely in a worker process: load bars, find IS cutoff, slice to
    IS-only bars, sweep the grid, return ranked rows + finalist configs."""
    cfg = ASSETS[asset_key]
    t0 = _time.time()
    bars = list(load_bars_csv(cfg["data_path"]))
    load_s = _time.time() - t0

    cutoff = find_is_cutoff(bars, asset_key, cfg["tick_size_f"], cfg["instrument"])
    # Buffer 3 calendar days past the Eastern IS cutoff to be safely inclusive
    # without leaking a meaningful amount of OOS into the phase-1 screen.
    cutoff_dt = (
        __import__("datetime").datetime.combine(cutoff, dtime.min, tzinfo=timezone.utc)
        + timedelta(days=3)
    )
    bars_is = [b for b in bars if b.ts <= cutoff_dt]

    combos = [dict(zip(GRID.keys(), v, strict=True)) for v in product(*GRID.values())]
    rows = []
    for params in combos:
        result = run_cell(bars_is, asset_key, cfg["tick_size_f"], cfg["instrument"],
                           or_min=params["or_min"], tp=params["tp"],
                           stop_mode=params["stop_mode"], chop=params["chop"])
        daily = result.daily_pnl
        if len(daily) < 30:
            continue
        perf = performance(daily, result.equity_curve, starting_balance=RULES.starting_balance)
        rr30 = realized_pass_rate(daily, rules=RULES, starting_balance=RULES.starting_balance,
                                   window_days=30, stride_days=1)
        rr45 = realized_pass_rate(daily, rules=RULES, starting_balance=RULES.starting_balance,
                                   window_days=45, stride_days=1)
        arr = np.array([float(v) for _, v in sorted(daily.items())])
        trades_per_yr = (arr != 0).sum() / (len(daily) / 252.0)
        rows.append({
            "asset": asset_key, "params": params,
            "sharpe": perf.sharpe_annual, "pf": perf.profit_factor,
            "max_dd": float(perf.max_drawdown_dollars),
            "ev_day": float(perf.total_pnl) / len(daily),
            "trades_per_yr": trades_per_yr,
            "pass30": rr30.pass_rate, "pass45": rr45.pass_rate,
        })

    rows.sort(key=lambda r: r["pass30"], reverse=True)
    gate2_ok = [r for r in rows if r["ev_day"] > 0 and r["pf"] > 1.0
                and r["trades_per_yr"] >= 20]
    gate2_ok.sort(key=lambda r: r["pass30"], reverse=True)
    finalists = gate2_ok[:N_FINALISTS] if gate2_ok else rows[:N_FINALISTS]

    return {
        "asset": asset_key, "load_s": load_s, "n_bars": len(bars),
        "n_bars_is": len(bars_is), "is_cutoff": cutoff,
        "rows": rows, "finalists": finalists,
        "sweep_s": _time.time() - t0 - load_s,
    }


def phase2_verdict(asset_key: str, params: dict) -> object:
    """Full 6-gate checklist (harness.evaluate_strategy) on complete
    16-year history, one-touch OOS."""
    cfg = ASSETS[asset_key]
    bars = list(load_bars_csv(cfg["data_path"]))
    result = run_cell(bars, asset_key, cfg["tick_size_f"], cfg["instrument"],
                       or_min=params["or_min"], tp=params["tp"],
                       stop_mode=params["stop_mode"], chop=params["chop"])
    days = sorted(result.daily_pnl.keys())
    is_mask, oos_mask, is_days, oos_days = is_oos_split(days, {})
    daily_full = _to_series(result.daily_pnl, days)
    label = (f"{asset_key}/ORB or={params['or_min']} tp={params['tp']:.1f} "
             f"stop={params['stop_mode']} chop={params['chop']:.1f}")
    return evaluate_strategy(label, daily_full, is_mask, oos_mask, is_days, oos_days, RULES)


def main():
    print("=" * 78)
    print("ORB SWEEP -- DATABENTO DATASET (2010-06 -> 2026-07, ES+NQ+GC)")
    print("Phase 1: IS-only grid screen | Phase 2: full 6-gate OOS checklist")
    print("=" * 78)
    print(f"\nGrid per asset: {[len(v) for v in GRID.values()]} = "
          f"{np.prod([len(v) for v in GRID.values()])} combos, direction={DIRECTION}\n")

    t0 = _time.time()
    screens: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(sweep_asset, ak): ak for ak in ASSETS}
        for fut in as_completed(futs):
            ak = futs[fut]
            screens[ak] = fut.result()
            s = screens[ak]
            print(f"[{ak}] loaded {s['n_bars']:,} bars ({s['load_s']:.0f}s), "
                  f"IS cutoff={s['is_cutoff']}, IS bars={s['n_bars_is']:,}, "
                  f"swept {len(s['rows'])} cells in {s['sweep_s']:.0f}s", flush=True)

    print(f"\nPhase 1 wall time: {_time.time()-t0:.0f}s\n")

    for ak in ASSETS:
        s = screens[ak]
        print(f"{'='*78}\n{ak} -- top 10 by IS pass30 (IS-only screen)\n{'='*78}")
        print(f"{'or':>3} {'tp':>4} {'stop':>13} {'chop':>5} {'pass30':>7} {'pass45':>7} "
              f"{'sharpe':>7} {'pf':>5} {'ev/day':>8} {'trades/yr':>9}")
        for r in s["rows"][:10]:
            p = r["params"]
            print(f"{p['or_min']:>3} {p['tp']:>4.1f} {p['stop_mode']:>13} {p['chop']:>5.1f} "
                  f"{r['pass30']:>6.1%} {r['pass45']:>6.1%} {r['sharpe']:>+7.2f} "
                  f"{r['pf']:>5.2f} ${r['ev_day']:>+7,.0f} {r['trades_per_yr']:>9.0f}")
        print()

    print(f"{'='*78}\nPHASE 2 -- FULL 6-GATE CHECKLIST (IS+OOS, ONE-TOUCH) ON FINALISTS\n{'='*78}\n")
    verdicts = []
    for ak in ASSETS:
        for r in screens[ak]["finalists"]:
            t1 = _time.time()
            result = phase2_verdict(ak, r["params"])
            print(f"  [{_time.time()-t1:.0f}s] {result.label:<55} -> "
                  f"{result.promoted.upper():<14} IS-pass30={result.is_pass30:.1%} "
                  f"OOS-pass30={result.oos_pass30:.1%}", flush=True)
            verdicts.append((ak, r["params"], result))

    print(f"\n{'='*78}\nSUMMARY -- SURVIVORS (OOS_PROMOTED)\n{'='*78}")
    survivors = [(ak, p, res) for ak, p, res in verdicts if res.promoted == "oos_promoted"]
    if not survivors:
        print("  NONE of the finalists reached OOS_PROMOTED.")
        print("  Best IS-only results (is_only or better), ranked by OOS pass30:")
        ranked = sorted(verdicts, key=lambda x: x[2].oos_pass30, reverse=True)
        for ak, p, res in ranked[:5]:
            print(f"  {res.label:<55} {res.promoted:<12} IS-pass30={res.is_pass30:.1%} "
                  f"OOS-pass30={res.oos_pass30:.1%} OOS-Sharpe={res.oos_sharpe:+.2f}")
    else:
        survivors.sort(key=lambda x: x[2].oos_pass30, reverse=True)
        for ak, p, res in survivors:
            print(f"  {res.label}")
            print(f"    IS-pass30={res.is_pass30:.1%}  OOS-pass30={res.oos_pass30:.1%}  "
                  f"OOS-Sharpe={res.oos_sharpe:+.2f}")
            res.print_report()

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
