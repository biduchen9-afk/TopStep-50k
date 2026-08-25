"""ORB parameter sweep on the Databento dataset, WITH the volatility-
expansion regime gate (orb_expansion_gate: rv5_prior/rv20_prior >= 0.8)
that made ORB viable in the original 9-stream ensemble.

sweep_orb_databento.py (raw/ungated ORB) found that NO combination of
or_minutes/tp_multiple/stop_mode/chop survives Gate 2 (sharpe_annual >=
0.3) on the full 2010-2026 multi-regime history -- best Sharpe found was
+0.06. This script tests whether gating entries to volatility-expansion
days (the same causal, zero-free-parameter rule used in the promoted
9-stream ensemble) rescues it on the longer history, and if so, which
or_minutes/tp_multiple/direction combo is best.

Same two-phase structure and same OOM-safety fixes as sweep_orb_databento.py.
"""

from __future__ import annotations

import gc
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, time as dtime, timedelta, timezone
from decimal import Decimal
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

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
EASTERN = ZoneInfo("America/New_York")

GRID = {
    "or_min":    [15, 30, 60],
    "tp":        [1.0, 1.5, 2.0],
    "direction": ["long", "both"],
}
N_FINALISTS = 3


def run_cell(bars, asset_key, tick_size_f, instrument, gate, *, or_min, tp, direction):
    strat = OpeningRangeBreakout(
        symbol=asset_key, qty=1, or_minutes=or_min, direction=direction,
        stop_mode="opposite_range", tp_multiple=tp,
        flat_before_close_minutes=15, tick_size=tick_size_f,
        daily_filter=gate,
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


def is_cutoff_from_bars(bars) -> date:
    days = sorted({b.ts.astimezone(EASTERN).date() for b in bars})
    _, _, is_days, _ = is_oos_split(days, {})
    return is_days[-1]


def sweep_asset(asset_key: str) -> dict:
    cfg = ASSETS[asset_key]
    t0 = _time.time()
    bars = list(load_bars_csv(cfg["data_path"]))
    load_s = _time.time() - t0
    n_bars_full = len(bars)

    # Gate is built from FULL history (causal/backward-looking only, so no
    # leakage into the IS-only screen below) -- cheap, per-day not per-bar.
    stats = per_day_session_stats(bars)
    gate = orb_expansion_gate(stats)

    cutoff = is_cutoff_from_bars(bars)
    cutoff_dt = datetime.combine(cutoff, dtime.min, tzinfo=timezone.utc) + timedelta(days=3)
    bars_is = [b for b in bars if b.ts <= cutoff_dt]
    del bars
    gc.collect()

    combos = [dict(zip(GRID.keys(), v, strict=True)) for v in product(*GRID.values())]
    rows = []
    for params in combos:
        result = run_cell(bars_is, asset_key, cfg["tick_size_f"], cfg["instrument"], gate,
                           or_min=params["or_min"], tp=params["tp"],
                           direction=params["direction"])
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
    # Full Gate-2 pre-filter this time (avoid wasting phase-2 runs on
    # doomed configs like the ungated sweep did).
    gate2_ok = [r for r in rows if r["ev_day"] > 0 and r["pf"] > 1.0
                and r["trades_per_yr"] >= 20 and r["sharpe"] >= 0.3]
    gate2_ok.sort(key=lambda r: r["pass30"], reverse=True)
    finalists = gate2_ok[:N_FINALISTS] if gate2_ok else rows[:N_FINALISTS]

    return {
        "asset": asset_key, "load_s": load_s, "n_bars": n_bars_full,
        "n_bars_is": len(bars_is), "is_cutoff": cutoff,
        "rows": rows, "finalists": finalists, "any_gate2_ok": bool(gate2_ok),
        "sweep_s": _time.time() - t0 - load_s,
    }


def phase2_verdict(asset_key: str, params: dict) -> object:
    cfg = ASSETS[asset_key]
    bars = list(load_bars_csv(cfg["data_path"]))
    stats = per_day_session_stats(bars)
    gate = orb_expansion_gate(stats)
    result = run_cell(bars, asset_key, cfg["tick_size_f"], cfg["instrument"], gate,
                       or_min=params["or_min"], tp=params["tp"],
                       direction=params["direction"])
    days = sorted(result.daily_pnl.keys())
    is_mask, oos_mask, is_days, oos_days = is_oos_split(days, {})
    daily_full = _to_series(result.daily_pnl, days)
    label = (f"{asset_key}/ORB(gated) or={params['or_min']} tp={params['tp']:.1f} "
             f"dir={params['direction']}")
    return evaluate_strategy(label, daily_full, is_mask, oos_mask, is_days, oos_days, RULES)


def main():
    print("=" * 78)
    print("ORB SWEEP -- DATABENTO DATASET, VOL-EXPANSION GATED (rv5/rv20 >= 0.8)")
    print("Phase 1: IS-only grid screen | Phase 2: full 6-gate OOS checklist")
    print("=" * 78)
    print(f"\nGrid per asset: {[len(v) for v in GRID.values()]} = "
          f"{np.prod([len(v) for v in GRID.values()])} combos, stop=opposite_range\n")

    t0 = _time.time()
    screens: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(sweep_asset, ak): ak for ak in ASSETS}
        for fut in as_completed(futs):
            ak = futs[fut]
            screens[ak] = fut.result()
            s = screens[ak]
            print(f"[{ak}] loaded {s['n_bars']:,} bars ({s['load_s']:.0f}s), "
                  f"IS cutoff={s['is_cutoff']}, IS bars={s['n_bars_is']:,}, "
                  f"swept {len(s['rows'])} cells in {s['sweep_s']:.0f}s, "
                  f"any_gate2_ok={s['any_gate2_ok']}", flush=True)

    print(f"\nPhase 1 wall time: {_time.time()-t0:.0f}s\n")

    for ak in ASSETS:
        s = screens[ak]
        print(f"{'='*78}\n{ak} -- top 10 by IS pass30 (IS-only screen, gated)\n{'='*78}")
        print(f"{'or':>3} {'tp':>4} {'dir':>5} {'pass30':>7} {'pass45':>7} "
              f"{'sharpe':>7} {'pf':>5} {'ev/day':>8} {'trades/yr':>9}")
        for r in s["rows"][:10]:
            p = r["params"]
            print(f"{p['or_min']:>3} {p['tp']:>4.1f} {p['direction']:>5} "
                  f"{r['pass30']:>6.1%} {r['pass45']:>6.1%} {r['sharpe']:>+7.2f} "
                  f"{r['pf']:>5.2f} ${r['ev_day']:>+7,.0f} {r['trades_per_yr']:>9.0f}")
        print()

    print(f"{'='*78}\nPHASE 2 -- FULL 6-GATE CHECKLIST (IS+OOS, ONE-TOUCH) ON FINALISTS\n{'='*78}\n")
    verdicts = []
    for ak in ASSETS:
        for r in screens[ak]["finalists"]:
            t1 = _time.time()
            result = phase2_verdict(ak, r["params"])
            print(f"  [{_time.time()-t1:.0f}s] {result.label:<50} -> "
                  f"{result.promoted.upper():<14} IS-pass30={result.is_pass30:.1%} "
                  f"OOS-pass30={result.oos_pass30:.1%} OOS-Sharpe={result.oos_sharpe:+.2f}",
                  flush=True)
            verdicts.append((ak, r["params"], result))

    print(f"\n{'='*78}\nSUMMARY -- SURVIVORS (OOS_PROMOTED)\n{'='*78}")
    survivors = [(ak, p, res) for ak, p, res in verdicts if res.promoted == "oos_promoted"]
    if not survivors:
        print("  NONE of the finalists reached OOS_PROMOTED.")
        ranked = sorted(verdicts, key=lambda x: (x[2].promoted, x[2].oos_pass30), reverse=True)
        for ak, p, res in ranked[:9]:
            print(f"  {res.label:<50} {res.promoted:<12} IS-pass30={res.is_pass30:.1%} "
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
