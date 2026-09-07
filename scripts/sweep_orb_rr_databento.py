"""Risk:reward-focused ORB sweep on the (now timezone-corrected) Databento
dataset, recent window (>= 2021-12-31, matching the original promoted
run's era).

Motivation: every ORB sweep run so far used tp_multiple (TP relative to
the opening-range width). This sweep instead controls risk:reward
DIRECTLY via independent fixed-tick stop/TP (stop_mode='fixed_ticks',
tp_ticks), covering everything from a tight-TP/wide-stop high-win-rate
style (inspired by a high-win-rate/low-R:R social-media ORB screenshot)
through to wide-TP/tight-stop. Vol-expansion-gated (rv5/rv20 >= 0.8),
since that's the zero-free-parameter refinement already established to
matter for ORB.

For every finalist that reaches OOS_PROMOTED (or the closest misses),
also reports pass30/45/60/90 -- testing whether a strategy that clears
the 30-day bar keeps clearing it (or clears it BETTER) on longer windows,
as would be expected from a strategy with real, consistent edge and
bounded per-day risk.

Two-phase structure + memory-safety practices per sweep_orb_databento.py.
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

from topstep50k.analysis.dsr import deflated_sharpe
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
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)

GRID = {
    "or_min":    [15, 30, 60],
    "stop_ticks": [15, 25, 40],
    "tp_ticks":  [8, 10, 15, 20, 30, 40],
}
DIRECTION = "both"
N_FINALISTS = 4


def run_cell(bars, asset_key, tick_size_f, instrument, gate, *, or_min, stop_ticks, tp_ticks,
             direction=DIRECTION):
    strat = OpeningRangeBreakout(
        symbol=asset_key, qty=1, or_minutes=or_min, direction=direction,
        stop_mode="fixed_ticks", stop_ticks=stop_ticks, tp_ticks=tp_ticks,
        tp_multiple=None, flat_before_close_minutes=15, tick_size=tick_size_f,
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


def load_recent_bars(data_path) -> list:
    return [b for b in load_bars_csv(data_path) if b.ts >= RECENT_START]


def is_cutoff_from_bars(bars) -> date:
    days = sorted({b.ts.astimezone(EASTERN).date() for b in bars})
    _, _, is_days, _ = is_oos_split(days, {})
    return is_days[-1]


def sweep_asset(asset_key: str) -> dict:
    cfg = ASSETS[asset_key]
    t0 = _time.time()
    bars = load_recent_bars(cfg["data_path"])
    load_s = _time.time() - t0
    n_bars_full = len(bars)

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
                           or_min=params["or_min"], stop_ticks=params["stop_ticks"],
                           tp_ticks=params["tp_ticks"])
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
            "trades_per_yr": trades_per_yr, "win_rate": perf.win_rate,
            "pass30": rr30.pass_rate, "pass45": rr45.pass_rate,
        })

    rows.sort(key=lambda r: r["pass30"], reverse=True)
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


def phase2_verdict(asset_key: str, params: dict, all_trial_sharpes: list[float]) -> tuple:
    cfg = ASSETS[asset_key]
    bars = load_recent_bars(cfg["data_path"])
    stats = per_day_session_stats(bars)
    gate = orb_expansion_gate(stats)
    result = run_cell(bars, asset_key, cfg["tick_size_f"], cfg["instrument"], gate,
                       or_min=params["or_min"], stop_ticks=params["stop_ticks"],
                       tp_ticks=params["tp_ticks"])
    days = sorted(result.daily_pnl.keys())
    is_mask, oos_mask, is_days, oos_days = is_oos_split(days, {})
    daily_full = _to_series(result.daily_pnl, days)
    label = (f"{asset_key}/ORB(gated,RR) or={params['or_min']} "
             f"stop={params['stop_ticks']}t tp={params['tp_ticks']}t")
    verdict = evaluate_strategy(label, daily_full, is_mask, oos_mask, is_days, oos_days, RULES)
    dsr = deflated_sharpe(daily_full[is_mask].tolist(),
                           all_trial_sharpes_annual=all_trial_sharpes)

    # Longer-window pass-rate scaling test on the FULL daily series.
    windows = {}
    for w in (30, 45, 60, 90):
        if len(days) >= w:
            rr = realized_pass_rate(result.daily_pnl, rules=RULES,
                                     starting_balance=RULES.starting_balance,
                                     window_days=w, stride_days=1)
            windows[w] = rr.pass_rate
    return verdict, windows, dsr


def main():
    print("=" * 78)
    print("ORB RISK:REWARD SWEEP -- DATABENTO (TZ-CORRECTED), VOL-EXPANSION GATED")
    print("Fixed stop_ticks x fixed tp_ticks (independent R:R control)")
    print("=" * 78)
    print(f"\nGrid per asset: {[len(v) for v in GRID.values()]} = "
          f"{np.prod([len(v) for v in GRID.values()])} combos, direction={DIRECTION}\n")

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
        print(f"{'='*78}\n{ak} -- top 12 by IS pass30 (R:R screen, gated)\n{'='*78}")
        print(f"{'or':>3} {'stop':>5} {'tp':>4} {'R:R':>6} {'pass30':>7} {'pass45':>7} "
              f"{'sharpe':>7} {'pf':>5} {'win%':>5} {'ev/day':>8} {'trades/yr':>9}")
        for r in s["rows"][:12]:
            p = r["params"]
            rr_ratio = p["tp_ticks"] / p["stop_ticks"]
            print(f"{p['or_min']:>3} {p['stop_ticks']:>5} {p['tp_ticks']:>4} {rr_ratio:>5.2f}x "
                  f"{r['pass30']:>6.1%} {r['pass45']:>6.1%} {r['sharpe']:>+7.2f} "
                  f"{r['pf']:>5.2f} {r['win_rate']*100:>4.0f}% ${r['ev_day']:>+7,.0f} "
                  f"{r['trades_per_yr']:>9.0f}")
        print()

    print(f"{'='*78}\nPHASE 2 -- FULL 6-GATE CHECKLIST + WINDOW-SCALING TEST\n{'='*78}\n")
    print("DSR = P(true IS Sharpe > 0) after deflating for how many grid combos\n"
          "were searched for this asset (Bailey et al. Deflated Sharpe Ratio).\n")
    verdicts = []
    for ak in ASSETS:
        all_sharpes = [r["sharpe"] for r in screens[ak]["rows"]]
        for r in screens[ak]["finalists"]:
            t1 = _time.time()
            result, windows, dsr = phase2_verdict(ak, r["params"], all_sharpes)
            win_s = "  ".join(f"pass{w}={p:.1%}" for w, p in windows.items())
            print(f"  [{_time.time()-t1:.0f}s] {result.label:<48} -> "
                  f"{result.promoted.upper():<14} IS-pass30={result.is_pass30:.1%} "
                  f"OOS-pass30={result.oos_pass30:.1%}", flush=True)
            print(f"           full-series window scaling: {win_s}", flush=True)
            print(f"           DSR={dsr.dsr:.1%}  (n_trials={dsr.n_trials}, "
                  f"E[max Sharpe|noise]={dsr.expected_max_sharpe:+.2f})", flush=True)
            verdicts.append((ak, r["params"], result, windows, dsr))

    print(f"\n{'='*78}\nSUMMARY -- SURVIVORS (OOS_PROMOTED)\n{'='*78}")
    survivors = [(ak, p, res, w, dsr) for ak, p, res, w, dsr in verdicts if res.promoted == "oos_promoted"]
    if not survivors:
        print("  NONE of the finalists reached OOS_PROMOTED.")
        ranked = sorted(verdicts, key=lambda x: (x[2].promoted, x[2].oos_pass30), reverse=True)
        for ak, p, res, w, dsr in ranked[:10]:
            win_s = "  ".join(f"pass{k}={v:.1%}" for k, v in w.items())
            print(f"  {res.label:<48} {res.promoted:<12} IS-pass30={res.is_pass30:.1%} "
                  f"OOS-pass30={res.oos_pass30:.1%} OOS-Sharpe={res.oos_sharpe:+.2f} "
                  f"DSR={dsr.dsr:.1%}")
            print(f"    {win_s}")
    else:
        survivors.sort(key=lambda x: x[2].oos_pass30, reverse=True)
        for ak, p, res, w, dsr in survivors:
            win_s = "  ".join(f"pass{k}={v:.1%}" for k, v in w.items())
            print(f"  {res.label}")
            print(f"    IS-pass30={res.is_pass30:.1%}  OOS-pass30={res.oos_pass30:.1%}  "
                  f"OOS-Sharpe={res.oos_sharpe:+.2f}  DSR={dsr.dsr:.1%} "
                  f"(n_trials={dsr.n_trials})")
            print(f"    {win_s}")
            res.print_report()

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
