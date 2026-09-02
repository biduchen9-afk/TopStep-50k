"""Parameter sweep for MeanRev and OD on the timezone-corrected Databento
data, recent window. These two strategies have never been re-examined
this session -- their parameters are the original pre-databento values,
carried forward unchanged through every ORB experiment so far. Since
the timezone fix changed what real time-of-day both strategies actually
operate on (MeanRev's RTH trading window, OD's close/open offsets), it's
worth checking whether the original lookback/sigma_mult/stop_ticks/
time_stop (MeanRev) and entry/exit offsets (OD) are still sensible, or
just leftover guesses from before the fix.

IS-only screen (same two-phase discipline as the ORB sweeps): rank by
IS pass30 among configs that clear a light Gate-2 sanity filter
(EV>0, PF>1, trades/yr>=20, Sharpe>=0.3). Winners get manually reviewed
and, if promising, plugged into a v5 full-ensemble evaluation (one
touch of OOS) -- not run automatically here.

Grids:
  MeanRev : lookback [30,60,90] x sigma_mult [1.5,2.0,2.5] x
            stop_ticks [15,20,30] x time_stop_minutes [30,45]  (54/asset)
  OD      : entry_offset_minutes [0,5,10,15] x
            exit_offset_minutes [0,5,10,15]                    (16/asset)
Both gated by their existing causal, zero-free-parameter regime gates
(meanrev_low_vol_gate / overnight_drift_post_selloff_gate).
"""

from __future__ import annotations

import gc
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
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
from topstep50k.evaluation.harness import is_oos_split
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import (
    meanrev_low_vol_gate,
    overnight_drift_post_selloff_gate,
    per_day_session_stats,
)
from topstep50k.rules import combine_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.overnight_drift import OvernightDrift


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

MR_GRID = {
    "lookback": [30, 60, 90],
    "sigma_mult": [1.5, 2.0, 2.5],
    "stop_ticks": [15, 20, 30],
    "time_stop_minutes": [30, 45],
}
OD_GRID = {
    "entry_offset_minutes": [0, 5, 10, 15],
    "exit_offset_minutes": [0, 5, 10, 15],
}
N_FINALISTS = 5


def load_recent_bars(data_path) -> list:
    return [b for b in load_bars_csv(data_path) if b.ts >= RECENT_START]


def is_cutoff_from_bars(bars) -> date:
    days = sorted({b.ts.astimezone(EASTERN).date() for b in bars})
    _, _, is_days, _ = is_oos_split(days, {})
    return is_days[-1]


def run_mr_cell(bars, asset_key, tick_size_f, instrument, gate, params):
    strat = MeanReversionBollinger(symbol=asset_key, qty=1, tick_size=tick_size_f,
                                    daily_filter=gate, **params)
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: instrument}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, src)


def run_od_cell(bars, asset_key, instrument, gate, params):
    strat = OvernightDrift(symbol=asset_key, qty=1, entry_filter=gate, **params)
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: instrument}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, src)


def _row_from_result(result, params):
    daily = result.daily_pnl
    if len(daily) < 30:
        return None
    perf = performance(daily, result.equity_curve, starting_balance=RULES.starting_balance)
    rr30 = realized_pass_rate(daily, rules=RULES, starting_balance=RULES.starting_balance,
                               window_days=30, stride_days=1)
    pass45 = 0.0
    if len(daily) >= 45:
        rr45 = realized_pass_rate(daily, rules=RULES, starting_balance=RULES.starting_balance,
                                   window_days=45, stride_days=1)
        pass45 = rr45.pass_rate
    arr = np.array([float(v) for _, v in sorted(daily.items())])
    trades_per_yr = (arr != 0).sum() / (len(daily) / 252.0)
    return {
        "params": params, "sharpe": perf.sharpe_annual, "pf": perf.profit_factor,
        "ev_day": float(perf.total_pnl) / len(daily), "trades_per_yr": trades_per_yr,
        "win_rate": perf.win_rate, "pass30": rr30.pass_rate, "pass45": pass45,
        "is_returns": arr,  # kept for DSR on the eventual #1 finalist -- small (IS-only)
    }


def sweep_asset(asset_key: str) -> dict:
    cfg = ASSETS[asset_key]
    t0 = _time.time()
    bars = load_recent_bars(cfg["data_path"])
    load_s = _time.time() - t0
    n_bars_full = len(bars)

    stats = per_day_session_stats(bars)
    mr_gate = meanrev_low_vol_gate(stats)
    od_gate = overnight_drift_post_selloff_gate(stats)

    cutoff = is_cutoff_from_bars(bars)
    cutoff_dt = datetime.combine(cutoff, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=3)
    bars_is = [b for b in bars if b.ts <= cutoff_dt]
    del bars
    gc.collect()

    mr_combos = [dict(zip(MR_GRID.keys(), v, strict=True)) for v in product(*MR_GRID.values())]
    mr_rows = []
    for params in mr_combos:
        result = run_mr_cell(bars_is, asset_key, cfg["tick_size_f"], cfg["instrument"], mr_gate, params)
        row = _row_from_result(result, params)
        if row:
            mr_rows.append(row)
    mr_rows.sort(key=lambda r: r["pass30"], reverse=True)
    mr_ok = [r for r in mr_rows if r["ev_day"] > 0 and r["pf"] > 1.0
             and r["trades_per_yr"] >= 20 and r["sharpe"] >= 0.3]
    mr_ok.sort(key=lambda r: r["pass30"], reverse=True)

    od_combos = [dict(zip(OD_GRID.keys(), v, strict=True)) for v in product(*OD_GRID.values())]
    od_rows = []
    for params in od_combos:
        result = run_od_cell(bars_is, asset_key, cfg["instrument"], od_gate, params)
        row = _row_from_result(result, params)
        if row:
            od_rows.append(row)
    od_rows.sort(key=lambda r: r["pass30"], reverse=True)
    od_ok = [r for r in od_rows if r["ev_day"] > 0 and r["pf"] > 1.0
             and r["trades_per_yr"] >= 20 and r["sharpe"] >= 0.3]
    od_ok.sort(key=lambda r: r["pass30"], reverse=True)

    mr_finalists = (mr_ok or mr_rows)[:N_FINALISTS]
    od_finalists = (od_ok or od_rows)[:N_FINALISTS]

    # DSR on the #1 finalist: how much should its IS Sharpe be deflated
    # for having been picked as the best of len(mr_rows)/len(od_rows)
    # grid combos actually searched for this asset? (Bailey et al.)
    mr_dsr = od_dsr = None
    if mr_finalists:
        mr_dsr = deflated_sharpe(mr_finalists[0]["is_returns"].tolist(),
                                  all_trial_sharpes_annual=[r["sharpe"] for r in mr_rows])
    if od_finalists:
        od_dsr = deflated_sharpe(od_finalists[0]["is_returns"].tolist(),
                                  all_trial_sharpes_annual=[r["sharpe"] for r in od_rows])

    return {
        "asset": asset_key, "load_s": load_s, "n_bars": n_bars_full,
        "n_bars_is": len(bars_is), "is_cutoff": cutoff,
        "mr_rows": mr_rows, "mr_finalists": mr_finalists, "mr_dsr": mr_dsr,
        "od_rows": od_rows, "od_finalists": od_finalists, "od_dsr": od_dsr,
        "sweep_s": _time.time() - t0 - load_s,
    }


def print_top(title, rows, keys, n=8):
    print(f"\n{title}")
    for r in rows[:n]:
        p = r["params"]
        pstr = " ".join(f"{k}={p[k]}" for k in keys)
        print(f"  {pstr:<48} pass30={r['pass30']:>6.1%} pass45={r['pass45']:>6.1%} "
              f"sharpe={r['sharpe']:>+6.2f} pf={r['pf']:>5.2f} win%={r['win_rate']*100:>4.0f}% "
              f"ev/day=${r['ev_day']:>+6,.0f} trades/yr={r['trades_per_yr']:>5.0f}")


def main():
    print("=" * 78)
    print("MEANREV + OD PARAMETER SWEEP -- DATABENTO (TZ-CORRECTED), IS-ONLY SCREEN")
    print("=" * 78)
    print(f"\nMeanRev grid: {[len(v) for v in MR_GRID.values()]} = "
          f"{np.prod([len(v) for v in MR_GRID.values()])} combos/asset")
    print(f"OD grid: {[len(v) for v in OD_GRID.values()]} = "
          f"{np.prod([len(v) for v in OD_GRID.values()])} combos/asset\n")

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
                  f"swept {len(s['mr_rows'])} MR + {len(s['od_rows'])} OD cells "
                  f"in {s['sweep_s']:.0f}s", flush=True)

    print(f"\nPhase 1 wall time: {_time.time()-t0:.0f}s\n")

    print("DSR (on the #1 finalist only) = P(true IS Sharpe > 0) after deflating\n"
          "for how many grid combos were searched for this asset/strategy\n"
          "(Bailey et al. Deflated Sharpe Ratio). Low DSR despite a good raw\n"
          "Sharpe means 'best of N' noise, not necessarily real edge.\n")
    for ak in ASSETS:
        s = screens[ak]
        print(f"{'='*78}\n{ak}\n{'='*78}")
        print_top("MeanRev -- top by IS pass30", s["mr_rows"],
                   ["lookback", "sigma_mult", "stop_ticks", "time_stop_minutes"])
        if s["mr_dsr"] is not None:
            d = s["mr_dsr"]
            print(f"  -> #1 finalist DSR={d.dsr:.1%}  (n_trials={d.n_trials}, "
                  f"IS Sharpe={d.sharpe:+.2f}, E[max Sharpe|noise]={d.expected_max_sharpe:+.2f})")
        print_top("OD -- top by IS pass30", s["od_rows"],
                   ["entry_offset_minutes", "exit_offset_minutes"])
        if s["od_dsr"] is not None:
            d = s["od_dsr"]
            print(f"  -> #1 finalist DSR={d.dsr:.1%}  (n_trials={d.n_trials}, "
                  f"IS Sharpe={d.sharpe:+.2f}, E[max Sharpe|noise]={d.expected_max_sharpe:+.2f})")

    print(f"\n{'='*78}\nCURRENT (pre-committed) PARAMS FOR COMPARISON\n{'='*78}")
    print("  MeanRev: lookback=60 sigma_mult=2.0 stop_ticks=15 time_stop_minutes=45")
    print("  OD:      entry_offset_minutes=5 exit_offset_minutes=5")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
