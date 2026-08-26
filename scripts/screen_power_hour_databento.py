"""IS-only screen of PowerHourContinuation (new, time-disjoint from
ORB/MeanRev by construction) across ES/NQ/GC, ungated -- same
discipline used for every other new-candidate screen this session
(raw first, gate only if there's a real, causal reason to add one).

Run with: python scripts/screen_power_hour_databento.py
"""

from __future__ import annotations

import sys
import time as _time
from datetime import datetime, time as dtime, timedelta, timezone
from decimal import Decimal
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
from topstep50k.evaluation.harness import is_oos_split
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.power_hour import PowerHourContinuation


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
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)
EASTERN = ZoneInfo("America/New_York")


def load_recent_bars(data_path) -> list:
    return [b for b in load_bars_csv(data_path) if b.ts >= RECENT_START]


def run_cell(bars, asset_key, tick_size_f, instrument):
    strat = PowerHourContinuation(symbol=asset_key, tick_size=tick_size_f)
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: instrument}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, src)


def row_from_result(result):
    daily = result.daily_pnl
    if len(daily) < 30:
        return None
    perf = performance(daily, result.equity_curve, starting_balance=RULES.starting_balance)
    rr30 = realized_pass_rate(daily, rules=RULES, starting_balance=RULES.starting_balance,
                               window_days=30, stride_days=1)
    arr = np.array([float(v) for _, v in sorted(daily.items())])
    trades_per_yr = (arr != 0).sum() / (len(daily) / 252.0)
    return {
        "sharpe": perf.sharpe_annual, "pf": perf.profit_factor,
        "ev_day": float(perf.total_pnl) / len(daily), "trades_per_yr": trades_per_yr,
        "pass30": rr30.pass_rate, "nz": int((arr != 0).sum()), "total": float(perf.total_pnl),
        "win_rate": perf.win_rate,
    }


def main():
    print("=" * 78)
    print("SCREEN: PowerHourContinuation -- IS-only, ES/NQ/GC, ungated")
    print("=" * 78)

    t0 = _time.time()
    rows = {}
    for ak in ASSETS:
        print(f"\nLoading {ak} bars...", flush=True)
        tl = _time.time()
        bars = load_recent_bars(ASSETS[ak]["data_path"])
        print(f"  {len(bars):,} bars in {_time.time()-tl:.1f}s", flush=True)

        days = sorted({b.ts.astimezone(EASTERN).date() for b in bars})
        _, _, is_days, _ = is_oos_split(days, {})
        cutoff = is_days[-1]
        cutoff_dt = datetime.combine(cutoff, dtime.min, tzinfo=timezone.utc) + timedelta(days=3)
        bars_is = [b for b in bars if b.ts <= cutoff_dt]
        del bars
        print(f"  IS cutoff={cutoff}, IS bars={len(bars_is):,}", flush=True)

        t1 = _time.time()
        result = run_cell(bars_is, ak, ASSETS[ak]["tick_size_f"], ASSETS[ak]["instrument"])
        row = row_from_result(result)
        el = _time.time() - t1
        if row is None:
            print(f"  PowerHour: too few active days ({el:.0f}s)")
            del bars_is
            continue
        rows[ak] = row
        gate2_ok = (row["ev_day"] > 0 and row["pf"] > 1.0
                    and row["trades_per_yr"] >= 20 and row["sharpe"] >= 0.3)
        flag = "" if gate2_ok else "  [fails light Gate2]"
        print(f"  PowerHour: nz={row['nz']:>4} total=${row['total']:>+9,.0f} "
              f"Sharpe={row['sharpe']:>+6.2f} PF={row['pf']:>5.2f} win%={row['win_rate']*100:>4.0f}% "
              f"pass30={row['pass30']:>6.1%} trades/yr={row['trades_per_yr']:>5.0f} "
              f"({el:.0f}s){flag}", flush=True)
        del bars_is

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for ak, row in rows.items():
        gate2_ok = (row["ev_day"] > 0 and row["pf"] > 1.0
                    and row["trades_per_yr"] >= 20 and row["sharpe"] >= 0.3)
        print(f"  {ak}/PowerHour: pass30={row['pass30']:>6.1%} Sharpe={row['sharpe']:>+6.2f} "
              f"PF={row['pf']:>5.2f} trades/yr={row['trades_per_yr']:>5.0f}  "
              f"{'GATE2-OK' if gate2_ok else 'fails-gate2'}")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
