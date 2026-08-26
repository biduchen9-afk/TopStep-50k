"""IS-only screen of 5 day-session-only strategy candidates to replace OD.

OvernightDrift is retired (TopStep disallows overnight/weekend holding
at every account stage, confirmed July 2026 -- see docs/rules_sources.md).
v10 (ORB+MeanRev only) failed Gate 2/3 -- two strategies of tested edge
isn't enough on this asset set. This screens 5 ALREADY-BUILT, day-session
-only, literature-grounded strategies that were never evaluated against
the databento dataset, using their pre-committed (no-data-tuning) default
parameters -- no grid search yet, matching how ORB/MeanRev/OD were first
screened before any gating was added:

  GapFill               -- fade the overnight RTH gap (Avellaneda & Lee 2010)
  IntradayMomentum      -- first-30min return predicts last-30min (Gao et al. 2018)
  IntradayMomentumTP    -- same signal, symmetric 1:1 RR variant
  VolumeProfileMeanRev  -- VWAP value-area fade (5-min bars)
  VwapStdFractal        -- VWAP+stdev+Williams-fractal pullback (15-min bars)
  ThreeDayReversal      -- mean reversion after 3 consecutive same-direction closes

All are flat well before TopStep's 3:10 PM CT cutoff (exit_time_local /
flat_before_close_minutes fields), so none of them re-run into the
overnight-holding problem. Ungated (no daily_filter) for this first pass,
same discipline used for the original ORB/MeanRev/OD screens.

Run with: python scripts/screen_new_daysession_strategies.py
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date, datetime, time as dtime, timedelta, timezone
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
from topstep50k.strategy.gap_fill import GapFill
from topstep50k.strategy.intraday_momentum import IntradayMomentum
from topstep50k.strategy.intraday_momentum_tp import IntradayMomentumTP
from topstep50k.strategy.three_day_reversal import ThreeDayReversal
from topstep50k.strategy.volume_profile_mr import VolumeProfileMeanReversion
from topstep50k.strategy.vwap_std_fractal import VwapStdFractal


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

STRATEGY_BUILDERS = {
    "GapFill":       lambda a, ts: GapFill(symbol=a, tick_size=ts),
    "IntraMom":      lambda a, ts: IntradayMomentum(symbol=a, tick_size=ts),
    "IntraMomTP":    lambda a, ts: IntradayMomentumTP(symbol=a, tick_size=ts),
    "VolProfileMR":  lambda a, ts: VolumeProfileMeanReversion(symbol=a, tick_size=ts),
    "VwapFractal":   lambda a, ts: VwapStdFractal(symbol=a, tick_size=ts),
    "ThreeDayRev":   lambda a, ts: ThreeDayReversal(symbol=a, tick_size=ts),
}


def run_cell(bars, asset_key, tick_size_f, instrument, strat_name):
    strat = STRATEGY_BUILDERS[strat_name](asset_key, tick_size_f)
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: instrument}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, src)


def load_recent_bars(data_path) -> list:
    return [b for b in load_bars_csv(data_path) if b.ts >= RECENT_START]


def row_from_result(result, is_days):
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
        "win_rate": perf.win_rate, "pass30": rr30.pass_rate,
        "nz": int((arr != 0).sum()), "total": float(perf.total_pnl),
    }


def main():
    print("=" * 78)
    print("SCREEN: 5 new day-session strategy candidates x ES/NQ/GC, IS-only")
    print("Pre-committed (no-tuning) default params, ungated, recent window")
    print("=" * 78)

    t0 = _time.time()
    rows: dict[tuple[str, str], dict] = {}
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

        for sname in STRATEGY_BUILDERS:
            t1 = _time.time()
            result = run_cell(bars_is, ak, ASSETS[ak]["tick_size_f"], ASSETS[ak]["instrument"], sname)
            row = row_from_result(result, is_days)
            el = _time.time() - t1
            if row is None:
                print(f"  {sname:<12}: too few active days ({el:.0f}s)")
                continue
            rows[(ak, sname)] = row
            gate2_ok = (row["ev_day"] > 0 and row["pf"] > 1.0
                        and row["trades_per_yr"] >= 20 and row["sharpe"] >= 0.3)
            flag = "" if gate2_ok else "  [fails light Gate2]"
            print(f"  {sname:<12}: nz={row['nz']:>4} total=${row['total']:>+9,.0f} "
                  f"Sharpe={row['sharpe']:>+6.2f} PF={row['pf']:>5.2f} "
                  f"pass30={row['pass30']:>6.1%} trades/yr={row['trades_per_yr']:>5.0f} "
                  f"({el:.0f}s){flag}", flush=True)
        del bars_is

    print(f"\n{'='*78}\nSUMMARY -- ranked by IS pass30\n{'='*78}")
    ranked = sorted(rows.items(), key=lambda kv: -kv[1]["pass30"])
    for (ak, sname), row in ranked:
        gate2_ok = (row["ev_day"] > 0 and row["pf"] > 1.0
                    and row["trades_per_yr"] >= 20 and row["sharpe"] >= 0.3)
        print(f"  {ak}/{sname:<12}: pass30={row['pass30']:>6.1%} Sharpe={row['sharpe']:>+6.2f} "
              f"PF={row['pf']:>5.2f} trades/yr={row['trades_per_yr']:>5.0f}  "
              f"{'GATE2-OK' if gate2_ok else 'fails-gate2'}")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
