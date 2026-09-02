"""IS-only screen of a purpose-built "trend day" ORB variant: SAME
causal entry (opening-range breakout) and SAME regime gate
(orb_expansion_gate) already validated in the promoted v1 ensemble --
but NO fixed take-profit. Position holds from the breakout to
end-of-day (flat_before_close_minutes) unless stopped out on the wide
structural stop (opposite side of the opening range).

Motivation: the Mesfin (2026) MNQ falsification study found that raw
single-bar/quick-exit OHLCV signals systematically fail OOS, and the
only signals that survived combined regime classification WITH a
multi-bar hold structure. Every candidate tried this session so far
(ThreeDayReversal, IntraMomTP, GapFill, VolProfileMR) paired a gate
with a SHORT hold (minutes to a couple hours) -- none were purpose-
built around "hold the position for the rest of the session once the
regime is confirmed." This is: not a new strategy class, not a new
free parameter search -- literally the same OpeningRangeBreakout class
and orb_expansion_gate already promoted in v1, with tp_multiple/
tp_ticks set to None (ORB already supports this natively) so the ONLY
exit is the wide structural stop or the end-of-day flatten. One
specific, well-motivated hypothesis test, not a grid search.

Run with: python scripts/screen_orb_trendhold_databento.py
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
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)
EASTERN = ZoneInfo("America/New_York")

TRENDHOLD_PARAMS = dict(
    qty=1, or_minutes=30, direction="both",
    stop_mode="opposite_range",   # wide structural stop, not a tight tick stop
    tp_multiple=None, tp_ticks=None,  # NO fixed target -- ride it to close
    flat_before_close_minutes=15,
)


def load_recent_bars(data_path) -> list:
    return [b for b in load_bars_csv(data_path) if b.ts >= RECENT_START]


def run_cell(bars, asset_key, tick_size_f, instrument, gate):
    strat = OpeningRangeBreakout(symbol=asset_key, tick_size=tick_size_f,
                                  daily_filter=gate, **TRENDHOLD_PARAMS)
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
    print("SCREEN: ORB-TrendHold (same entry+gate as v1's ORB, no fixed TP,")
    print("        holds to end-of-day) -- IS-only, ES/NQ/GC")
    print("=" * 78)

    t0 = _time.time()
    rows = {}
    for ak in ASSETS:
        print(f"\nLoading {ak} bars...", flush=True)
        tl = _time.time()
        bars = load_recent_bars(ASSETS[ak]["data_path"])
        print(f"  {len(bars):,} bars in {_time.time()-tl:.1f}s", flush=True)

        stats = per_day_session_stats(bars)
        gate = orb_expansion_gate(stats)

        days = sorted({b.ts.astimezone(EASTERN).date() for b in bars})
        _, _, is_days, _ = is_oos_split(days, {})
        cutoff = is_days[-1]
        cutoff_dt = datetime.combine(cutoff, dtime.min, tzinfo=timezone.utc) + timedelta(days=3)
        bars_is = [b for b in bars if b.ts <= cutoff_dt]
        del bars
        print(f"  IS cutoff={cutoff}, IS bars={len(bars_is):,}", flush=True)

        t1 = _time.time()
        result = run_cell(bars_is, ak, ASSETS[ak]["tick_size_f"], ASSETS[ak]["instrument"], gate)
        row = row_from_result(result)
        el = _time.time() - t1
        if row is None:
            print(f"  ORB_TrendHold: too few active days ({el:.0f}s)")
            del bars_is
            continue
        rows[ak] = row
        gate2_ok = (row["ev_day"] > 0 and row["pf"] > 1.0
                    and row["trades_per_yr"] >= 20 and row["sharpe"] >= 0.3)
        flag = "" if gate2_ok else "  [fails light Gate2]"
        print(f"  ORB_TrendHold: nz={row['nz']:>4} total=${row['total']:>+9,.0f} "
              f"Sharpe={row['sharpe']:>+6.2f} PF={row['pf']:>5.2f} win%={row['win_rate']*100:>4.0f}% "
              f"pass30={row['pass30']:>6.1%} trades/yr={row['trades_per_yr']:>5.0f} "
              f"({el:.0f}s){flag}", flush=True)
        del bars_is

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for ak, row in rows.items():
        gate2_ok = (row["ev_day"] > 0 and row["pf"] > 1.0
                    and row["trades_per_yr"] >= 20 and row["sharpe"] >= 0.3)
        print(f"  {ak}/ORB_TrendHold: pass30={row['pass30']:>6.1%} Sharpe={row['sharpe']:>+6.2f} "
              f"PF={row['pf']:>5.2f} trades/yr={row['trades_per_yr']:>5.0f}  "
              f"{'GATE2-OK' if gate2_ok else 'fails-gate2'}")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
