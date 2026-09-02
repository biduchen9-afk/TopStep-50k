"""Replicate Paper 2 (VWAP + StdDev + Williams Fractal, RR=3:1).

CAUSAL implementation: running VWAP, expanding STD of hlc3, fractal
confirmed two bars later. No look-ahead.

Paper claims for reference:
  Period               : 2022-04-12 -> 2026-04-09
  Trades               : 212
  Win rate             : 69.8%
  Avg win              : 21.52 pts (~$430)
  Avg loss             : -7.65 pts (~$-153)
  Profit factor        : 2.81
  Total PnL            : $53,895
  Total return         : 107.8% (on $50K)
  Annualised Sharpe    : ~12.58 (using daily PnL)
  MaxDD (equity)       : ~-1.39%
"""

from __future__ import annotations

import sys
import time as _time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.vwap_std_fractal import VwapStdFractal


# Paper 2 costs: 0.5 pt slippage + 0.25 pt commission per side
# = 0.75 pt/side. NQ point value $20 -> $15/side baked into commission.
NQ = Instrument(symbol="NQ", point_value=Decimal("20"),
                tick_size=Decimal("0.25"),
                commission_per_side=Decimal("15"))
RULES = combine_50k()
STARTING_CAPITAL = 50_000.0


def main():
    print("Loading NQ 1-minute bars...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(ROOT / "data" / "raw" / "nq_cleaned.txt"))
    print(f"  {len(bars):,} bars in {_time.time() - t0:.1f}s", flush=True)

    print("\nRunning VWAP+StdDev+Fractal strategy (paper params, RR=3)...",
          flush=True)
    strat = VwapStdFractal(
        symbol="NQ", qty=1,
        bar_minutes=15,
        band_proximity_points=2.0,    # paper
        min_stop_points=8.0,          # paper
        rr=3.0,                        # paper (fine-tuned)
        flat_before_close_minutes=15, # paper (15:45 ET)
    )
    pf = PortfolioStrategy(components={"NQ": strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({"NQ": bars}, clk)
    audit = InMemoryAuditLog()
    bt = Backtester(rules=RULES, instruments={"NQ": NQ}, strategy=pf,
                    audit=audit, combine_enforcement=False)

    t0 = _time.time()
    result = bt.run(clk, src)
    print(f"  backtest done in {_time.time() - t0:.1f}s", flush=True)

    fills = audit.of_kind("fill")
    print(f"  total fill events: {len(fills)}", flush=True)

    # Pair entries with exits
    trades = []
    open_pos = None
    for f in fills:
        p = f.payload
        side = p["side"]
        price = float(p["price"])
        if open_pos is None:
            open_pos = {"side": +1 if side == "buy" else -1,
                        "qty": p["qty"], "entry_price": price,
                        "entry_ts": f.ts, "entry_tag": p.get("tag", "")}
        else:
            pnl_pts = (price - open_pos["entry_price"]) * open_pos["side"]
            net_pts = pnl_pts - 1.5  # 1.5 pt RT (paper)
            trades.append({
                "entry_ts": open_pos["entry_ts"],
                "exit_ts": f.ts,
                "side": open_pos["side"],
                "entry_price": open_pos["entry_price"],
                "exit_price": price,
                "exit_tag": p.get("tag", ""),
                "gross_pts": pnl_pts,
                "net_pts": net_pts,
            })
            open_pos = None

    if not trades:
        print("  NO trades produced.", flush=True)
        return

    nt = len(trades)
    wins = [t for t in trades if t["net_pts"] > 0]
    losses = [t for t in trades if t["net_pts"] <= 0]
    win_rate = len(wins) / nt
    avg_win = float(np.mean([t["net_pts"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t["net_pts"] for t in losses])) if losses else 0.0
    total_pts = sum(t["net_pts"] for t in trades)
    if losses and avg_loss < 0:
        gross_wins = sum(t["net_pts"] for t in wins)
        gross_losses = -sum(t["net_pts"] for t in losses)
        pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    else:
        pf = float("inf")
    eq = np.cumsum([t["net_pts"] for t in trades])
    peaks = np.maximum.accumulate(eq)
    max_dd_pts = float((eq - peaks).min())

    # Per-trade exit-tag breakdown -- useful diagnostic
    by_tag = defaultdict(int)
    for t in trades:
        by_tag[t["exit_tag"]] += 1

    # Per-year breakdown
    EASTERN = __import__("zoneinfo").ZoneInfo("America/New_York")
    by_year = defaultdict(lambda: {"n": 0, "pnl_pts": 0.0, "wins": 0})
    for t in trades:
        y = t["exit_ts"].astimezone(EASTERN).year
        by_year[y]["n"] += 1
        by_year[y]["pnl_pts"] += t["net_pts"]
        if t["net_pts"] > 0:
            by_year[y]["wins"] += 1

    # Daily PnL ($)
    by_day = defaultdict(float)
    for t in trades:
        d = t["exit_ts"].astimezone(EASTERN).date()
        by_day[d] += t["net_pts"] * 20.0
    days_sorted = sorted(by_day.keys())
    daily_arr = np.array([by_day[d] for d in days_sorted]) if by_day else np.array([])

    print(f"\n{'='*72}")
    print(f"PAPER 2 REPLICATION: VWAP + StdDev + Fractal on NQ (RR=3:1)")
    print(f"{'='*72}")
    print(f"{'Metric':<28}{'Paper claim':>18}{'Our replication':>24}")
    print(f"{'-'*72}")
    print(f"{'Period':<28}{'2022-04-12 -> 2026-04-09':>18}")
    print(f"{'Trades':<28}{'212':>18}{nt:>24,}")
    print(f"{'Win rate':<28}{'69.8%':>18}{win_rate:>24.1%}")
    print(f"{'Avg win (pts)':<28}{'+21.52':>18}{avg_win:>24.2f}")
    print(f"{'Avg loss (pts)':<28}{'-7.65':>18}{avg_loss:>24.2f}")
    print(f"{'Profit factor':<28}{'2.81':>18}{pf:>24.2f}")
    net_usd = total_pts * 20.0
    print(f"{'Net PnL ($, 1 contract)':<28}{'+$53,895':>18}{net_usd:>24,.0f}")
    pct_ret = net_usd / STARTING_CAPITAL * 100
    print(f"{'Total return on $50K':<28}{'+107.8%':>18}{pct_ret:>23.1f}%")
    if daily_arr.size >= 2 and daily_arr.std(ddof=1) > 0:
        sharpe = (daily_arr.mean() / daily_arr.std(ddof=1)) * np.sqrt(252)
    else:
        sharpe = float("nan")
    print(f"{'Annualised Sharpe (daily $)':<28}{'~12.58':>18}{sharpe:>24.2f}")
    print(f"{'MaxDD (pts)':<28}{'~-1.39%':>18}{max_dd_pts:>24,.0f}")
    print(f"{'Trading days w/ PnL':<28}{'':>18}{len(by_day):>24,}")

    print(f"\nExit-tag breakdown:")
    for tag, n in sorted(by_tag.items(), key=lambda x: -x[1]):
        print(f"  {tag:<30} {n:>6,}  ({n/nt:.1%})")

    print(f"\nYear-by-year (our replication):")
    print(f"  {'Year':<8}{'Trades':>10}{'Win %':>10}"
          f"{'PnL (pts)':>14}{'PnL ($)':>14}")
    for y in sorted(by_year.keys()):
        x = by_year[y]
        wr = x["wins"] / x["n"] if x["n"] else 0.0
        print(f"  {y:<8}{x['n']:>10,}{wr:>10.1%}"
              f"{x['pnl_pts']:>14,.0f}{x['pnl_pts'] * 20:>14,.0f}")


if __name__ == "__main__":
    main()
