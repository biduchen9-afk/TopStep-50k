"""Replicate Paper 1 (Intraday Volume-Profile Mean-Reversion) on the NQ
1-minute file. Report trade-level stats side-by-side with the paper's
claims so any discrepancy is visible.

CAUSAL implementation: VWAP/STD are running session statistics over
CLOSED 5-min bars only. No look-ahead. The paper's claims look
implausibly high (Sharpe in the 15-20 range, profit factor 4.59) so
this is also a bug-hunt for whichever look-ahead the paper's
ChatGPT-generated code had.

Paper claims for reference:
  Period           : 2022-04-07 to 2026-04-10
  Trades           : 7,813
  Win rate         : 66.8%
  Avg win          : 38.3 pts  ($766 at NQ $20/pt)
  Avg loss         : -16.8 pts ($-336)
  Profit factor    : 4.59
  Net PnL          : 156,232 pts (~$3.1M)
  MaxDD            : ~-311 pts (~$6.2K)
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
from topstep50k.strategy.volume_profile_mr import VolumeProfileMeanReversion


# Paper 1 cost: 0.5 pt slippage + 0.25 pt commission per side = 0.75 pt/side.
# On NQ ($20/pt) that's $15/side -- bake it all into commission_per_side
# since our engine fills at exact next-bar-open.
NQ = Instrument(symbol="NQ", point_value=Decimal("20"),
                tick_size=Decimal("0.25"),
                commission_per_side=Decimal("15"))
RULES = combine_50k()


def main():
    print("Loading NQ 1-minute bars...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(ROOT / "data" / "raw" / "nq_cleaned.txt"))
    print(f"  {len(bars):,} bars in {_time.time() - t0:.1f}s", flush=True)
    print(f"  range: {bars[0].ts} -> {bars[-1].ts}", flush=True)

    print("\nRunning Volume-Profile Mean-Reversion (paper params)...",
          flush=True)
    strat = VolumeProfileMeanReversion(
        symbol="NQ", qty=1,
        bar_minutes=5,
        entry_threshold_points=2.0,   # paper
        stop_sigma_mult=2.0,          # paper
        flat_before_close_minutes=15, # paper (15:45 ET)
        min_warm_5min_bars=6,         # 30 min warm-up
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

    # -------- trade extraction from the audit log --------
    fills = audit.of_kind("fill")
    print(f"  total fill events: {len(fills)}", flush=True)

    # Pair entries with exits to form trades
    trades = []
    open_pos = None
    for f in fills:
        p = f.payload
        side = p["side"]
        qty = p["qty"]
        price = float(p["price"])
        if open_pos is None:
            open_pos = {"side": +1 if side == "buy" else -1,
                        "qty": qty, "entry_price": price,
                        "entry_ts": f.ts, "entry_tag": p.get("tag", "")}
        else:
            # closing fill
            exit_price = price
            pnl_pts = (exit_price - open_pos["entry_price"]) * open_pos["side"]
            # commission_per_side already baked in (one-sided) -- here we
            # have TWO fills total per round trip, each charged side-wise.
            # The engine does that internally so PnL via ledger is already
            # net-of-cost. But for the per-trade summary we recompute
            # gross points and subtract paper's 1.5 pt round-trip cost.
            gross_pts = pnl_pts
            net_pts = gross_pts - 1.5  # 1.5 pt RT per the paper
            trades.append({
                "entry_ts": open_pos["entry_ts"],
                "exit_ts": f.ts,
                "side": open_pos["side"],
                "entry_price": open_pos["entry_price"],
                "exit_price": exit_price,
                "entry_tag": open_pos["entry_tag"],
                "exit_tag": p.get("tag", ""),
                "gross_pts": gross_pts,
                "net_pts": net_pts,
            })
            open_pos = None

    if not trades:
        print("  NO trades produced. Investigating...", flush=True)
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

    # max drawdown on the trade-by-trade PnL series
    eq = np.cumsum([t["net_pts"] for t in trades])
    peaks = np.maximum.accumulate(eq)
    max_dd_pts = float((eq - peaks).min())

    # daily PnL (in dollars at $20/pt) for pass-rate / Sharpe later
    by_day = defaultdict(float)
    from datetime import timezone
    EASTERN = __import__("zoneinfo").ZoneInfo("America/New_York")
    for t in trades:
        d = t["exit_ts"].astimezone(EASTERN).date()
        by_day[d] += t["net_pts"] * 20.0  # $20/pt for NQ

    print(f"\n{'='*70}")
    print(f"PAPER 1 REPLICATION: Volume-Profile Mean-Reversion on NQ")
    print(f"{'='*70}")
    print(f"{'Metric':<25}{'Paper claim':>18}{'Our replication':>22}")
    print(f"{'-'*70}")
    print(f"{'Period':<25}{'2022-04-07 -> 2026-04-10':>18}")
    print(f"{'Trades':<25}{'7,813':>18}{nt:>22,}")
    print(f"{'Win rate':<25}{'66.8%':>18}{win_rate:>22.1%}")
    print(f"{'Avg win (pts)':<25}{'+38.3':>18}{avg_win:>22.2f}")
    print(f"{'Avg loss (pts)':<25}{'-16.8':>18}{avg_loss:>22.2f}")
    print(f"{'Profit factor':<25}{'4.59':>18}{pf:>22.2f}")
    print(f"{'Net PnL (pts)':<25}{'+156,232':>18}{total_pts:>22,.0f}")
    net_usd = total_pts * 20.0
    print(f"{'Net PnL ($, 1 contract)':<25}{'+$3,124,640':>18}{net_usd:>22,.0f}")
    print(f"{'Max DD (pts)':<25}{'~-311':>18}{max_dd_pts:>22,.0f}")
    print(f"{'Trading days w/ PnL':<25}{'':>18}{len(by_day):>22,}")

    # Year-by-year (matches paper's breakdown)
    by_year = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in trades:
        y = t["exit_ts"].astimezone(EASTERN).year
        by_year[y]["n"] += 1
        by_year[y]["pnl"] += t["net_pts"]
    print(f"\nYear-by-year (our replication):")
    print(f"  {'Year':<8}{'Trades':>10}{'PnL (pts)':>16}{'Mean/trade':>14}")
    for y in sorted(by_year.keys()):
        x = by_year[y]
        mean_pnl = x["pnl"] / x["n"] if x["n"] else 0.0
        print(f"  {y:<8}{x['n']:>10,}{x['pnl']:>16,.0f}{mean_pnl:>14.2f}")

    # Annualised Sharpe on daily PnL series
    days_sorted = sorted(by_day.keys())
    daily_arr = np.array([by_day[d] for d in days_sorted])
    if daily_arr.size >= 2 and daily_arr.std(ddof=1) > 0:
        sharpe = (daily_arr.mean() / daily_arr.std(ddof=1)) * np.sqrt(252)
    else:
        sharpe = float("nan")
    print(f"\n{'Annualised Sharpe (daily, $)':<32}{sharpe:>22.2f}")
    print(f"{'Mean daily $':<32}{daily_arr.mean():>22,.0f}")
    print(f"{'Stdev daily $':<32}{daily_arr.std(ddof=1):>22,.0f}")
    print(f"{'Best day $':<32}{daily_arr.max():>22,.0f}")
    print(f"{'Worst day $':<32}{daily_arr.min():>22,.0f}")


if __name__ == "__main__":
    main()
