"""Risk UP as the account proves itself -- the user's next ask after the
tight-risk MeanRev/NQ candidate crossed 90% survival at flat 1-3 micro
contracts. Tests scaling_plan_micros() (analysis/xfa_sizing.py): size
follows Topstep's OWN Scaling Plan (2 -> 3 -> 5 contracts at $1,500 /
$2,000 cumulative profit), read as MICRO contracts, against a more
conservative custom schedule with lower thresholds -- the official
$1,500/$2,000 steps are calibrated for accounts that build profit much
faster than this specific edge's small size does, so there's a real
chance they rarely engage within a 1-year horizon; the custom schedule
tests whether lower thresholds actually get exercised.

Reuses the winning candidate from screen_funded_phase_tight_risk.py:
MeanReversionBollinger/NQ, tight variant (sigma_mult=1.5, stop_ticks=10,
time_stop_minutes=30), full history as the forward-looking basis (same
convention as every other XFA economics script this session).

Run with: python scripts/evaluate_xfa_scaling_plan.py
"""

from __future__ import annotations

import sys
import time as _time
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.xfa_economics import monte_carlo_xfa_economics, take_fixed_amount, take_max_payout
from topstep50k.analysis.xfa_sizing import constant_scale, scaling_plan_micros
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import meanrev_low_vol_gate, per_day_session_stats
from topstep50k.rules import combine_50k
from topstep50k.rules.topstep_xfa import ScalingStep, xfa_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger

NQ_ASSET = {"instrument": Instrument(symbol="NQ", point_value=Decimal("20"), tick_size=Decimal("0.25"),
                                      commission_per_side=Decimal("2.50")),
            "tick_size_f": 0.25, "data_path": ROOT / "data" / "raw" / "nq_databento.txt"}
RULES = combine_50k()
XFA = xfa_50k()
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)
MR_TIGHT_PARAMS = dict(qty=1, lookback=60, sigma_mult=1.5, stop_ticks=10, time_stop_minutes=30)

XFA_HORIZON_DAYS = 252
N_SIMS = 3000
BASE_UNIT_K = 0.1  # 1 "contract" in the scaling plan = 1 micro

# Conservative custom schedule: lower thresholds than Topstep's stated
# $1,500/$2,000 (which this edge's small size may rarely even reach
# within a year), starting at 1 micro instead of 2.
CONSERVATIVE_XFA = replace(XFA, scaling_plan=(
    ScalingStep(Decimal("0"), 1),
    ScalingStep(Decimal("500"), 2),
    ScalingStep(Decimal("1000"), 3),
))
# Much lower still -- pinned close to the $150 winning-day threshold
# itself, to check whether ANY threshold this edge could realistically
# reach within a year makes the scaling mechanism actually engage.
VERY_LOW_XFA = replace(XFA, scaling_plan=(
    ScalingStep(Decimal("0"), 1),
    ScalingStep(Decimal("150"), 2),
    ScalingStep(Decimal("300"), 3),
))


def main():
    print("=" * 96)
    print("XFA SCALING-UP TEST -- risk more once the account has proven it can carry it")
    print("=" * 96)
    t0 = _time.time()

    print("\nLoading NQ bars...", flush=True)
    bars = [b for b in load_bars_csv(NQ_ASSET["data_path"]) if b.ts >= RECENT_START]
    print(f"  {len(bars):,} bars in {_time.time()-t0:.1f}s", flush=True)
    stats = per_day_session_stats(bars)
    gate = meanrev_low_vol_gate(stats)

    strat = MeanReversionBollinger(symbol="NQ", tick_size=NQ_ASSET["tick_size_f"],
                                    daily_filter=gate, **MR_TIGHT_PARAMS)
    pf = PortfolioStrategy(components={"NQ": strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({"NQ": bars}, clk)
    bt = Backtester(rules=RULES, instruments={"NQ": NQ_ASSET["instrument"]},
                     strategy=pf, audit=InMemoryAuditLog(), combine_enforcement=False)
    daily_pnl = bt.run(clk, src).daily_pnl
    days = sorted(daily_pnl)
    arr = np.array([float(daily_pnl[d]) for d in days])
    mean, std = arr.mean(), arr.std()
    print(f"\nMR/NQ [tight], full history {days[0]} -> {days[-1]} ({len(days)}d): "
          f"mean/d=${mean:.1f}  std/d=${std:.1f}  Sharpe(ann)={mean/std*np.sqrt(252):.2f}")
    print(f"Setup: {_time.time()-t0:.0f}s")

    series = [Decimal(str(round(v, 2))) for v in arr]

    def mc(sizing_fn, payout_policy, preserve_cushion):
        return monte_carlo_xfa_economics(
            series, xfa=XFA, horizon_days=XFA_HORIZON_DAYS, n_sims=N_SIMS, block_len=10,
            seed=42, sizing_fn=sizing_fn, payout_policy=payout_policy,
            preserve_cushion=preserve_cushion)

    scenarios = [
        ("flat 1 micro",                    constant_scale(0.1)),
        ("flat 2 micro",                    constant_scale(0.2)),
        ("Topstep plan (2/3/5 @ $0/1.5k/2k)", scaling_plan_micros(XFA, BASE_UNIT_K)),
        ("conservative plan (1/2/3 @ $0/500/1k)", scaling_plan_micros(CONSERVATIVE_XFA, BASE_UNIT_K)),
        ("very-low plan (1/2/3 @ $0/150/300)", scaling_plan_micros(VERY_LOW_XFA, BASE_UNIT_K)),
    ]

    print(f"\n{'='*96}\nRESULTS  [{N_SIMS} sims each]\n{'='*96}")
    for label, sizing_fn in scenarios:
        print(f"\n  {label}:")
        r_max = mc(sizing_fn, take_max_payout, False)
        r_partial = mc(sizing_fn, take_fixed_amount(Decimal("500")), True)
        print(f"    take-max, reset-to-zero:        survive={r_max.prob_survive:6.1%}  "
              f"EV=${r_max.mean_income:>7,.0f}  payouts/yr={r_max.mean_n_payouts:4.2f}  "
              f"DD: mean=${r_max.mean_max_drawdown:>6,.0f} ({r_max.mean_max_drawdown/50000:.1%})  "
              f"p95=${r_max.p95_max_drawdown:>6,.0f}")
        print(f"    partial $500, preserve cushion:  survive={r_partial.prob_survive:6.1%}  "
              f"EV=${r_partial.mean_income:>7,.0f}  payouts/yr={r_partial.mean_n_payouts:4.2f}  "
              f"DD: mean=${r_partial.mean_max_drawdown:>6,.0f} ({r_partial.mean_max_drawdown/50000:.1%})  "
              f"p95=${r_partial.p95_max_drawdown:>6,.0f}")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
