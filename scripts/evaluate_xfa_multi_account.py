"""How many funded accounts, run in parallel, does it take to turn a
"once every ~110-127 trading days" individual payout rate into
something closer to monthly cash flow? The user's own suggestion after
seeing the single-account payout timing: bring in more accounts,
reinvest payout profit into funding more of them.

Uses the winning candidate (MeanReversionBollinger/NQ, tight variant)
at 1 micro contract with the partial-$500/preserve-cushion payout
policy -- the one scenario that hit 100.0% single-account survival,
so this tests the account-pooling question on the SAFEST base case,
not one where individual accounts are also dying.

Also gives a rough (not a full dynamic simulation) answer to "would
reinvesting payouts actually fund opening more accounts fast enough":
each new account costs ~$220 to pass the Combine (evaluate_combine_
pass_time_cost.py) and takes ~34-46 trading days to do it.

Run with: python scripts/evaluate_xfa_multi_account.py
"""

from __future__ import annotations

import sys
import time as _time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.xfa_economics import monte_carlo_xfa_portfolio, take_fixed_amount
from topstep50k.analysis.xfa_sizing import constant_scale
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import meanrev_low_vol_gate, per_day_session_stats
from topstep50k.rules import combine_50k
from topstep50k.rules.topstep_xfa import xfa_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger

NQ_ASSET = {"instrument": Instrument(symbol="NQ", point_value=Decimal("20"), tick_size=Decimal("0.25"),
                                      commission_per_side=Decimal("2.50")),
            "tick_size_f": 0.25, "data_path": ROOT / "data" / "raw" / "nq_databento.txt"}
RULES = combine_50k()
XFA = xfa_50k()
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)
MR_TIGHT_PARAMS = dict(qty=1, lookback=60, sigma_mult=1.5, stop_ticks=10, time_stop_minutes=30)

XFA_HORIZON_DAYS = 252
N_SIMS = 800
ACCOUNT_COUNTS = [1, 3, 5, 10, 15, 20, 25, 30]

COMBINE_COST_PER_ACCOUNT = 220  # from evaluate_combine_pass_time_cost.py's mean
COMBINE_DAYS_PER_ACCOUNT = 46   # same source, mean trading days to fund


def main():
    print("=" * 96)
    print("MULTI-ACCOUNT POOLING -- how many accounts for ~monthly cash flow?")
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
    series = [Decimal(str(round(v, 2))) for v in arr]
    print(f"\nMR/NQ [tight], full history ({len(days)}d). Setup: {_time.time()-t0:.0f}s")

    sizing_fn = constant_scale(0.1)  # 1 micro
    payout_policy = take_fixed_amount(Decimal("500"))

    print(f"\n{'='*96}\nRESULTS  [{N_SIMS} sims, {XFA_HORIZON_DAYS}-day horizon, 1 micro, "
          f"partial $500 + preserve cushion]\n{'='*96}")
    print(f"{'accounts':>9}{'payouts/yr':>12}{'payouts/mo':>12}{'%months w/ payout':>19}"
          f"{'combined EV':>14}{'mean # breached':>17}")
    t1 = _time.time()
    for n in ACCOUNT_COUNTS:
        r = monte_carlo_xfa_portfolio(
            series, xfa=XFA, n_accounts=n, horizon_days=XFA_HORIZON_DAYS,
            n_sims=N_SIMS, block_len=10, seed=42, sizing_fn=sizing_fn,
            payout_policy=payout_policy, preserve_cushion=True)
        print(f"{n:>9}{r.mean_n_payouts:>12.2f}{r.mean_n_payouts/12:>12.2f}"
              f"{r.frac_months_with_a_payout:>18.1%} "
              f"${r.mean_income:>12,.0f}{r.mean_n_breached:>17.2f}")
    print(f"  ({_time.time()-t1:.0f}s)")

    print(f"\n{'='*96}\nREINVESTMENT ROUGH CHECK -- funding one MORE account from payout income\n{'='*96}")
    print(f"  Combine cost per new account: ~${COMBINE_COST_PER_ACCOUNT} "
          f"(~{COMBINE_DAYS_PER_ACCOUNT} trading days to pass, separately from the funded accounts already running)")
    ev_1_account = 306  # from evaluate_xfa_scaling_plan.py's flat-1-micro/partial result
    payback_years = COMBINE_COST_PER_ACCOUNT / ev_1_account
    print(f"  At ~${ev_1_account}/yr EV per funded account, one account's income alone pays for opening\n"
          f"  another (~${COMBINE_COST_PER_ACCOUNT}) in ~{payback_years:.1f} years -- reinvestment can fund SLOW,\n"
          f"  steady growth in account count, but not a fast compounding ramp at this edge's income level.")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
