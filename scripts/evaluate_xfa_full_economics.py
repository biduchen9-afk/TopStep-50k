"""Full lifecycle economics: cost to pass the $50K Combine + expected
XFA payout income once funded, combined into one net-expected-value
picture. Nothing here has been modeled together before this session --
XFARules/PostPayoutDrawdown existed but were never simulated, and the
Combine-side cost estimate was never combined with a funded-account
income projection.

Two different daily-P&L series are used deliberately:
  Combine phase : the SIZING-OVERLAY-adjusted series (target_proximity
                  @ $1500 -> 0.5x, selected on IS / validated on OOS
                  earlier this session) -- its trigger is specifically
                  the Combine's $1,500 checkpoint, so it only makes
                  sense during the Combine attempt itself.
  XFA phase     : the BASELINE (qty=1, no cutback) series -- once
                  funded there's no $3,000 target to slow down for,
                  so this models running the strategy at full size
                  continuously. (A funded-account-specific sizing
                  overlay is a real, unexplored possible enhancement --
                  out of scope here.)

Both phases use the SAME underlying edge (ORB+MeanRev, IS_WEIGHTS from
live_portfolio.py) over the full available history (2021-12-31 to
present) as the empirical basis for Monte Carlo projection -- this is
a forward-looking projection, not an OOS validity test, so there's no
reason to hold out data here the way strategy SELECTION did.

Run with: python scripts/evaluate_xfa_full_economics.py
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.montecarlo import monte_carlo_pass_rate
from topstep50k.analysis.sizing import target_proximity_scaling
from topstep50k.analysis.xfa_economics import monte_carlo_xfa_economics
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.portfolio.live_portfolio import IS_WEIGHTS
from topstep50k.regime import meanrev_low_vol_gate, orb_expansion_gate, per_day_session_stats
from topstep50k.rules import combine_50k
from topstep50k.rules.topstep_xfa import xfa_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout


ASSETS = {
    "ES": {
        "instrument": Instrument(symbol="ES", point_value=Decimal("50"),
                                  tick_size=Decimal("0.25"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.25,
        "data_path": ROOT / "data" / "raw" / "es_databento.txt",
    },
    "NQ": {
        "instrument": Instrument(symbol="NQ", point_value=Decimal("20"),
                                  tick_size=Decimal("0.25"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.25,
        "data_path": ROOT / "data" / "raw" / "nq_databento.txt",
    },
    "GC": {
        "instrument": Instrument(symbol="GC", point_value=Decimal("100"),
                                  tick_size=Decimal("0.10"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.10,
        "data_path": ROOT / "data" / "raw" / "gc_databento.txt",
    },
}
RULES = combine_50k()
XFA = xfa_50k()
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)
ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both",
                   stop_mode="opposite_range", stop_ticks=40,
                   tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS  = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                   time_stop_minutes=45)

# TopStep $50K Combine, Standard Path pricing (per the earlier session's
# WebSearch-verified numbers -- see docs/rules_sources.md).
REBILL_OR_RESET_COST = Decimal("49")
ACTIVATION_FEE = Decimal("149")

XFA_HORIZON_DAYS = 252  # ~1 trading year


def run_stream(bars, build_fn, asset_key) -> dict[date, Decimal]:
    asset = ASSETS[asset_key]
    strat = build_fn()
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: asset["instrument"]},
                    strategy=pf, audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, src).daily_pnl


def to_series(daily_pnl: dict, day_index: list[date]) -> np.ndarray:
    out = np.zeros(len(day_index), dtype=float)
    for i, d in enumerate(day_index):
        if d in daily_pnl:
            out[i] = float(daily_pnl[d])
    return out


def main():
    print("=" * 78)
    print("FULL LIFECYCLE ECONOMICS -- Combine cost + XFA payout income")
    print("=" * 78)

    t0 = _time.time()
    all_bars, gates = {}, {}
    for ak in ASSETS:
        print(f"\nLoading {ak} bars...", flush=True)
        tl = _time.time()
        bars = [b for b in load_bars_csv(ASSETS[ak]["data_path"]) if b.ts >= RECENT_START]
        print(f"  {len(bars):,} bars in {_time.time()-tl:.1f}s", flush=True)
        all_bars[ak] = bars
        stats = per_day_session_stats(bars)
        gates[ak] = {"orb": orb_expansion_gate(stats), "mr": meanrev_low_vol_gate(stats)}

    print("\nRunning 6 ORB+MeanRev streams (live_portfolio.IS_WEIGHTS)...", flush=True)
    streams = {}
    for ak in ASSETS:
        ts = ASSETS[ak]["tick_size_f"]
        bars = all_bars[ak]
        g = gates[ak]
        streams[(ak, "ORB")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: OpeningRangeBreakout(
                symbol=a, tick_size=tts, daily_filter=gg["orb"], **ORB_PARAMS), ak)
        streams[(ak, "MeanRev")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: MeanReversionBollinger(
                symbol=a, tick_size=tts, daily_filter=gg["mr"], **MR_PARAMS), ak)
    all_bars.clear()

    union_days = sorted(set().union(*[set(s.keys()) for s in streams.values()]))
    full_arrays = {k: to_series(pnl, union_days) for k, pnl in streams.items()}

    z = sum(IS_WEIGHTS.values())
    n = len(union_days)
    ensemble = np.zeros(n, dtype=float)
    for k, w in IS_WEIGHTS.items():
        if k in full_arrays:
            ensemble += (w / z) * full_arrays[k]
    daily = {d: Decimal(str(round(float(v), 2))) for d, v in zip(union_days, ensemble)}
    print(f"\nFull history: {union_days[0]} -> {union_days[-1]} ({n} d).  "
          f"Setup: {_time.time()-t0:.0f}s")

    # ── COMBINE PHASE: sizing-overlay-adjusted pass rate ───────────────────
    print(f"\n{'='*78}\nCOMBINE PHASE (sizing overlay: target_proximity @ $1500 -> 0.5x)\n{'='*78}")
    sizing_fn = target_proximity_scaling(Decimal("1500"), 0.5)
    t1 = _time.time()
    mc_combine = monte_carlo_pass_rate(daily, rules=RULES, starting_balance=RULES.starting_balance,
                                        n_sims=2000, block_len=10, checkpoint=Decimal("1500"),
                                        seed=42, sizing_fn=sizing_fn)
    print(f"  ({_time.time()-t1:.1f}s) Monte Carlo pass rate: mean={mc_combine.mean_pass_rate:.1%}  "
          f"90%CI=[{mc_combine.p05:.1%},{mc_combine.p95:.1%}]")

    p = mc_combine.mean_pass_rate
    p_lo, p_hi = mc_combine.p05, mc_combine.p95
    def expected_cost(pass_rate: float) -> float:
        attempts = 1.0 / pass_rate
        return attempts * float(REBILL_OR_RESET_COST) + float(ACTIVATION_FEE)
    cost_mean = expected_cost(p)
    cost_lo = expected_cost(p_hi)  # higher pass rate -> lower cost
    cost_hi = expected_cost(p_lo)  # lower pass rate -> higher cost
    print(f"  Expected cost to pass: ~${cost_mean:,.0f}  (range ~${cost_lo:,.0f}-${cost_hi:,.0f} "
          f"across the 90% pass-rate interval)")

    # ── XFA PHASE: baseline (unscaled) daily P&L, full-size continuous ────
    print(f"\n{'='*78}\nXFA PHASE ({XFA_HORIZON_DAYS}-day horizon, baseline full-size trading)\n{'='*78}")
    t2 = _time.time()
    mc_xfa = monte_carlo_xfa_economics(list(daily.values()), xfa=XFA,
                                        horizon_days=XFA_HORIZON_DAYS,
                                        n_sims=2000, block_len=10, seed=42)
    print(f"  ({_time.time()-t2:.1f}s) Trader income distribution over {XFA_HORIZON_DAYS} days:")
    print(f"    mean=${mc_xfa.mean_income:,.0f}  median=${mc_xfa.median_income:,.0f}  "
          f"90%CI=[${mc_xfa.p05_income:,.0f}, ${mc_xfa.p95_income:,.0f}]")
    print(f"    P(MLL breach within horizon) = {mc_xfa.prob_breach:.1%}")
    print(f"    mean payouts per horizon = {mc_xfa.mean_n_payouts:.1f}")
    if mc_xfa.mean_days_to_first_payout is not None:
        print(f"    mean days to first payout (sims with >=1 payout) = "
              f"{mc_xfa.mean_days_to_first_payout:.0f}")

    # ── COMBINED PICTURE ────────────────────────────────────────────────
    print(f"\n{'='*78}\nNET EXPECTED VALUE ({XFA_HORIZON_DAYS}-day horizon after funding)\n{'='*78}")
    net_mean = mc_xfa.mean_income - cost_mean
    print(f"  E[XFA income]  = ${mc_xfa.mean_income:>+9,.0f}")
    print(f"  E[Combine cost]= ${-cost_mean:>+9,.0f}")
    print(f"  {'-'*30}")
    print(f"  NET E[value]   = ${net_mean:>+9,.0f}")
    print(f"\n  CAVEATS (read before trusting this number):")
    print(f"  - P(breach within {XFA_HORIZON_DAYS}d) = {mc_xfa.prob_breach:.1%} -- a breach means")
    print(f"    losing the funded account and re-incurring the Combine cost to try")
    print(f"    again; this simple single-cycle model does NOT compound that risk")
    print(f"    across multiple XFA lifecycles, so treat NET E[value] as optimistic")
    print(f"    if the breach probability is high.")
    print(f"  - PostPayoutDrawdown's post-payout floor reset is a CONSERVATIVE,")
    print(f"    third-party-sourced assumption (see rules/topstep_xfa.py), not")
    print(f"    confirmed against Topstep's own wording.")
    print(f"  - Payout eligibility is modeled as resetting after each payout,")
    print(f"    not cumulative-since-funding -- also unconfirmed (see")
    print(f"    docs/rules_sources.md). If eligibility is actually cumulative,")
    print(f"    real payout frequency (and therefore income) would be higher.")
    print(f"  - XFA phase uses the BASELINE (unsized) edge -- no funded-account-")
    print(f"    specific sizing overlay has been explored yet.")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
