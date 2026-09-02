"""How much, how long, how many: the three Combine-phase numbers the
user actually cares about (funded-phase economics are a separate,
later question -- see xfa_economics.py / evaluate_xfa_full_economics.py
for that, deliberately NOT re-run here).

Reuses the exact same edge and sizing overlay already validated this
session (ORB+MeanRev ensemble, live_portfolio.IS_WEIGHTS, sizing
overlay = target_proximity @ $1500 -> 0.5x, selected on IS / touched
OOS once) -- no new fitting happens here, this just adds a
TIME dimension (trading days to fund) to the existing pass-rate/cost
Monte Carlo by tracking each resampled path's sequence of account
attempts up to and including the first pass, instead of only the
aggregate pass rate across the whole path.

Run with: python scripts/evaluate_combine_pass_time_cost.py
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

from topstep50k.analysis.sizing import target_proximity_scaling, simulate_sequential_accounts_sized
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.portfolio.live_portfolio import IS_WEIGHTS
from topstep50k.regime import meanrev_low_vol_gate, orb_expansion_gate, per_day_session_stats
from topstep50k.rules import combine_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout

ASSETS = {
    "ES": {"instrument": Instrument(symbol="ES", point_value=Decimal("50"), tick_size=Decimal("0.25"),
                                     commission_per_side=Decimal("2.50")),
           "tick_size_f": 0.25, "data_path": ROOT / "data" / "raw" / "es_databento.txt"},
    "NQ": {"instrument": Instrument(symbol="NQ", point_value=Decimal("20"), tick_size=Decimal("0.25"),
                                     commission_per_side=Decimal("2.50")),
           "tick_size_f": 0.25, "data_path": ROOT / "data" / "raw" / "nq_databento.txt"},
    "GC": {"instrument": Instrument(symbol="GC", point_value=Decimal("100"), tick_size=Decimal("0.10"),
                                     commission_per_side=Decimal("2.50")),
           "tick_size_f": 0.10, "data_path": ROOT / "data" / "raw" / "gc_databento.txt"},
}
RULES = combine_50k()
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)
ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both", stop_mode="opposite_range",
                   stop_ticks=40, tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15, time_stop_minutes=45)

REBILL_OR_RESET_COST = Decimal("49")
ACTIVATION_FEE = Decimal("149")

N_SIMS = 2000
BLOCK_LEN = 10
CHECKPOINT = Decimal("1500")
SEED = 42


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
    print("COMBINE PHASE -- pass rate, trading days to fund, dollar cost to fund")
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
    daily_values = ensemble
    print(f"\nFull history: {union_days[0]} -> {union_days[-1]} ({n} d).  Setup: {_time.time()-t0:.0f}s")

    sizing_fn = target_proximity_scaling(CHECKPOINT, 0.5)

    rng = np.random.default_rng(SEED)
    n_blocks = -(-n // BLOCK_LEN)

    pass_rates = np.empty(N_SIMS)
    days_to_fund = []      # trading days elapsed across all attempts up to & incl. the first pass
    attempts_to_fund = []  # number of account attempts up to & incl. the first pass
    cost_to_fund = []      # dollar cost of those attempts
    no_pass_in_horizon = 0

    t1 = _time.time()
    for s in range(N_SIMS):
        starts = rng.integers(0, n, size=n_blocks)
        blocks = []
        for st in starts:
            if st + BLOCK_LEN <= n:
                blocks.append(daily_values[st:st + BLOCK_LEN])
            else:
                blocks.append(np.concatenate([daily_values[st:], daily_values[:BLOCK_LEN - (n - st)]]))
        resampled = np.concatenate(blocks)[:n]
        synth_daily = {d: Decimal(str(round(float(v), 2))) for d, v in zip(union_days, resampled)}

        summary = simulate_sequential_accounts_sized(
            synth_daily, rules=RULES, starting_balance=RULES.starting_balance,
            sizing_fn=sizing_fn, checkpoint=CHECKPOINT,
        )
        pass_rates[s] = summary.pass_rate

        cum_days = 0
        found = False
        for k, acct in enumerate(summary.accounts):
            cum_days += acct.n_days
            if acct.outcome == "pass":
                days_to_fund.append(cum_days)
                attempts_to_fund.append(k + 1)
                # first attempt uses the $149 activation fee; every
                # subsequent restart after a breach is a $49 reset/rebill
                cost_to_fund.append(float(ACTIVATION_FEE) + k * float(REBILL_OR_RESET_COST))
                found = True
                break
        if not found:
            no_pass_in_horizon += 1

    days_to_fund = np.array(days_to_fund)
    attempts_to_fund = np.array(attempts_to_fund)
    cost_to_fund = np.array(cost_to_fund)

    print(f"\n({_time.time()-t1:.1f}s) {N_SIMS} resampled paths, block_len={BLOCK_LEN}")
    print(f"\n{'='*78}\nRESULTS\n{'='*78}")
    print(f"\n1) PASS RATE (per Combine account attempt)")
    print(f"   mean = {pass_rates.mean():.1%}   median = {np.median(pass_rates):.1%}")
    print(f"   90% CI = [{np.percentile(pass_rates,5):.1%}, {np.percentile(pass_rates,95):.1%}]")

    print(f"\n2) HOW LONG -- trading days from a fresh account until first pass")
    print(f"   (across {len(days_to_fund)}/{N_SIMS} sims that funded within the "
          f"available history; {no_pass_in_horizon} did not fund in the sample)")
    print(f"   mean = {days_to_fund.mean():.0f} trading days   median = {np.median(days_to_fund):.0f} trading days")
    print(f"   90% CI = [{np.percentile(days_to_fund,5):.0f}, {np.percentile(days_to_fund,95):.0f}] trading days")
    print(f"   in calendar terms (~21 trading days/month): "
          f"mean ~= {days_to_fund.mean()/21:.1f} months, "
          f"90% CI ~= [{np.percentile(days_to_fund,5)/21:.1f}, {np.percentile(days_to_fund,95)/21:.1f}] months")

    print(f"\n3) HOW MANY ACCOUNTS -- attempts needed until first pass")
    print(f"   mean = {attempts_to_fund.mean():.2f}   median = {np.median(attempts_to_fund):.0f}")
    print(f"   90% CI = [{np.percentile(attempts_to_fund,5):.0f}, {np.percentile(attempts_to_fund,95):.0f}] attempts")

    print(f"\n4) HOW MUCH -- dollar cost until first pass "
          f"(${ACTIVATION_FEE} activation + ${REBILL_OR_RESET_COST} per restart after a breach)")
    print(f"   mean = ${cost_to_fund.mean():,.0f}   median = ${np.median(cost_to_fund):,.0f}")
    print(f"   90% CI = [${np.percentile(cost_to_fund,5):,.0f}, ${np.percentile(cost_to_fund,95):,.0f}]")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
