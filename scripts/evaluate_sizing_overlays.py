"""Evaluate position-sizing overlays against the real ORB+MeanRev OOS
daily P&L series (v10 -- the current best real, legal result).

Not alpha search: same underlying edge, different bet-sizing policy
against the Combine's discrete pass/fail risk structure. Compares:
  baseline               -- qty=1 throughout (v10 as already reported)
  target_proximity_50pct -- half size once profit >= $1,500
  target_proximity_75pct -- half size once profit >= $2,250
  drawdown_responsive    -- half size the day after a loss
  combined                -- min() of the two rules above

Each policy's sequential pass rate AND Monte Carlo distribution are
reported so a policy only "wins" if it improves the honest, resampled
number -- not just the single realized path.

Run with: python scripts/evaluate_sizing_overlays.py
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
from topstep50k.analysis.passrate import realized_pass_rate, simulate_sequential_accounts
from topstep50k.analysis.sizing import (
    combined_scaling,
    drawdown_responsive_scaling,
    full_size,
    simulate_sequential_accounts_sized,
    target_proximity_scaling,
)
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.evaluation.harness import is_oos_split
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import meanrev_low_vol_gate, orb_expansion_gate, per_day_session_stats
from topstep50k.rules import combine_50k
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
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)
ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both",
                   stop_mode="opposite_range", stop_ticks=40,
                   tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS  = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                   time_stop_minutes=45)


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


def pass_rate_30(arr: np.ndarray, days: list[date]):
    pnl = {d: Decimal(str(round(float(v), 2))) for d, v in zip(days, arr)}
    return realized_pass_rate(pnl, rules=RULES, starting_balance=RULES.starting_balance,
                               window_days=30, stride_days=1)


def main():
    print("=" * 78)
    print("SIZING OVERLAYS -- evaluated on the real ORB+MeanRev OOS series (v10)")
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

    print("\nRunning 6 ORB+MeanRev streams...", flush=True)
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
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
    full_arrays = {k: to_series(pnl, union_days) for k, pnl in streams.items()}
    is_arrays  = {k: a[is_mask]  for k, a in full_arrays.items()}
    oos_arrays = {k: a[oos_mask] for k, a in full_arrays.items()}

    is_pr30, is_ev_positive = {}, {}
    for k, arr in is_arrays.items():
        rr30 = pass_rate_30(arr, is_days)
        is_pr30[k] = rr30.pass_rate
        is_ev_positive[k] = float(arr.mean()) > 0
    gated_pr30 = {k: (v if is_ev_positive[k] else 0.0) for k, v in is_pr30.items()}
    total_pr = sum(gated_pr30.values())
    weights = ({k: v / total_pr for k, v in gated_pr30.items()}
               if total_pr > 0 else {k: 1.0 / len(is_pr30) for k in is_pr30})

    def ensemble(arrays_dict, wt):
        z = sum(wt.values())
        n = next(iter(arrays_dict.values())).size
        out = np.zeros(n, dtype=float)
        for k, w in wt.items():
            out += (w / z) * arrays_dict[k]
        return out

    ens_oos = ensemble(oos_arrays, weights)
    oos_daily = {d: Decimal(str(round(float(v), 2))) for d, v in zip(oos_days, ens_oos)}
    print(f"\nOOS: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d), "
          f"streams+weights match v10 exactly. Setup: {_time.time()-t0:.0f}s")

    policies = {
        "baseline (qty=1 always)":       full_size,
        "target_proximity @ $1500 -> 0.5x": target_proximity_scaling(Decimal("1500"), 0.5),
        "target_proximity @ $2250 -> 0.5x": target_proximity_scaling(Decimal("2250"), 0.5),
        "drawdown_responsive -> 0.5x":      drawdown_responsive_scaling(0.5),
        "combined (both @ 0.5x)":           combined_scaling(
            target_proximity_scaling(Decimal("1500"), 0.5),
            drawdown_responsive_scaling(0.5),
        ),
    }

    print(f"\n{'='*78}\nRESULTS\n{'='*78}")
    for name, sizing_fn in policies.items():
        t1 = _time.time()
        if sizing_fn is full_size:
            seq = simulate_sequential_accounts(oos_daily, rules=RULES,
                                                starting_balance=RULES.starting_balance,
                                                checkpoint=Decimal("1500"))
        else:
            seq = simulate_sequential_accounts_sized(
                oos_daily, rules=RULES, starting_balance=RULES.starting_balance,
                sizing_fn=sizing_fn, checkpoint=Decimal("1500"))
        mc = monte_carlo_pass_rate(oos_daily, rules=RULES, starting_balance=RULES.starting_balance,
                                    n_sims=2000, block_len=10, checkpoint=Decimal("1500"),
                                    seed=42, sizing_fn=sizing_fn)
        print(f"\n{name}:")
        print(f"  Sequential: {seq.n_accounts} accounts, {seq.count('pass')} pass "
              f"({seq.pass_rate:.1%})  [{_time.time()-t1:.1f}s]")
        print(f"  Monte Carlo: mean={mc.mean_pass_rate:.1%}  median={mc.median_pass_rate:.1%}  "
              f"90%CI=[{mc.p05:.1%},{mc.p95:.1%}]  P(>50%)={mc.prob_above_50pct:.1%}")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
