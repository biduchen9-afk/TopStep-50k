"""Structural search: can a risk-adjustment (position-sizing) policy on
the SAME edge push the funded-account (XFA) survival probability toward
the user's >=90% target, and what does that cost in expected income?

This is EXPLORATORY -- a grid scan across policy families/parameters
using the full history as the Monte Carlo resampling basis, same as
evaluate_xfa_full_economics.py's XFA-phase treatment (a forward-looking
projection, not a strategy-selection test, so no OOS split is needed to
avoid leakage on the EDGE itself). But POLICY selection over many grid
points is its own multiple-comparisons exposure -- this script's output
is a Pareto view (survival vs. mean income) to pick a couple of
candidates from, not a final validated number. Whatever gets picked
here should still get a single held-out check before being called
"the" answer, the same discipline used for the Combine-phase sizing
overlay (see evaluate_sizing_overlays.py's docstring).

First re-establishes the CORRECTED baseline (xfa_full_size) using the
just-fixed simulate_xfa_lifecycle -- the earlier session's "100% breach
within 1 year" figure was partly a same-day tautological-breach bug
(every payout re-anchors the floor to exactly the post-payout balance,
so `balance <= line` was trivially true the instant ANY payout
cleared, regardless of any real subsequent loss); see
analysis/xfa_economics.py's fix note. This script's baseline number is
the first honest read of that probability.

Run with: python scripts/evaluate_xfa_sizing_overlays.py
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

from topstep50k.analysis.xfa_economics import monte_carlo_xfa_economics, xfa_full_size
from topstep50k.analysis.xfa_sizing import (
    combined_xfa_scaling,
    cushion_proportional_scaling,
    hard_stop_after,
    post_payout_cooldown,
    time_decay_scaling,
)
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
XFA = xfa_50k()
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)
ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both", stop_mode="opposite_range",
                   stop_ticks=40, tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15, time_stop_minutes=45)

XFA_HORIZON_DAYS = 252
N_SIMS_SCAN = 500
N_SIMS_FINAL = 2000
BLOCK_LEN = 10
SEED = 42
SURVIVAL_TARGET = 0.90


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


def report(label: str, mc, n_sims: int) -> dict:
    print(f"  {label:<38} survive={mc.prob_survive:6.1%}  "
          f"mean_income=${mc.mean_income:>7,.0f}  median=${mc.median_income:>7,.0f}  "
          f"payouts/yr={mc.mean_n_payouts:4.2f}")
    return {"label": label, "survive": mc.prob_survive, "mean_income": mc.mean_income,
            "median_income": mc.median_income, "payouts": mc.mean_n_payouts}


def main():
    print("=" * 90)
    print("XFA RISK-ADJUSTMENT SEARCH -- can sizing alone reach >=90% survival?")
    print("=" * 90)
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
    ensemble = np.zeros(len(union_days), dtype=float)
    for k, w in IS_WEIGHTS.items():
        if k in full_arrays:
            ensemble += (w / z) * full_arrays[k]
    daily_pnl_list = [Decimal(str(round(float(v), 2))) for v in ensemble]
    print(f"\nFull history: {union_days[0]} -> {union_days[-1]} ({len(union_days)} d).  "
          f"Setup: {_time.time()-t0:.0f}s")

    def mc(sizing_fn, n_sims=N_SIMS_SCAN):
        return monte_carlo_xfa_economics(
            daily_pnl_list, xfa=XFA, horizon_days=XFA_HORIZON_DAYS,
            n_sims=n_sims, block_len=BLOCK_LEN, seed=SEED, sizing_fn=sizing_fn)

    print(f"\n{'='*90}\nCORRECTED BASELINE (post-bugfix, full size, qty=1 continuous)\n{'='*90}")
    t1 = _time.time()
    baseline = mc(xfa_full_size, n_sims=N_SIMS_FINAL)
    results = [report("baseline (full size)", baseline, N_SIMS_FINAL)]
    print(f"  ({_time.time()-t1:.1f}s)")

    print(f"\n{'='*90}\nGRID: cushion_proportional_scaling(floor_frac, ramp_frac)  [{N_SIMS_SCAN} sims]\n{'='*90}")
    t1 = _time.time()
    for floor_frac in (0.0, 0.1, 0.2, 0.4):
        for ramp_frac in (0.5, 1.0):
            fn = cushion_proportional_scaling(XFA.mll_distance, floor_frac, ramp_frac)
            results.append(report(f"cushion(floor={floor_frac},ramp={ramp_frac})", mc(fn), N_SIMS_SCAN))
    print(f"  ({_time.time()-t1:.1f}s)")

    print(f"\n{'='*90}\nGRID: post_payout_cooldown(cooldown_days, scale_during)  [{N_SIMS_SCAN} sims]\n{'='*90}")
    t1 = _time.time()
    for cd in (5, 10, 20, 40):
        for sd in (0.0, 0.2, 0.4):
            fn = post_payout_cooldown(cd, sd)
            results.append(report(f"cooldown(days={cd},scale={sd})", mc(fn), N_SIMS_SCAN))
    print(f"  ({_time.time()-t1:.1f}s)")

    print(f"\n{'='*90}\nGRID: combined (best-looking cushion + cooldown pairs)  [{N_SIMS_SCAN} sims]\n{'='*90}")
    t1 = _time.time()
    combo_specs = [
        (0.0, 1.0, 10, 0.0), (0.0, 1.0, 20, 0.0), (0.1, 1.0, 10, 0.2),
        (0.2, 1.0, 10, 0.2), (0.0, 0.5, 10, 0.0), (0.0, 0.5, 20, 0.0),
    ]
    for floor_frac, ramp_frac, cd, sd in combo_specs:
        fn = combined_xfa_scaling(
            cushion_proportional_scaling(XFA.mll_distance, floor_frac, ramp_frac),
            post_payout_cooldown(cd, sd),
        )
        results.append(report(
            f"combo(cushion:{floor_frac}/{ramp_frac} + cooldown:{cd}d@{sd})", mc(fn), N_SIMS_SCAN))
    print(f"  ({_time.time()-t1:.1f}s)")

    print(f"\n{'='*90}\nGRID: hard_stop_after(max_days, scale_after)  -- TENURE-bounded, "
          f"not state-reactive  [{N_SIMS_SCAN} sims]\n{'='*90}")
    t1 = _time.time()
    for max_days in (21, 42, 63, 84, 126, 189):
        for scale_after in (0.0, 0.2):
            fn = hard_stop_after(max_days, scale_after)
            results.append(report(f"hard_stop(days={max_days},after={scale_after})", mc(fn), N_SIMS_SCAN))
    print(f"  ({_time.time()-t1:.1f}s)")

    print(f"\n{'='*90}\nGRID: time_decay_scaling(half_life_days, floor_frac)  -- TENURE-bounded, "
          f"smooth  [{N_SIMS_SCAN} sims]\n{'='*90}")
    t1 = _time.time()
    for hl in (21, 42, 63, 126):
        for ff in (0.0, 0.1, 0.3):
            fn = time_decay_scaling(hl, ff)
            results.append(report(f"time_decay(half_life={hl},floor={ff})", mc(fn), N_SIMS_SCAN))
    print(f"  ({_time.time()-t1:.1f}s)")

    print(f"\n{'='*90}\nPARETO VIEW -- sorted by survival, target >= {SURVIVAL_TARGET:.0%}\n{'='*90}")
    for r in sorted(results, key=lambda r: -r["survive"]):
        flag = "  <-- meets target" if r["survive"] >= SURVIVAL_TARGET else ""
        print(f"  survive={r['survive']:6.1%}  mean_income=${r['mean_income']:>7,.0f}  "
              f"{r['label']:<45}{flag}")

    # ── EDGE-STRENGTH SENSITIVITY ──────────────────────────────────────
    # Sizing changes both drift and variance TOGETHER, which barely
    # moves ruin probability against a fixed $ barrier (classic Kelly /
    # risk-of-ruin behavior -- consistent with every result above
    # clustering near 0%). This instead adds a constant $/day SHIFT to
    # the empirical series -- volatility unchanged, mean (and therefore
    # Sharpe) up -- to answer "how much stronger would the edge itself
    # need to be" as a concrete target for the NEXT strategy search,
    # not a claim that this shift is achievable.
    print(f"\n{'='*90}\nSENSITIVITY -- constant $/day drift added to the SAME series "
          f"(vol unchanged)  [{N_SIMS_SCAN} sims]\n{'='*90}")
    t1 = _time.time()
    daily_arr = np.array([float(v) for v in daily_pnl_list])
    realized_daily_std = float(daily_arr.std())
    print(f"  (realized daily P&L: mean=${daily_arr.mean():.2f}  std=${realized_daily_std:.2f}  "
          f"Sharpe(ann.)={daily_arr.mean()/realized_daily_std*np.sqrt(252):.2f})")
    sens_results = []
    for shift in (0, 10, 25, 50, 75, 100, 150, 200, 300):
        shifted = [Decimal(str(round(float(v) + shift, 2))) for v in daily_arr]
        r = monte_carlo_xfa_economics(shifted, xfa=XFA, horizon_days=XFA_HORIZON_DAYS,
                                       n_sims=N_SIMS_SCAN, block_len=BLOCK_LEN, seed=SEED,
                                       sizing_fn=xfa_full_size)
        implied_sharpe = (daily_arr.mean() + shift) / realized_daily_std * np.sqrt(252)
        print(f"  +${shift:>4}/day (implied ann. Sharpe={implied_sharpe:5.2f})  "
              f"survive={r.prob_survive:6.1%}  mean_income=${r.mean_income:>7,.0f}  "
              f"payouts/yr={r.mean_n_payouts:4.2f}")
        sens_results.append((shift, implied_sharpe, r.prob_survive))
    print(f"  ({_time.time()-t1:.1f}s)")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
