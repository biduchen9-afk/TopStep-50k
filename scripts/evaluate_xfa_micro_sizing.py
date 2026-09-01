"""Test the user's hypothesis directly: trade MICRO contracts (MES/MNQ/
MGC) instead of minis (ES/NQ/GC) to survive the funded phase, then
pull payouts once a survivable cushion has built up.

Every sizing policy tried in evaluate_xfa_sizing_overlays.py was
REACTIVE -- it cuts size around cushion or a recent payout, then ramps
back to FULL (mini) size once that specific condition clears. None of
them tested a permanently smaller BASE size. Trading micros instead of
minis is exactly that: MES/MNQ/MGC are each precisely 1/10th the
$-per-point of ES/NQ/GC (same tick size in POINTS, 1/10th the
multiplier -- see CME contract specs), and every strategy in this
project sizes stops/targets in TICKS, not dollars, so replaying the
exact same trade-by-trade decisions at micro size is mathematically
EXACT: multiply the existing $ P&L series by a constant k=0.1 (or
k=0.2 for 2 micros per mini-equivalent, etc.) -- not an approximation,
the identical sequence of wins/losses at 1/10th the dollar magnitude.

Why this should work where reactive sizing didn't: the ruin exponent
2*mu*D/sigma^2 (see evaluate_xfa_sizing_overlays.py's sensitivity
sweep) scales as 1/k under a constant multiplier k, since both mu and
sigma scale by k (mu/sigma^2 ~ k/k^2 = 1/k). A 10x size cut (k=0.1) is
worth roughly a 10x improvement in the exponent -- dwarfing anything
a reactive policy achieved (best reactive result: 6.6% survival vs.
~0% baseline).

Tests TWO candidate edges across a k sweep, then checks whether
layering a reactive post-payout policy ON TOP of micro sizing helps
further (the user's "cushion that's survivable through the CONSISTENCY
of pulling payouts" -- i.e. don't just shrink size once, also protect
the account right after each payout when cushion is at its thinnest):

  1. LIVE ensemble  -- the current live_portfolio.py 6-stream
     ORB+MeanRev ensemble (IS_WEIGHTS) -- the actual Combine-validated
     edge, what would really be traded.
  2. MIN-VAR portfolio -- the 3-stream minimum-variance combination
     found by screen_xfa_survival_universe.py (VwapFractal/ES +
     ThreeDayRev/NQ + IntraMom/NQ), for comparison.

Run with: python scripts/evaluate_xfa_micro_sizing.py
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
from topstep50k.analysis.xfa_sizing import combined_xfa_scaling, constant_scale, post_payout_cooldown
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.portfolio.live_portfolio import IS_WEIGHTS
from topstep50k.regime import meanrev_low_vol_gate, orb_expansion_gate, per_day_session_stats
from topstep50k.rules import combine_50k
from topstep50k.rules.topstep_xfa import xfa_50k
from topstep50k.strategy.intraday_momentum import IntradayMomentum
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout
from topstep50k.strategy.three_day_reversal import ThreeDayReversal
from topstep50k.strategy.vwap_std_fractal import VwapStdFractal

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
N_SIMS = 1000
N_SIMS_FINAL = 3000
BLOCK_LEN = 10
SEED = 42
SURVIVAL_TARGET = 0.90
# k values expressed as "number of MICRO contracts" (1 micro = 0.1x one
# mini, exactly, per CME contract specs -- MES/MNQ/MGC are each 1/10th
# the multiplier of ES/NQ/GC at the same tick size in points).
MICRO_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10]  # 10 micros == 1 full mini (k=1.0)


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


def xfa_mc(values: np.ndarray, n_sims: int, sizing_fn=xfa_full_size):
    daily = [Decimal(str(round(float(v), 2))) for v in values]
    return monte_carlo_xfa_economics(daily, xfa=XFA, horizon_days=XFA_HORIZON_DAYS,
                                      n_sims=n_sims, block_len=BLOCK_LEN, seed=SEED,
                                      sizing_fn=sizing_fn)


def sweep(label: str, series: np.ndarray, n_sims: int):
    print(f"\n{'-'*90}\n{label}\n{'-'*90}")
    base_mean, base_std = series.mean(), series.std()
    print(f"  at 1 mini (k=1.0): mean/day=${base_mean:.1f}  std/day=${base_std:.1f}  "
          f"Sharpe(ann)={base_mean/base_std*np.sqrt(252):.2f}")
    print(f"  {'micros':>7}{'k':>7}{'mean/d':>10}{'std/d':>10}{'survive':>10}{'mean_income':>13}{'median':>10}")
    crossing = None
    for n_micro in MICRO_COUNTS:
        k = n_micro / 10.0
        r = xfa_mc(series, n_sims, sizing_fn=constant_scale(k))
        flag = ""
        if r.prob_survive >= SURVIVAL_TARGET and crossing is None:
            crossing = n_micro
            flag = "  <-- crosses 90%"
        print(f"  {n_micro:>7}{k:>7.2f}{base_mean*k:>10.1f}{base_std*k:>10.1f}"
              f"{r.prob_survive:>9.1%} ${r.mean_income:>11,.0f} ${r.median_income:>8,.0f}{flag}")
    return crossing


def main():
    print("=" * 90)
    print("MICRO-CONTRACT SIZING TEST -- does trading smaller solve the survival problem?")
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

    print("\nBuilding streams (live 6-stream ensemble + min-var 3-stream portfolio)...", flush=True)
    t1 = _time.time()
    streams = {}
    for ak in ASSETS:
        ts = ASSETS[ak]["tick_size_f"]
        bars = all_bars[ak]
        g = gates[ak]
        # NOTE: keyed (asset, strategy) here to match IS_WEIGHTS's own
        # key order exactly -- a prior run silently zeroed the whole
        # LIVE ensemble because this was (strategy, asset), which never
        # matched any IS_WEIGHTS key, so `if k in arrays` always failed.
        streams[(ak, "ORB")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: OpeningRangeBreakout(
                symbol=a, tick_size=tts, daily_filter=gg["orb"], **ORB_PARAMS), ak)
        streams[(ak, "MeanRev")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: MeanReversionBollinger(
                symbol=a, tick_size=tts, daily_filter=gg["mr"], **MR_PARAMS), ak)
    streams[("ThreeDayRev", "NQ")] = run_stream(
        all_bars["NQ"], lambda gg=gates["NQ"], tts=ASSETS["NQ"]["tick_size_f"]: ThreeDayReversal(
            symbol="NQ", tick_size=tts, daily_filter=gg["mr"]), "NQ")
    streams[("IntraMom", "NQ")] = run_stream(
        all_bars["NQ"], lambda tts=ASSETS["NQ"]["tick_size_f"]: IntradayMomentum(
            symbol="NQ", tick_size=tts), "NQ")
    streams[("VwapFractal", "ES")] = run_stream(
        all_bars["ES"], lambda tts=ASSETS["ES"]["tick_size_f"]: VwapStdFractal(
            symbol="ES", tick_size=tts), "ES")
    all_bars.clear()
    print(f"  built in {_time.time()-t1:.0f}s")

    union_days = sorted(set().union(*[set(s.keys()) for s in streams.values()]))
    arrays = {k: to_series(pnl, union_days) for k, pnl in streams.items()}
    print(f"\nFull history: {union_days[0]} -> {union_days[-1]} ({len(union_days)} d).  "
          f"Setup: {_time.time()-t0:.0f}s")

    # LIVE ensemble (weighted average, weights sum to 1, per live_portfolio.py)
    z = sum(IS_WEIGHTS.values())
    live = np.zeros(len(union_days))
    for k, w in IS_WEIGHTS.items():
        if k in arrays:
            live += (w / z) * arrays[k]

    # MIN-VAR portfolio (inverse-variance weights, per screen_xfa_survival_universe.py)
    minvar_keys = [("VwapFractal", "ES"), ("ThreeDayRev", "NQ"), ("IntraMom", "NQ")]
    stds = np.array([arrays[k].std() for k in minvar_keys])
    inv_var = 1.0 / stds ** 2
    w = inv_var / inv_var.sum()
    minvar = sum(w[i] * arrays[k] for i, k in enumerate(minvar_keys))

    crossing_live = sweep("LIVE ensemble (6-stream ORB+MeanRev, IS_WEIGHTS)", live, N_SIMS)
    crossing_minvar = sweep("MIN-VAR portfolio (VwapFractal/ES + ThreeDayRev/NQ + IntraMom/NQ)", minvar, N_SIMS)

    # ── Layer a reactive post-payout policy on top of the micro base ───
    print(f"\n{'='*90}\nSTACKING: micro base size + post-payout cooldown, on the LIVE ensemble\n{'='*90}")
    for n_micro in (1, 2, 3):
        k = n_micro / 10.0
        base_r = xfa_mc(live, N_SIMS, sizing_fn=constant_scale(k))
        stacked_fn = combined_xfa_scaling(constant_scale(k), post_payout_cooldown(20, 0.2))
        stacked_r = xfa_mc(live, N_SIMS, sizing_fn=stacked_fn)
        print(f"  {n_micro} micro(s) alone:            survive={base_r.prob_survive:6.1%}  "
              f"mean_income=${base_r.mean_income:>7,.0f}")
        print(f"  {n_micro} micro(s) + payout cooldown: survive={stacked_r.prob_survive:6.1%}  "
              f"mean_income=${stacked_r.mean_income:>7,.0f}")

    # ── Final confirmation at the crossing point, higher n_sims ─────────
    if crossing_live is not None:
        print(f"\n{'='*90}\nFINAL CONFIRMATION -- LIVE ensemble @ {crossing_live} micro(s)  "
              f"[{N_SIMS_FINAL} sims]\n{'='*90}")
        k = crossing_live / 10.0
        r = xfa_mc(live, N_SIMS_FINAL, sizing_fn=constant_scale(k))
        print(f"  survive={r.prob_survive:.1%}  mean_income=${r.mean_income:,.0f}  "
              f"median=${r.median_income:,.0f}  90%CI=[${r.p05_income:,.0f},${r.p95_income:,.0f}]  "
              f"payouts/yr={r.mean_n_payouts:.2f}")
    else:
        print(f"\nLIVE ensemble never crossed {SURVIVAL_TARGET:.0%} survival in the tested range "
              f"(up to 1 full mini).")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
