"""Design a strategy specifically for the funded (XFA) phase instead of
reusing a Combine-tuned edge at reduced size (the user's suggestion 1,
after the micro-sizing test showed the current live ensemble's daily
$-volatility is still too high relative to a $2,000 floor even at 1
micro contract, the smallest tradable size).

Combine-tuned strategies (ORB especially: stop_ticks=40, targets scaled
to the day's own range) were built to hit a $3,000 target FAST -- wide
stops, bigger swings. The funded-phase objective rewards the opposite:
minimize daily $ volatility (sigma^2 sits in the denominator of the
ruin exponent 2*mu*D/sigma^2 derived in evaluate_xfa_sizing_overlays.py)
while keeping mu positive. This does NOT go looking for a brand new
signal -- this session already exhaustively searched for one (11+
candidates, all failed genuine OOS/DSR) -- it takes the ALREADY
OOS-relevant MeanReversionBollinger structure (the highest solo Sharpe
in the whole universe screen: ES 0.83) and re-parametrizes it for
tighter per-trade risk: smaller sigma_mult (entries closer to the
mean, so both win and loss magnitudes shrink) and a tighter stop_ticks.
A couple of tightened ORB variants are tested too for completeness.

Discipline: candidate SELECTION uses IS data only, scored by the ruin
exponent itself (2*mu*D/sigma^2) -- the actual funded-phase objective,
not Sharpe -- then the ONE selected candidate per asset touches OOS
exactly once (same one-touch discipline as every other candidate this
session). The winner then goes through the full XFA survival Monte
Carlo at micro-contract size, with both payout-policy readings
(take-max vs. the user's "build to ~$1,500, take ~$500, keep the
cushion" partial-payout idea), to get real EV and max-drawdown numbers.

Run with: python scripts/screen_funded_phase_tight_risk.py
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

from topstep50k.analysis.xfa_economics import monte_carlo_xfa_economics, take_fixed_amount, take_max_payout
from topstep50k.analysis.xfa_sizing import constant_scale
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.evaluation.harness import is_oos_split
from topstep50k.portfolio import PortfolioStrategy
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

# MeanRev: 3 tightness levels, principled (halving-ish steps), not a
# grid search for the best-looking number.
MR_VARIANTS = {
    "baseline":   dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15, time_stop_minutes=45),
    "tight":      dict(qty=1, lookback=60, sigma_mult=1.5, stop_ticks=10, time_stop_minutes=30),
    "very_tight": dict(qty=1, lookback=60, sigma_mult=1.0, stop_ticks=6,  time_stop_minutes=20),
}
# ORB: stop_ticks tightened; tp_multiple held fixed (still adaptive to
# the day's own range -- that's the strategy's whole mechanism).
ORB_VARIANTS = {
    "baseline": dict(qty=1, or_minutes=30, direction="both", stop_mode="opposite_range",
                      stop_ticks=40, tp_multiple=1.0, flat_before_close_minutes=15),
    "tight":    dict(qty=1, or_minutes=30, direction="both", stop_mode="opposite_range",
                      stop_ticks=15, tp_multiple=1.0, flat_before_close_minutes=15),
}

MICRO_COUNTS = [1, 2, 3]
XFA_HORIZON_DAYS = 252
N_SIMS_FINAL = 3000


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


def stats(arr: np.ndarray) -> dict:
    mean, std = arr.mean(), arr.std()
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0.0
    ruin_exp = 2 * mean * float(XFA.mll_distance) / (std ** 2) if std > 0 else -999.0
    return dict(mean=mean, std=std, sharpe=sharpe, ruin_exp=ruin_exp)


def main():
    print("=" * 96)
    print("FUNDED-PHASE STRATEGY DESIGN -- tight-risk variants, selected on the ruin exponent")
    print("=" * 96)
    t0 = _time.time()

    all_bars, gates = {}, {}
    for ak in ASSETS:
        print(f"\nLoading {ak} bars...", flush=True)
        tl = _time.time()
        bars = [b for b in load_bars_csv(ASSETS[ak]["data_path"]) if b.ts >= RECENT_START]
        print(f"  {len(bars):,} bars in {_time.time()-tl:.1f}s", flush=True)
        all_bars[ak] = bars
        pstats = per_day_session_stats(bars)
        gates[ak] = {"orb": orb_expansion_gate(pstats), "mr": meanrev_low_vol_gate(pstats)}

    print("\nBuilding MeanRev x 3 tightness x 3 assets, ORB x 2 tightness x 3 assets...", flush=True)
    t1 = _time.time()
    streams = {}
    for ak in ASSETS:
        ts = ASSETS[ak]["tick_size_f"]
        bars = all_bars[ak]
        g = gates[ak]
        for vname, params in MR_VARIANTS.items():
            streams[("MR", vname, ak)] = run_stream(
                bars, lambda p=params, gg=g, tts=ts, a=ak: MeanReversionBollinger(
                    symbol=a, tick_size=tts, daily_filter=gg["mr"], **p), ak)
        for vname, params in ORB_VARIANTS.items():
            streams[("ORB", vname, ak)] = run_stream(
                bars, lambda p=params, gg=g, tts=ts, a=ak: OpeningRangeBreakout(
                    symbol=a, tick_size=tts, daily_filter=gg["orb"], **p), ak)
    print(f"  built in {_time.time()-t1:.0f}s")

    union_days = sorted(set().union(*[set(s.keys()) for s in streams.values()]))
    arrays = {k: to_series(pnl, union_days) for k, pnl in streams.items()}
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, arrays)
    print(f"\nFull history: {union_days[0]} -> {union_days[-1]} ({len(union_days)} d).  "
          f"IS: {is_days[0]}->{is_days[-1]} ({len(is_days)}d)  OOS: {oos_days[0]}->{oos_days[-1]} ({len(oos_days)}d)")
    print(f"Setup: {_time.time()-t0:.0f}s")

    # ── SELECTION on IS only, by ruin exponent ──────────────────────────
    print(f"\n{'='*96}\nIS SCREEN -- selection metric is the ruin exponent (2*mu*D/sigma^2), not Sharpe\n{'='*96}")
    print(f"{'candidate':<26}{'IS mean/d':>11}{'IS std/d':>11}{'IS Sharpe':>11}{'IS ruin-exp':>13}")
    is_stats = {}
    for k, arr in arrays.items():
        s = stats(arr[is_mask])
        is_stats[k] = s
        label = f"{k[0]}/{k[2]} [{k[1]}]"
        print(f"{label:<26}{s['mean']:>11.1f}{s['std']:>11.1f}{s['sharpe']:>11.2f}{s['ruin_exp']:>13.3f}")

    ranked = sorted(is_stats.items(), key=lambda kv: -kv[1]["ruin_exp"])
    print(f"\nTop 3 by IS ruin-exponent:")
    for k, s in ranked[:3]:
        print(f"  {k[0]}/{k[2]} [{k[1]}]: ruin-exp={s['ruin_exp']:.3f}")

    # ── ONE-TOUCH OOS on the top candidate ──────────────────────────────
    winner_key = ranked[0][0]
    print(f"\n{'='*96}\nOOS CONFIRMATION (one touch) -- {winner_key[0]}/{winner_key[2]} [{winner_key[1]}]\n{'='*96}")
    arr = arrays[winner_key]
    oos_s = stats(arr[oos_mask])
    is_s = is_stats[winner_key]
    print(f"  IS:  mean/d=${is_s['mean']:.1f}  std/d=${is_s['std']:.1f}  Sharpe={is_s['sharpe']:.2f}  "
          f"ruin-exp={is_s['ruin_exp']:.3f}")
    print(f"  OOS: mean/d=${oos_s['mean']:.1f}  std/d=${oos_s['std']:.1f}  Sharpe={oos_s['sharpe']:.2f}  "
          f"ruin-exp={oos_s['ruin_exp']:.3f}")
    full_s = stats(arr)
    print(f"  FULL HISTORY: mean/d=${full_s['mean']:.1f}  std/d=${full_s['std']:.1f}  "
          f"Sharpe={full_s['sharpe']:.2f}  ruin-exp={full_s['ruin_exp']:.3f}")

    # ── XFA SURVIVAL, EV, DRAWDOWN at micro size, both payout policies ──
    print(f"\n{'='*96}\nXFA ECONOMICS on the full-history series, {winner_key[0]}/{winner_key[2]} [{winner_key[1]}]\n{'='*96}")
    for n_micro in MICRO_COUNTS:
        k = n_micro / 10.0
        r_max = monte_carlo_xfa_economics(
            [Decimal(str(round(float(v), 2))) for v in arr], xfa=XFA,
            horizon_days=XFA_HORIZON_DAYS, n_sims=N_SIMS_FINAL, block_len=10, seed=42,
            sizing_fn=constant_scale(k), payout_policy=take_max_payout, preserve_cushion=False)
        r_partial = monte_carlo_xfa_economics(
            [Decimal(str(round(float(v), 2))) for v in arr], xfa=XFA,
            horizon_days=XFA_HORIZON_DAYS, n_sims=N_SIMS_FINAL, block_len=10, seed=42,
            sizing_fn=constant_scale(k), payout_policy=take_fixed_amount(Decimal("500")),
            preserve_cushion=True)
        print(f"\n  {n_micro} micro(s) (k={k}):")
        print(f"    take-max, reset-to-zero:      survive={r_max.prob_survive:6.1%}  "
              f"EV(mean income)=${r_max.mean_income:>7,.0f}  "
              f"DD: mean=${r_max.mean_max_drawdown:>7,.0f} ({r_max.mean_max_drawdown/50000:.1%} of $50k)  "
              f"p95=${r_max.p95_max_drawdown:>7,.0f}")
        print(f"    partial $500, preserve cushion: survive={r_partial.prob_survive:6.1%}  "
              f"EV(mean income)=${r_partial.mean_income:>7,.0f}  "
              f"DD: mean=${r_partial.mean_max_drawdown:>7,.0f} ({r_partial.mean_max_drawdown/50000:.1%} of $50k)  "
              f"p95=${r_partial.p95_max_drawdown:>7,.0f}")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
