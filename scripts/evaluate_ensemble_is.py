"""Evaluate the 9-stream regime-gated multi-asset ensemble on 70/30 IS/OOS.

This script uses the SAME strategies, parameters, and gates as
run_ensemble_multi_asset.py but evaluates on the 70/30 IS/OOS split
used in Phase A (harness.py) for consistency.

IS (70%) = first 70% of all trading days (~2022-01 to 2025-01)
OOS (30%) = last 30% (~2025-01 to 2026-04)  ← NOT touched here

This is IS evaluation only. OOS is reserved for one-touch final verdict.
Pass-rate-aware weights derived on IS only (same methodology as existing).

No new strategies, no new trials — same 9 streams already in the ensemble.
DSR trial count not incremented (IS re-evaluation, not a new strategy trial).
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.evaluation.harness import is_oos_split
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import (
    meanrev_low_vol_gate,
    orb_expansion_gate,
    overnight_drift_post_selloff_gate,
    per_day_session_stats,
)
from topstep50k.rules import combine_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout
from topstep50k.strategy.overnight_drift import OvernightDrift


ASSETS = {
    "ES": {
        "instrument": Instrument(symbol="ES", point_value=Decimal("50"),
                                  tick_size=Decimal("0.25"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.25,
        "data_path": ROOT / "data" / "raw" / "es_cleaned.txt",
    },
    "NQ": {
        "instrument": Instrument(symbol="NQ", point_value=Decimal("20"),
                                  tick_size=Decimal("0.25"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.25,
        "data_path": ROOT / "data" / "raw" / "nq_cleaned.txt",
    },
    "GC": {
        "instrument": Instrument(symbol="GC", point_value=Decimal("100"),
                                  tick_size=Decimal("0.10"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.10,
        "data_path": ROOT / "data" / "raw" / "gc_cleaned.txt",
    },
}
RULES = combine_50k()
ORB_PARAMS  = dict(qty=1, or_minutes=30, direction="both",
                    stop_mode="opposite_range", stop_ticks=40,
                    tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS   = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                    time_stop_minutes=45)
OD_PARAMS   = dict(qty=1, entry_offset_minutes=5, exit_offset_minutes=5)


def run_stream(bars, build_fn, asset_key) -> dict[date, Decimal]:
    asset = ASSETS[asset_key]
    strat = build_fn()
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: asset["instrument"]},
                    strategy=pf, audit=InMemoryAuditLog(),
                    combine_enforcement=False)
    return bt.run(clk, src).daily_pnl


def to_series(daily_pnl: dict, day_index: list[date]) -> np.ndarray:
    out = np.zeros(len(day_index), dtype=float)
    for i, d in enumerate(day_index):
        if d in daily_pnl:
            out[i] = float(daily_pnl[d])
    return out


def sharpe(arr: np.ndarray) -> float:
    if arr.size < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    return float((arr.mean() / arr.std(ddof=1)) * np.sqrt(252))


def pass_rate(arr: np.ndarray, days: list[date], window: int) -> tuple[float, int, int]:
    pnl = {d: Decimal(str(round(float(v), 2))) for d, v in zip(days, arr)}
    rr = realized_pass_rate(pnl, rules=RULES, starting_balance=RULES.starting_balance,
                             window_days=window, stride_days=1)
    return rr.pass_rate, rr.n_passed, rr.n_windows


def report_is(label: str, arr: np.ndarray, days: list[date]) -> None:
    nz = int((arr != 0).sum())
    sh = sharpe(arr)
    pf_v = (arr[arr > 0].sum() / -arr[arr < 0].sum()
             if (arr < 0).any() else float("inf"))
    eq = np.cumsum(arr)
    mdd = float((eq - np.maximum.accumulate(eq)).min())
    pr30, n30, w30 = pass_rate(arr, days, 30)
    pr45, n45, w45 = pass_rate(arr, days, 45)
    print(f"  {label:<35}: nz={nz:>4} total=${arr.sum():>+9,.0f} "
          f"Sharpe={sh:>+5.2f} PF={pf_v:>4.2f} MaxDD=${mdd:>+8,.0f}")
    print(f"  {'':>35}  IS pass30={pr30:.1%} ({n30}/{w30})  "
          f"pass45={pr45:.1%} ({n45}/{w45})")


def main():
    print("=" * 72)
    print("9-STREAM ENSEMBLE — IS EVALUATION (70/30 SPLIT, IS ONLY)")
    print("Strategies: ORB + MeanRev + OvernightDrift × ES + NQ + GC")
    print("All with literature-grounded regime gates (pre-committed)")
    print("=" * 72)

    # ── Load bars and build gates ──────────────────────────────────────────
    all_bars = {}
    gates = {}
    for ak in ASSETS:
        print(f"\nLoading {ak} bars...", flush=True)
        t0 = _time.time()
        bars = list(load_bars_csv(ASSETS[ak]["data_path"]))
        print(f"  {len(bars):,} bars in {_time.time() - t0:.1f}s", flush=True)
        all_bars[ak] = bars
        stats = per_day_session_stats(bars)
        gates[ak] = {
            "orb": orb_expansion_gate(stats),
            "mr":  meanrev_low_vol_gate(stats),
            "od":  overnight_drift_post_selloff_gate(stats),
        }

    # ── Run 9 streams ──────────────────────────────────────────────────────
    print("\nRunning 9 strategy-asset streams...", flush=True)
    streams: dict[tuple[str, str], dict] = {}
    for ak in ASSETS:
        ts = ASSETS[ak]["tick_size_f"]
        bars = all_bars[ak]
        g = gates[ak]
        t0 = _time.time()

        streams[(ak, "ORB")] = run_stream(
            bars,
            lambda gg=g, tts=ts, a=ak: OpeningRangeBreakout(
                symbol=a, tick_size=tts, daily_filter=gg["orb"], **ORB_PARAMS),
            ak)

        streams[(ak, "MeanRev")] = run_stream(
            bars,
            lambda gg=g, tts=ts, a=ak: MeanReversionBollinger(
                symbol=a, tick_size=tts, daily_filter=gg["mr"], **MR_PARAMS),
            ak)

        streams[(ak, "OD")] = run_stream(
            bars,
            lambda gg=g, a=ak: OvernightDrift(
                symbol=a, entry_filter=gg["od"], **OD_PARAMS),
            ak)
        print(f"  {ak}: 3 streams in {_time.time() - t0:.1f}s", flush=True)

    # ── 70/30 IS/OOS split ─────────────────────────────────────────────────
    union_days = sorted(set().union(*[set(s.keys()) for s in streams.values()]))
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
    print(f"\nIS : {is_days[0]} -> {is_days[-1]} ({len(is_days)} d)")
    print(f"OOS: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d)  ← NOT evaluated")

    # Build aligned arrays (IS only — OOS not used)
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for k, pnl in streams.items():
        full = to_series(pnl, union_days)
        arrays[k] = full[is_mask]

    # ── Pass-rate-aware weights from IS ────────────────────────────────────
    print(f"\n{'─'*72}")
    print("PER-STREAM IS STATS (for weight derivation)")
    print(f"{'─'*72}")
    is_pr30 = {}
    for k, arr in arrays.items():
        pr30, n30, w30 = pass_rate(arr, is_days, 30)
        is_pr30[k] = pr30
        nz = int((arr != 0).sum())
        sh = sharpe(arr)
        print(f"  {k[0]}/{k[1]:<14}: nz={nz:>3} IS-total=${arr.sum():>+9,.0f} "
              f"Sharpe={sh:>+5.2f}  IS-pass30={pr30:.1%} ({n30}/{w30})")

    total_pr = sum(is_pr30.values())
    if total_pr > 0:
        weights = {k: v / total_pr for k, v in is_pr30.items()}
    else:
        weights = {k: 1.0 / len(is_pr30) for k in is_pr30}
    print(f"\nPass-rate-aware IS weights:")
    for k, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {k[0]}/{k[1]:<14}: {w:.3f}")

    # ── Ensemble IS stats ──────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("ENSEMBLE IS STATS")
    print(f"{'─'*72}")

    def ensemble_arr(wt: dict) -> np.ndarray:
        z = sum(wt.values())
        out = np.zeros(len(is_days), dtype=float)
        for k, w in wt.items():
            if k in arrays:
                out += (w / z) * arrays[k]
        return out

    ens_pr = ensemble_arr(weights)
    ens_eq = ensemble_arr({k: 1.0 for k in arrays})

    report_is("9-stream pass-rate-weighted", ens_pr, is_days)
    print()
    report_is("9-stream equal-weighted",     ens_eq, is_days)

    # Per-asset sub-ensembles
    print(f"\n{'─'*72}")
    print("PER-ASSET 3-STREAM SUB-ENSEMBLES (IS)")
    print(f"{'─'*72}")
    for ak in ASSETS:
        sub_w = {k: weights[k] for k in weights if k[0] == ak}
        sub_arr = ensemble_arr(sub_w)
        report_is(f"{ak} 3-strategy ensemble", sub_arr, is_days)

    print(f"\n{'='*72}")
    print("CONCLUSION")
    print(f"{'='*72}")
    pr30, n30, w30 = pass_rate(ens_pr, is_days, 30)
    pr45, n45, w45 = pass_rate(ens_pr, is_days, 45)
    print(f"  IS pass30 = {pr30:.1%}  ({n30}/{w30} windows)")
    print(f"  IS pass45 = {pr45:.1%}  ({n45}/{w45} windows)")
    print(f"\n  OOS NOT EVALUATED — reserved for final go/no-go decision.")
    print(f"  Existing baseline (old 2y split, corrected passrate): pass30=33.3%")
    print(f"  This run uses 70/30 IS split (IS ends {is_days[-1]}).")


if __name__ == "__main__":
    main()
