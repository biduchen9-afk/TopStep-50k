"""ONE-TOUCH OOS EVALUATION: 9-stream regime-gated multi-asset ensemble.

THIS SCRIPT IS RUN ONCE AND ONLY ONCE.
OOS period: 2025-01-08 → 2026-04-24 (~405 trading days)
IS period:  2021-12-31 → 2025-01-07 (~943 trading days, 70%)

Gate 4 checks (applied to combined ensemble):
  HARD: OOS EV > 0
  HARD: OOS Sharpe ≥ 0.5 × IS Sharpe  (IS Sharpe = +0.73 → need ≥ 0.365)
  ADVISORY: OOS pass30 ≥ 0.5 × IS pass30  (IS pass30 = 15.6% → need ≥ 7.8%)
  ADVISORY: OOS MLL rate ≤ 1.5 × IS MLL rate

Weighting: pass-rate-aware weights derived from IS period only.
All strategies, parameters, and gates are unchanged from Phase 1.
"""

from __future__ import annotations

import math
import sys
import time as _time
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.passrate import realized_pass_rate, simulate_combine_window
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
ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both",
                   stop_mode="opposite_range", stop_ticks=40,
                   tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS  = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                   time_stop_minutes=45)
OD_PARAMS  = dict(qty=1, entry_offset_minutes=5, exit_offset_minutes=5)

# IS reference values (from evaluate_ensemble_is.py — pre-committed, not re-derived)
IS_SHARPE   = 0.73
IS_PASS30   = 0.156
IS_DAYS_N   = 943


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


def pass_rate_30_45(arr: np.ndarray, days: list[date]):
    pnl = {d: Decimal(str(round(float(v), 2))) for d, v in zip(days, arr)}
    rr30 = realized_pass_rate(pnl, rules=RULES,
                               starting_balance=RULES.starting_balance,
                               window_days=30, stride_days=1)
    rr45 = realized_pass_rate(pnl, rules=RULES,
                               starting_balance=RULES.starting_balance,
                               window_days=45, stride_days=1)
    return rr30, rr45


def mll_breach_rate(arr: np.ndarray, days: list[date]) -> float:
    if arr.size < 30:
        return float("nan")
    total = arr.size - 30 + 1
    cnt = Counter()
    for s in range(total):
        pnls = [(days[s + i], Decimal(str(round(float(arr[s + i]), 2))))
                for i in range(30)]
        r = simulate_combine_window(pnls, rules=RULES,
                                     starting_balance=RULES.starting_balance)
        cnt[r.outcome] += 1
    return cnt["mll_breach"] / total if total > 0 else 0.0


def gate4_check(name: str, observed, threshold, hard: bool, note: str = "") -> bool:
    passed = observed > threshold if hard else observed >= threshold
    tick = "✓" if passed else ("✗" if hard else "~")
    kind = "HARD" if hard else "adv."
    obs_s = f"{observed:.4f}" if isinstance(observed, float) else str(observed)
    thr_s = f"{threshold:.4f}" if isinstance(threshold, float) else str(threshold)
    print(f"  [{tick}] ({kind}) {name:<35} obs={obs_s:<10} thr={thr_s}  {note}")
    return passed


def main():
    print("=" * 72)
    print("ONE-TOUCH OOS EVALUATION: 9-STREAM REGIME-GATED ENSEMBLE")
    print("Gate 4 — OOS validity (final verdict, run ONCE only)")
    print("=" * 72)
    print(f"\nIS reference: Sharpe={IS_SHARPE:.2f}  pass30={IS_PASS30:.1%}  n={IS_DAYS_N}d")
    print(f"Gate 4 thresholds:")
    print(f"  HARD: OOS EV > 0")
    print(f"  HARD: OOS Sharpe ≥ {0.5 * IS_SHARPE:.3f}  (0.5 × IS Sharpe {IS_SHARPE:.2f})")
    print(f"  adv.: OOS pass30 ≥ {0.5 * IS_PASS30:.1%}  (0.5 × IS pass30 {IS_PASS30:.1%})")
    print(f"  adv.: OOS MLL rate ≤ 1.5× IS MLL rate")

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
    print("\nRunning 9 streams...", flush=True)
    streams: dict[tuple[str, str], dict] = {}
    for ak in ASSETS:
        ts = ASSETS[ak]["tick_size_f"]
        bars = all_bars[ak]
        g = gates[ak]
        t0 = _time.time()
        streams[(ak, "ORB")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: OpeningRangeBreakout(
                symbol=a, tick_size=tts, daily_filter=gg["orb"], **ORB_PARAMS), ak)
        streams[(ak, "MeanRev")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: MeanReversionBollinger(
                symbol=a, tick_size=tts, daily_filter=gg["mr"], **MR_PARAMS), ak)
        streams[(ak, "OD")] = run_stream(
            bars, lambda gg=g, a=ak: OvernightDrift(
                symbol=a, entry_filter=gg["od"], **OD_PARAMS), ak)
        print(f"  {ak}: 3 streams in {_time.time() - t0:.1f}s", flush=True)

    # ── 70/30 split ────────────────────────────────────────────────────────
    union_days = sorted(set().union(*[set(s.keys()) for s in streams.values()]))
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
    print(f"\nIS : {is_days[0]} -> {is_days[-1]} ({len(is_days)} d)")
    print(f"OOS: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d)  ← EVALUATING NOW")

    full_arrays: dict[tuple[str, str], np.ndarray] = {
        k: to_series(pnl, union_days) for k, pnl in streams.items()
    }
    is_arrays  = {k: a[is_mask]  for k, a in full_arrays.items()}
    oos_arrays = {k: a[oos_mask] for k, a in full_arrays.items()}

    # ── Derive IS pass-rate-aware weights (IS ONLY) ────────────────────────
    print(f"\n{'─'*72}")
    print("IS PASS-RATE-AWARE WEIGHTS (IS only, used for OOS combination)")
    print(f"{'─'*72}")
    is_pr30 = {}
    for k, arr in is_arrays.items():
        pnl = {d: Decimal(str(round(float(v), 2))) for d, v in zip(is_days, arr)}
        rr = realized_pass_rate(pnl, rules=RULES,
                                 starting_balance=RULES.starting_balance,
                                 window_days=30, stride_days=1)
        is_pr30[k] = rr.pass_rate

    total_pr = sum(is_pr30.values())
    weights = ({k: v / total_pr for k, v in is_pr30.items()}
               if total_pr > 0 else {k: 1.0 / len(is_pr30) for k in is_pr30})
    for k, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {k[0]}/{k[1]:<14}: IS-pass30={is_pr30[k]:.1%}  weight={w:.3f}")

    # ── Build weighted ensemble IS and OOS ────────────────────────────────
    def ensemble(arrays_dict: dict, wt: dict) -> np.ndarray:
        z = sum(wt.values())
        n = next(iter(arrays_dict.values())).size
        out = np.zeros(n, dtype=float)
        for k, w in wt.items():
            out += (w / z) * arrays_dict[k]
        return out

    ens_is  = ensemble(is_arrays,  weights)
    ens_oos = ensemble(oos_arrays, weights)

    # ── IS summary (for reference) ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("IS SUMMARY (reference, not re-derived from this run)")
    print(f"{'─'*72}")
    is_sh = sharpe(ens_is)
    is_ev = float(ens_is.mean())
    is_pf = (ens_is[ens_is > 0].sum() / -ens_is[ens_is < 0].sum()
              if (ens_is < 0).any() else float("inf"))
    is_eq = np.cumsum(ens_is)
    is_mdd = float((is_eq - np.maximum.accumulate(is_eq)).min())
    rr30_is, rr45_is = pass_rate_30_45(ens_is, is_days)
    print(f"  IS total=${ens_is.sum():>+10,.0f}  Sharpe={is_sh:>+5.2f}  "
          f"EV=${is_ev:>+7.2f}/d  PF={is_pf:.2f}  MaxDD=${is_mdd:>+8,.0f}")
    print(f"  IS pass30={rr30_is.pass_rate:.1%} ({rr30_is.n_passed}/{rr30_is.n_windows})  "
          f"pass45={rr45_is.pass_rate:.1%} ({rr45_is.n_passed}/{rr45_is.n_windows})")

    # ── OOS results ───────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("OOS RESULTS")
    print(f"{'─'*72}")
    oos_sh  = sharpe(ens_oos)
    oos_ev  = float(ens_oos.mean())
    oos_pf  = (ens_oos[ens_oos > 0].sum() / -ens_oos[ens_oos < 0].sum()
               if (ens_oos < 0).any() else float("inf"))
    oos_eq  = np.cumsum(ens_oos)
    oos_mdd = float((oos_eq - np.maximum.accumulate(oos_eq)).min())
    rr30_oos, rr45_oos = pass_rate_30_45(ens_oos, oos_days)

    print(f"  OOS total=${ens_oos.sum():>+10,.0f}  Sharpe={oos_sh:>+5.2f}  "
          f"EV=${oos_ev:>+7.2f}/d  PF={oos_pf:.2f}  MaxDD=${oos_mdd:>+8,.0f}")
    print(f"  OOS pass30={rr30_oos.pass_rate:.1%} ({rr30_oos.n_passed}/{rr30_oos.n_windows})  "
          f"pass45={rr45_oos.pass_rate:.1%} ({rr45_oos.n_passed}/{rr45_oos.n_windows})")

    # Per-stream OOS breakdown
    print(f"\n  Per-stream OOS breakdown:")
    for k in sorted(oos_arrays, key=lambda x: -is_pr30[x]):
        arr = oos_arrays[k]
        sh_k = sharpe(arr)
        pnl_k = {d: Decimal(str(round(float(v), 2))) for d, v in zip(oos_days, arr)}
        rr_k = realized_pass_rate(pnl_k, rules=RULES,
                                   starting_balance=RULES.starting_balance,
                                   window_days=30, stride_days=1)
        print(f"    {k[0]}/{k[1]:<14}: total=${arr.sum():>+9,.0f}  "
              f"Sharpe={sh_k:>+5.2f}  OOS-pass30={rr_k.pass_rate:.1%}  "
              f"w={weights[k]:.3f}")

    # ── MLL breach rates ──────────────────────────────────────────────────
    print(f"\nComputing MLL breach rates...", flush=True)
    t0 = _time.time()
    is_mll  = mll_breach_rate(ens_is,  is_days)
    oos_mll = mll_breach_rate(ens_oos, oos_days)
    print(f"  IS MLL breach rate  = {is_mll:.1%}")
    print(f"  OOS MLL breach rate = {oos_mll:.1%}  ({_time.time() - t0:.1f}s)")

    # ── Gate 4 verdict ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("GATE 4 — OOS VALIDITY")
    print(f"{'='*72}")
    sh_ratio = (oos_sh / IS_SHARPE) if (not math.isnan(IS_SHARPE) and IS_SHARPE > 0) else 0.0
    pr_ratio = (rr30_oos.pass_rate / IS_PASS30) if IS_PASS30 > 0 else 0.0
    mll_ok   = (math.isnan(oos_mll) or math.isnan(is_mll)
                or oos_mll <= is_mll * 1.5)

    g4_hard = [
        gate4_check("OOS EV > 0 ($)",             oos_ev,    0.0,        hard=True),
        gate4_check("OOS Sharpe ≥ 0.5×IS",         sh_ratio,  0.50,       hard=True,
                    note=f"OOS={oos_sh:.3f}  IS={IS_SHARPE:.3f}"),
    ]
    g4_adv = [
        gate4_check("OOS pass30 ≥ 0.5×IS pass30", pr_ratio,  0.50,       hard=False,
                    note=f"OOS={rr30_oos.pass_rate:.1%}  IS={IS_PASS30:.1%}"),
        gate4_check("OOS MLL rate ≤ 1.5×IS",       mll_ok,    True,       hard=False,
                    note=f"IS={is_mll:.1%}  OOS={oos_mll:.1%}"),
    ]

    hard_passed = all(g4_hard)
    adv_passed  = sum(g4_adv)

    print(f"\n{'─'*72}")
    if hard_passed:
        promoted = "OOS_PROMOTED"
        verdict  = "ENSEMBLE PASSES Gate 4 → OOS_PROMOTED"
    else:
        promoted = "RETIRED"
        verdict  = "ENSEMBLE FAILS Gate 4 → RETIRED"

    print(f"  OUTCOME: >>> {promoted} <<<")
    print(f"  {verdict}")
    print(f"  Advisory gates passed: {adv_passed}/2")
    print(f"{'─'*72}")

    print(f"\n{'='*72}")
    print("FULL CAMPAIGN SUMMARY")
    print(f"{'='*72}")
    print(f"  IS  period: {is_days[0]}  →  {is_days[-1]}  ({len(is_days)} d)")
    print(f"  OOS period: {oos_days[0]} →  {oos_days[-1]} ({len(oos_days)} d)")
    print(f"  IS  pass30={rr30_is.pass_rate:.1%}  pass45={rr45_is.pass_rate:.1%}  Sharpe={is_sh:+.2f}")
    print(f"  OOS pass30={rr30_oos.pass_rate:.1%}  pass45={rr45_oos.pass_rate:.1%}  Sharpe={oos_sh:+.2f}")
    print(f"  OOS/IS Sharpe ratio = {sh_ratio:.2f}  (≥0.50 required)")
    print(f"  OOS/IS pass30 ratio = {pr_ratio:.2f}  (≥0.50 advisory)")
    print(f"  IS MLL rate={is_mll:.1%}  OOS MLL rate={oos_mll:.1%}")
    print(f"\n  FINAL VERDICT: {promoted}")
    print(f"  DSR trial count: ~40 trials used in Phase A + Phase B search")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
