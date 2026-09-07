"""Walk-forward inverse-vol weighting on the literature-gated ensemble.

Strategies, parameters, and gates are UNCHANGED from
`run_ensemble_regime_gated.py`. The only difference is sizing:

  * Baseline FIXED weights: derived once on the first-2y TRAIN
    window, applied unchanged to all of TEST.
  * Walk-forward WF weights at day t: derived from the last
    `lookback_days` calendar trading days of each strategy's gated
    daily PnL, ending at day t-1. Strictly causal -- no use of day-t
    PnL.

Discipline (overfit safeguards):

  * Single hyperparameter: lookback_days. Pre-commit to 252 (~1y).
    No sweep over lookback values.
  * Weights are pure inverse-vol; we do NOT add a momentum or
    skill term. That would be a new fit.
  * For the first lookback_days of TEST the WF window includes
    train data so the weights are warm from day 1 of TEST. Using
    train daily PnL for SIZING (not for parameter fitting) is fine.
  * All three strategies always included. No drop-the-loser.
  * DSR honest trial count: prior runs accumulated 11; this pass
    adds 2 more variants (walk-forward main + walk-forward
    equal-weight floor sensitivity) = 13.
"""

from __future__ import annotations

import sys
import time as _time
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.dsr import deflated_sharpe
from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import (
    meanrev_low_vol_gate,
    orb_expansion_gate,
    overnight_drift_post_selloff_gate,
    per_day_session_stats,
)
from topstep50k.rules import combine_25k, combine_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout
from topstep50k.strategy.overnight_drift import OvernightDrift


ES = Instrument(symbol="ES", point_value=Decimal("50"),
                tick_size=Decimal("0.25"),
                commission_per_side=Decimal("2.50"))
RULESETS = {"$50K Combine": combine_50k(), "$25K Combine": combine_25k()}
RULES = combine_50k()

ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both",
                   stop_mode="opposite_range", stop_ticks=40,
                   tp_multiple=1.0, flat_before_close_minutes=15,
                   tick_size=0.25)
MR_PARAMS = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                  time_stop_minutes=45, tick_size=0.25)
OD_PARAMS = dict(qty=1, entry_offset_minutes=5, exit_offset_minutes=5)

# PRE-COMMITTED walk-forward parameters
WF_LOOKBACK_DAYS = 252       # ~1 trading year
WF_MIN_BARS = 60             # minimum bars before WF weights are usable


def run_strategy(bars, build_strategy, label):
    strat = build_strategy()
    pf = PortfolioStrategy(components={"ES": strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({"ES": bars}, clk)
    bt = Backtester(rules=RULES, instruments={"ES": ES}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    t0 = _time.time()
    result = bt.run(clk, src)
    print(f"  [{label}] {len(result.daily_pnl)} days, "
          f"{_time.time() - t0:.1f}s", flush=True)
    return result.daily_pnl


def to_series(daily_pnl, day_index):
    out = np.zeros(len(day_index), dtype=float)
    for i, d in enumerate(day_index):
        if d in daily_pnl:
            out[i] = float(daily_pnl[d])
    return out


def sharpe_annual(daily):
    if daily.size < 2 or daily.std(ddof=1) == 0:
        return float("nan")
    return (daily.mean() / daily.std(ddof=1)) * np.sqrt(252)


def max_drawdown(daily):
    if daily.size == 0:
        return 0.0
    eq = np.cumsum(daily)
    peaks = np.maximum.accumulate(eq)
    return float((eq - peaks).min())


def pass_rate(daily, day_index, window, rules):
    daily_pnl = {d: Decimal(str(round(float(v), 2)))
                 for d, v in zip(day_index, daily)}
    return realized_pass_rate(daily_pnl, rules=rules,
                               starting_balance=rules.starting_balance,
                               window_days=window, stride_days=1)


def report(label, daily, day_index):
    if daily.size < 30:
        print(f"  [{label}] insufficient ({daily.size} days)")
        return
    sh = sharpe_annual(daily)
    pf = (daily[daily > 0].sum() / -daily[daily < 0].sum()) \
        if (daily < 0).any() else float("inf")
    wr = float((daily > 0).sum()) / float(daily.size)
    mdd = max_drawdown(daily)
    nz = int((daily != 0).sum())
    print(f"  {label:>26} : days={daily.size}  nz={nz:>4}  "
          f"total=${daily.sum():>+9,.0f}  Sharpe={sh:>+5.2f}  "
          f"WR={wr:.1%}  PF={pf:>4.2f}  MaxDD=${mdd:>+8,.0f}")
    for rname, rules in RULESETS.items():
        rr30 = pass_rate(daily, day_index, 30, rules)
        rr45 = pass_rate(daily, day_index, 45, rules)
        print(f"  {'':>26}   {rname:<14}  "
              f"pass30={rr30.pass_rate:>5.1%} ({rr30.n_passed}/{rr30.n_windows})  "
              f"pass45={rr45.pass_rate:>5.1%} ({rr45.n_passed}/{rr45.n_windows})")


def walk_forward_inv_vol(per_strat_series: dict[str, np.ndarray],
                          lookback_days: int = WF_LOOKBACK_DAYS,
                          min_bars: int = WF_MIN_BARS,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """For each day t, compute inverse-vol weights from the last
    `lookback_days` PnL values (ending at day t-1) for each strategy,
    then form the weighted-sum daily PnL at day t.

    Returns
    -------
    combined : (T,) ndarray  -- daily PnL using rolling weights
    weights_history : (T, n_strats) ndarray -- weights actually used
        on each day. NaN where the window was too short.
    """
    names = list(per_strat_series.keys())
    arrays = np.stack([per_strat_series[n] for n in names], axis=1)  # (T, N)
    T, N = arrays.shape
    combined = np.zeros(T, dtype=float)
    weights_history = np.full((T, N), np.nan)
    for t in range(T):
        lo = max(0, t - lookback_days)
        hi = t  # exclusive -- strictly past
        if hi - lo < min_bars:
            # not enough warm-up; use equal weights
            w = np.ones(N) / N
        else:
            window = arrays[lo:hi]
            stds = window.std(axis=0, ddof=1)
            # Where stdev is zero, fall back to equal share of remaining
            zero_mask = stds <= 0
            inv = np.where(zero_mask, 0.0, 1.0 / np.maximum(stds, 1e-12))
            if inv.sum() <= 0:
                w = np.ones(N) / N
            else:
                w = inv / inv.sum()
        weights_history[t] = w
        combined[t] = float(np.dot(w, arrays[t]))
    return combined, weights_history


def main():
    print("Loading ES bars...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(ROOT / "data" / "raw" / "es_cleaned.txt"))
    print(f"  {len(bars):,} bars in {_time.time() - t0:.1f}s", flush=True)

    print("\nBuilding regime conditioners from per-day session stats...",
          flush=True)
    stats = per_day_session_stats(bars)
    g_orb = orb_expansion_gate(stats)
    g_mr = meanrev_low_vol_gate(stats)
    g_od = overnight_drift_post_selloff_gate(stats)

    print("\nRunning GATED strategies (literature conditioners) once each...",
          flush=True)
    g_orb_daily = run_strategy(bars,
        lambda: OpeningRangeBreakout(symbol="ES", daily_filter=g_orb,
                                      **ORB_PARAMS), "ORB-gated")
    g_mr_daily = run_strategy(bars,
        lambda: MeanReversionBollinger(symbol="ES", daily_filter=g_mr,
                                        **MR_PARAMS), "MR-gated")
    g_od_daily = run_strategy(bars,
        lambda: OvernightDrift(symbol="ES", entry_filter=g_od,
                                **OD_PARAMS), "OD-gated")

    # Also un-gated baseline for direct comparison
    print("\nRunning BASELINE (un-gated) strategies for reference...",
          flush=True)
    b_orb_daily = run_strategy(bars,
        lambda: OpeningRangeBreakout(symbol="ES", **ORB_PARAMS), "ORB-base")
    b_mr_daily = run_strategy(bars,
        lambda: MeanReversionBollinger(symbol="ES", **MR_PARAMS), "MR-base")
    b_od_daily = run_strategy(bars,
        lambda: OvernightDrift(symbol="ES", **OD_PARAMS), "OD-base")

    union_days = sorted(set(g_orb_daily) | set(g_mr_daily) | set(g_od_daily))
    first_day = union_days[0]
    train_end = first_day + timedelta(days=365 * 2)
    train_mask = np.array([d < train_end for d in union_days])
    test_mask = ~train_mask
    train_days = [d for d, m in zip(union_days, train_mask) if m]
    test_days = [d for d, m in zip(union_days, test_mask) if m]
    print(f"\nTRAIN: {train_days[0]} -> {train_days[-1]} ({len(train_days)} d)")
    print(f"TEST : {test_days[0]} -> {test_days[-1]} ({len(test_days)} d)")

    gated_series = {
        "ORB": to_series(g_orb_daily, union_days),
        "MeanRev": to_series(g_mr_daily, union_days),
        "OvernightDrift": to_series(g_od_daily, union_days),
    }
    base_series = {
        "ORB": to_series(b_orb_daily, union_days),
        "MeanRev": to_series(b_mr_daily, union_days),
        "OvernightDrift": to_series(b_od_daily, union_days),
    }

    # 1) FIXED-WEIGHT gated ensemble: baseline TRAIN inverse-vol (from
    #    prior phase 5 result). Reproduce here for direct comparison.
    train_stds = {n: float(base_series[n][train_mask].std(ddof=1))
                   for n in base_series}
    inv_vol = {n: 1.0 / s if s > 0 else 0.0 for n, s in train_stds.items()}
    z = sum(inv_vol.values())
    fixed_weights = {n: w / z for n, w in inv_vol.items()}
    print(f"\nFIXED weights (baseline TRAIN inverse-vol): "
          f"{ {n: round(w, 3) for n, w in fixed_weights.items()} }")

    def fixed_ensemble(per_strat: dict[str, np.ndarray]) -> np.ndarray:
        sub_w = {n: fixed_weights[n] for n in per_strat}
        zz = sum(sub_w.values())
        sub_w = {n: w / zz for n, w in sub_w.items()}
        out = np.zeros(next(iter(per_strat.values())).size, dtype=float)
        for n, s in per_strat.items():
            out += sub_w[n] * s
        return out

    # 2) WALK-FORWARD weights derived on each day from the trailing 252
    #    days of GATED PnL (causal -- ends at day t-1).
    wf_gated, wf_history = walk_forward_inv_vol(gated_series,
                                                  lookback_days=WF_LOOKBACK_DAYS,
                                                  min_bars=WF_MIN_BARS)

    # Inspect: how much do the WF weights move?
    wf_test = wf_history[test_mask]
    print(f"\nWalk-forward weights on TEST window (mean +/- stdev):")
    for j, name in enumerate(gated_series.keys()):
        col = wf_test[:, j]
        col = col[~np.isnan(col)]
        if col.size:
            print(f"  {name:>22} :  mean={col.mean():.3f}  "
                  f"stdev={col.std():.3f}  "
                  f"min={col.min():.3f}  max={col.max():.3f}")

    # 3) Reports on TEST window
    gated_test = {n: s[test_mask] for n, s in gated_series.items()}
    base_test = {n: s[test_mask] for n, s in base_series.items()}

    fixed_baseline_test = fixed_ensemble(base_test)
    fixed_gated_test = fixed_ensemble(gated_test)
    wf_gated_test = wf_gated[test_mask]

    print(f"\n{'='*72}")
    print(f"TEST window: comparison")
    print(f"{'='*72}")
    report("BASELINE fixed", fixed_baseline_test, test_days)
    print()
    report("GATED fixed", fixed_gated_test, test_days)
    print()
    report("GATED walk-forward", wf_gated_test, test_days)

    # 4) Drop-one sensitivity (gated walk-forward), reporting only
    print(f"\n{'='*72}")
    print(f"Walk-forward drop-one sensitivity (gated)")
    print(f"{'='*72}")
    for drop in list(gated_series.keys()):
        sub = {n: s for n, s in gated_series.items() if n != drop}
        wf_sub, _ = walk_forward_inv_vol(sub,
                                          lookback_days=WF_LOOKBACK_DAYS,
                                          min_bars=WF_MIN_BARS)
        report(f"WF gated drop {drop}", wf_sub[test_mask], test_days)

    # 5) DSR honest count
    print(f"\n{'='*72}")
    print(f"Deflated Sharpe")
    print(f"{'='*72}")
    trials = []
    for arr in base_test.values():
        trials.append(sharpe_annual(arr))
    for arr in gated_test.values():
        trials.append(sharpe_annual(arr))
    trials.append(sharpe_annual(fixed_baseline_test))
    trials.append(sharpe_annual(fixed_gated_test))
    trials.append(sharpe_annual(wf_gated_test))
    for drop in list(gated_series.keys()):
        sub = {n: s for n, s in gated_series.items() if n != drop}
        wf_sub, _ = walk_forward_inv_vol(sub,
                                          lookback_days=WF_LOOKBACK_DAYS,
                                          min_bars=WF_MIN_BARS)
        trials.append(sharpe_annual(wf_sub[test_mask]))
    trials = [s for s in trials if not np.isnan(s)]
    starting = float(RULES.starting_balance)
    try:
        dsr = deflated_sharpe(
            (wf_gated_test / starting).tolist(),
            all_trial_sharpes_annual=trials,
            periods_per_year=252,
        )
        print(f"  winner candidate: GATED walk-forward")
        print(f"    raw Sharpe (annual): {dsr.sharpe:.3f}")
        print(f"    expected-max under null: {dsr.expected_max_sharpe:.3f}")
        print(f"    DSR: {dsr.dsr:.4f}")
        print(f"    trials accumulated this pass: {len(trials)}")
        v = ("statistically significant" if dsr.dsr >= 0.95
             else "NOT statistically distinguishable from selection bias")
        print(f"    -> {v}")
    except Exception as e:
        print(f"  DSR failed: {e}")


if __name__ == "__main__":
    main()
