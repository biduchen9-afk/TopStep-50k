"""Pass-rate-aware sizing for the gated 3-strategy ensemble.

The previous walk-forward inverse-vol attempt confirmed that minimising
ENSEMBLE VARIANCE does not improve pass-rate; the binding constraint
is per-day MEAN PnL, not stdev. This script targets pass-rate
DIRECTLY:

  For each strategy s and at each time t, compute the strategy's SOLO
  pass-rate over the trailing 252 days ending at t-1 (causal).
  Weight strategy s by its solo pass-rate.

Two variants:

  * STATIC: Solo pass-rates computed once over the entire TRAIN window
    (first 2y). Weights fixed for TEST.
  * WALK-FORWARD (WF): Solo pass-rates recomputed at each day t over
    the trailing 252 days, weights applied at day t.

Discipline (overfit safeguards):

  * One pre-committed hyperparameter: lookback_days = 252. No sweep.
  * One pre-committed choice: weight by SOLO pass-rate (no exponents,
    no power-law). Linear, no free parameters.
  * Use $25K Combine rules for the weight criterion -- the smaller
    target is what makes the pass-rate signal non-trivial on TRAIN
    where the $50K target is unreachable for most strategies.
  * Both rulesets reported on TEST.
  * All three strategies always present (no drop).
  * Fast inlined pass-rate avoids consistency check -- a slight
    approximation but kept for tractability and stability of the
    weight signal.
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
RULES = combine_50k()  # for DSR fractions; weight criterion uses $25K below
RULESETS = {"$50K Combine": combine_50k(), "$25K Combine": combine_25k()}

ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both",
                   stop_mode="opposite_range", stop_ticks=40,
                   tp_multiple=1.0, flat_before_close_minutes=15,
                   tick_size=0.25)
MR_PARAMS = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                  time_stop_minutes=45, tick_size=0.25)
OD_PARAMS = dict(qty=1, entry_offset_minutes=5, exit_offset_minutes=5)

# PRE-COMMITTED parameters
WF_LOOKBACK_DAYS = 252
WF_MIN_BARS_FOR_PR = 60     # need at least 60 days before WF weights kick in
WEIGHT_CRITERION_WINDOW = 30  # 30-day pass-rate as the weight signal


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


def fast_pass_rate(daily_pnl: np.ndarray, target: float,
                    mll_distance: float, window: int = 30) -> float:
    """Approximate pass-rate over rolling windows of `window` days.

    Pass condition (approximation, drops consistency):
      * at SOME point in the window, running cumulative >= target
      * up to that point, max(running cumulative) - running cumulative
        never crosses mll_distance (i.e., no MLL breach)
    """
    n = daily_pnl.size
    if n < window:
        return 0.0
    total = n - window + 1
    passes = 0
    for start in range(total):
        cum = 0.0
        peak = 0.0
        breached = False
        target_hit = False
        for i in range(window):
            cum += daily_pnl[start + i]
            if cum > peak:
                peak = cum
            if cum <= peak - mll_distance:
                breached = True
                break
            if cum >= target:
                target_hit = True
                break
        if target_hit and not breached:
            passes += 1
    return passes / total


def proper_pass_rate(daily_arr, day_index, window, rules):
    """Use the project's realized_pass_rate for the headline TEST report."""
    daily_pnl = {d: Decimal(str(round(float(v), 2)))
                 for d, v in zip(day_index, daily_arr)}
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
    print(f"  {label:>32} : days={daily.size}  nz={nz:>4}  "
          f"total=${daily.sum():>+9,.0f}  Sharpe={sh:>+5.2f}  "
          f"WR={wr:.1%}  PF={pf:>4.2f}  MaxDD=${mdd:>+8,.0f}")
    for rname, rules in RULESETS.items():
        rr30 = proper_pass_rate(daily, day_index, 30, rules)
        rr45 = proper_pass_rate(daily, day_index, 45, rules)
        print(f"  {'':>32}   {rname:<14}  "
              f"pass30={rr30.pass_rate:>5.1%} ({rr30.n_passed}/{rr30.n_windows})  "
              f"pass45={rr45.pass_rate:>5.1%} ({rr45.n_passed}/{rr45.n_windows})")


# ---------- the two sizing methods ----------------------------------------


def static_pass_rate_weights(per_strategy_series: dict[str, np.ndarray],
                              train_mask: np.ndarray,
                              rules) -> dict[str, float]:
    """Compute solo pass-rate of each strategy on the TRAIN window;
    weight proportionally. Falls back to equal weight if all solo pass-
    rates are zero."""
    target = float(rules.profit_target)
    mll = float(rules.max_loss_limit_distance)
    pass_rates = {}
    for name, series in per_strategy_series.items():
        train_arr = series[train_mask]
        pr = fast_pass_rate(train_arr, target=target,
                             mll_distance=mll,
                             window=WEIGHT_CRITERION_WINDOW)
        pass_rates[name] = pr
    total = sum(pass_rates.values())
    if total <= 0:
        n = len(pass_rates)
        return {k: 1.0 / n for k in pass_rates}
    return {k: v / total for k, v in pass_rates.items()}, pass_rates


def walk_forward_pass_rate_weights(per_strategy_series: dict[str, np.ndarray],
                                    rules,
                                    lookback_days: int = WF_LOOKBACK_DAYS,
                                    min_bars: int = WF_MIN_BARS_FOR_PR,
                                    window: int = WEIGHT_CRITERION_WINDOW,
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """At each day t, compute solo pass-rate of each strategy over the
    trailing `lookback_days` ending at t-1; weight by it."""
    target = float(rules.profit_target)
    mll = float(rules.max_loss_limit_distance)
    names = list(per_strategy_series.keys())
    arrays = np.stack([per_strategy_series[n] for n in names], axis=1)
    T, N = arrays.shape
    combined = np.zeros(T, dtype=float)
    weights_history = np.full((T, N), np.nan)
    for t in range(T):
        lo = max(0, t - lookback_days)
        hi = t  # exclusive (causal)
        if hi - lo < min_bars + window:
            w = np.ones(N) / N
        else:
            pass_rates = np.zeros(N)
            for j in range(N):
                col = arrays[lo:hi, j]
                pass_rates[j] = fast_pass_rate(col, target=target,
                                                 mll_distance=mll,
                                                 window=window)
            total = pass_rates.sum()
            if total <= 0:
                w = np.ones(N) / N
            else:
                w = pass_rates / total
        weights_history[t] = w
        combined[t] = float(np.dot(w, arrays[t]))
    return combined, weights_history


# --------------------------- main ----------------------------------------


def main():
    print("Loading ES bars...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(ROOT / "data" / "raw" / "es_cleaned.txt"))
    print(f"  {len(bars):,} bars in {_time.time() - t0:.1f}s", flush=True)

    print("\nBuilding regime conditioners...", flush=True)
    stats = per_day_session_stats(bars)
    g_orb = orb_expansion_gate(stats)
    g_mr = meanrev_low_vol_gate(stats)
    g_od = overnight_drift_post_selloff_gate(stats)

    print("\nRunning GATED strategies once each...", flush=True)
    g_orb_daily = run_strategy(bars,
        lambda: OpeningRangeBreakout(symbol="ES", daily_filter=g_orb,
                                      **ORB_PARAMS), "ORB-gated")
    g_mr_daily = run_strategy(bars,
        lambda: MeanReversionBollinger(symbol="ES", daily_filter=g_mr,
                                        **MR_PARAMS), "MR-gated")
    g_od_daily = run_strategy(bars,
        lambda: OvernightDrift(symbol="ES", entry_filter=g_od,
                                **OD_PARAMS), "OD-gated")

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

    # ---- STATIC pass-rate weights from TRAIN ($25K criterion) ----
    print(f"\n{'='*72}")
    print(f"STATIC pass-rate sizing (TRAIN-derived, $25K criterion)")
    print(f"{'='*72}")
    static_weights, solo_prs = static_pass_rate_weights(
        gated_series, train_mask, combine_25k())
    print(f"  Solo pass-rates on TRAIN (gated, 30-day window, $25K rules):")
    for n, pr in solo_prs.items():
        print(f"    {n:>22}: {pr:.1%}")
    print(f"  -> static weights: {dict((k, round(v, 3)) for k, v in static_weights.items())}")

    def static_ensemble(per_strat: dict[str, np.ndarray]) -> np.ndarray:
        w = static_weights
        out = np.zeros(next(iter(per_strat.values())).size, dtype=float)
        for n, s in per_strat.items():
            out += w[n] * s
        return out

    # ---- WALK-FORWARD pass-rate weights ($25K criterion) ----
    print(f"\nWALK-FORWARD pass-rate weights ({WF_LOOKBACK_DAYS}-day lookback)...",
          flush=True)
    t0 = _time.time()
    wf_combined, wf_hist = walk_forward_pass_rate_weights(
        gated_series, rules=combine_25k(),
        lookback_days=WF_LOOKBACK_DAYS,
        min_bars=WF_MIN_BARS_FOR_PR,
        window=WEIGHT_CRITERION_WINDOW)
    print(f"  computed in {_time.time() - t0:.1f}s", flush=True)

    print(f"\n  WF weights on TEST (mean / range):")
    wf_test = wf_hist[test_mask]
    for j, name in enumerate(gated_series.keys()):
        col = wf_test[:, j]
        valid = col[~np.isnan(col)]
        if valid.size:
            print(f"    {name:>22}: mean={valid.mean():.3f}  "
                  f"min={valid.min():.3f}  max={valid.max():.3f}")

    # ---- baseline (inverse-vol fixed) for comparison ----
    # Reproduce the prior phase-5 fixed-weights gated ensemble
    train_stds = {n: float(gated_series[n][train_mask].std(ddof=1))
                   for n in gated_series}
    inv_vol = {n: (1.0 / s if s > 0 else 0.0)
                for n, s in train_stds.items()}
    z = sum(inv_vol.values())
    invvol_weights = {n: w / z for n, w in inv_vol.items()}

    def invvol_ensemble(per_strat):
        zz = sum(invvol_weights[n] for n in per_strat)
        out = np.zeros(next(iter(per_strat.values())).size, dtype=float)
        for n, s in per_strat.items():
            out += (invvol_weights[n] / zz) * s
        return out

    # ---- reports on TEST ----
    print(f"\n{'='*72}")
    print(f"TEST window comparison")
    print(f"{'='*72}")
    gated_test = {n: s[test_mask] for n, s in gated_series.items()}
    inv_vol_test = invvol_ensemble(gated_test)
    static_test = static_ensemble(gated_test)
    wf_test_combined = wf_combined[test_mask]

    report("inverse-vol fixed (prior baseline)", inv_vol_test, test_days)
    print()
    report("STATIC pass-rate", static_test, test_days)
    print()
    report("WALK-FORWARD pass-rate", wf_test_combined, test_days)

    # ---- drop-one sensitivity for static pass-rate ----
    print(f"\n{'='*72}")
    print(f"Drop-one sensitivity (static pass-rate weights)")
    print(f"{'='*72}")
    for drop in list(gated_series.keys()):
        sub_series = {n: s for n, s in gated_series.items() if n != drop}
        sub_w, sub_prs = static_pass_rate_weights(sub_series, train_mask,
                                                    combine_25k())
        # Inline ensemble using sub_w (already normalised over the subset)
        out = np.zeros(sub_series[next(iter(sub_series))].size, dtype=float)
        for n, s in sub_series.items():
            out += sub_w[n] * s
        report(f"static drop {drop}", out[test_mask], test_days)

    # ---- DSR ----
    print(f"\n{'='*72}")
    print(f"Deflated Sharpe")
    print(f"{'='*72}")
    trials = []
    for arr in gated_test.values():
        trials.append(sharpe_annual(arr))
    trials.append(sharpe_annual(inv_vol_test))
    trials.append(sharpe_annual(static_test))
    trials.append(sharpe_annual(wf_test_combined))
    for drop in list(gated_series.keys()):
        sub_series = {n: s for n, s in gated_series.items() if n != drop}
        sub_w, _ = static_pass_rate_weights(sub_series, train_mask,
                                              combine_25k())
        out = np.zeros(sub_series[next(iter(sub_series))].size, dtype=float)
        for n, s in sub_series.items():
            out += sub_w[n] * s
        trials.append(sharpe_annual(out[test_mask]))
    trials = [s for s in trials if not np.isnan(s)]
    starting = float(RULES.starting_balance)
    for label, arr in [("STATIC pass-rate", static_test),
                        ("WALK-FORWARD pass-rate", wf_test_combined)]:
        try:
            dsr = deflated_sharpe(
                (arr / starting).tolist(),
                all_trial_sharpes_annual=trials,
                periods_per_year=252,
            )
            print(f"  candidate winner: {label}")
            print(f"    raw Sharpe (annual): {dsr.sharpe:.3f}")
            print(f"    expected-max under null: {dsr.expected_max_sharpe:.3f}")
            print(f"    DSR: {dsr.dsr:.4f}")
            print(f"    trials this pass: {len(trials)}")
            v = ("statistically significant" if dsr.dsr >= 0.95
                 else "NOT statistically distinguishable from selection bias")
            print(f"    -> {v}")
        except Exception as e:
            print(f"  DSR for {label} failed: {e}")


if __name__ == "__main__":
    main()
