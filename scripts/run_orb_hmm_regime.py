"""ORB on ES, gated by HMM regime detected from daily features.

Walk-forward layout (~4 years of ES, 2021-12 -> 2026-04):

  TRAIN window: first 24 calendar months. Fit HMM(n_states=3) on the
    daily feature matrix (log_return, abs_return, range_pct, vol_5,
    vol_20). Strict train-only fit; the forward filter sees no test
    bars at fit time.

  SELECT window: months 25-36. Replay ORB without filtering to get a
    per-day PnL; bucket by prior-EOD HMM regime; pick the regime(s)
    with positive mean PnL as the "tradable" set.

  TEST window: months 37-end (~12-16 months). Re-run ORB with a daily
    filter that only allows trading when the prior-EOD regime is in
    the tradable set. Report realised 30/45-day pass-rate ONLY on this
    window -- that is the OOS number the user cares about.

No look-ahead anywhere:
  * HMM fit consumes only TRAIN features.
  * Forward filter is online (Rabiner alpha-recurrence) so the regime
    at end of day t depends only on feature_1..feature_t.
  * The "tradable regime" set is fixed at the end of SELECT and is
    used unchanged on TEST.
  * Daily filter for day d uses regime computed at end of day d-1 (the
    PRIOR EOD), applied at d's session open.
"""

from __future__ import annotations

import sys
import time as _time
from collections import defaultdict
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.analysis.stats import performance
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import (
    HMMRegimeFilter,
    daily_features_from_bars,
    fit_hmm,
)
from topstep50k.rules import combine_50k
from topstep50k.strategy.orb import OpeningRangeBreakout


ES = Instrument(symbol="ES", point_value=Decimal("50"),
                tick_size=Decimal("0.25"),
                commission_per_side=Decimal("2.50"))
RULES = combine_50k()
# Default ORB baseline (highest pass-rate from prior single-ES runs)
DEFAULT_OR_MIN = 30
DEFAULT_TP = 1.0
DEFAULT_DIR = "both"
DEFAULT_STOP_MODE = "opposite_range"
DEFAULT_STOP_TICKS = 40


def run_orb(bars, *, daily_filter=None, or_min=DEFAULT_OR_MIN, tp=DEFAULT_TP,
            direction=DEFAULT_DIR, stop_mode=DEFAULT_STOP_MODE,
            stop_ticks=DEFAULT_STOP_TICKS, qty=1):
    strat = OpeningRangeBreakout(
        symbol="ES", qty=qty, or_minutes=or_min, direction=direction,
        stop_mode=stop_mode, stop_ticks=stop_ticks, tp_multiple=tp,
        flat_before_close_minutes=15, tick_size=0.25,
        daily_filter=daily_filter,
    )
    pf = PortfolioStrategy(components={"ES": strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({"ES": bars}, clk)
    bt = Backtester(rules=RULES, instruments={"ES": ES}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    return bt.run(clk, src)


def main():
    print("Loading ES bars...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(ROOT / "data" / "raw" / "es_cleaned.txt"))
    print(f"  {len(bars):,} bars in {_time.time()-t0:.1f}s", flush=True)

    # ----- 1) compute daily features once on the full series ----------
    feats = daily_features_from_bars(bars)
    print(f"\nDaily features: {len(feats)} trading days, "
          f"{feats.dates[0]} -> {feats.dates[-1]}", flush=True)

    # ----- 2) define walk-forward windows -----------------------------
    first_day = feats.dates[0]
    train_end_day = first_day + timedelta(days=365 * 2)
    select_end_day = first_day + timedelta(days=365 * 3)
    train_idx = [i for i, d in enumerate(feats.dates) if d < train_end_day]
    select_idx = [i for i, d in enumerate(feats.dates)
                  if train_end_day <= d < select_end_day]
    test_idx = [i for i, d in enumerate(feats.dates) if d >= select_end_day]
    print(f"\nWalk-forward windows:")
    print(f"  TRAIN: {feats.dates[train_idx[0]]} -> {feats.dates[train_idx[-1]]} "
          f"({len(train_idx)} days)")
    print(f"  SELECT: {feats.dates[select_idx[0]]} -> {feats.dates[select_idx[-1]]} "
          f"({len(select_idx)} days)")
    print(f"  TEST: {feats.dates[test_idx[0]]} -> {feats.dates[test_idx[-1]]} "
          f"({len(test_idx)} days)")

    # ----- 3) fit HMM on TRAIN only ----------------------------------
    n_states = 3
    print(f"\nFitting HMM(n_states={n_states}) on TRAIN features...", flush=True)
    t0 = _time.time()
    fitted = fit_hmm(feats.matrix[train_idx], n_states=n_states, seed=0)
    print(f"  fit in {_time.time()-t0:.1f}s, converged={fitted.converged}, "
          f"train log-likelihood={fitted.train_log_likelihood:.1f}")
    # Inspect the learned regimes by their means
    for j in range(n_states):
        means = fitted.means[j]
        print(f"  state {j}: log_r={means[0]:+.4f} abs_r={means[1]:.4f} "
              f"range={means[2]:.4f} vol5={means[3]:.4f} vol20={means[4]:.4f}")

    # ----- 4) replay forward filter across the FULL series -----------
    flt = HMMRegimeFilter(fitted, symbol_tag="ES")
    regime_by_day: dict[date, int] = {}
    for i, d in enumerate(feats.dates):
        ts = feats.ts_eod_utc[i]
        asn = flt.update(ts, feats.matrix[i])
        regime_by_day[d] = asn.most_likely_state
    counts = {j: sum(1 for v in regime_by_day.values() if v == j)
              for j in range(n_states)}
    print(f"\nRegime distribution across all {len(regime_by_day)} days: {counts}")

    # ----- 5) baseline ORB without filter (full history) -------------
    print(f"\nRunning baseline ORB (no filter) for regime PnL bucketing...", flush=True)
    t0 = _time.time()
    baseline = run_orb(bars)
    print(f"  done in {_time.time()-t0:.1f}s, {len(baseline.daily_pnl)} trading days")

    # ----- 6) bucket SELECT daily PnL by PRIOR-day regime -----------
    # The decision on day d uses regime as of EOD d-1, so bucket day d's PnL
    # by regime_by_day[d-1].
    select_dates = set(feats.dates[i] for i in select_idx)
    test_dates = set(feats.dates[i] for i in test_idx)
    select_pnl_by_regime: dict[int, list[float]] = defaultdict(list)
    for d, pnl in sorted(baseline.daily_pnl.items()):
        prior = d - timedelta(days=1)
        # Walk back to find the most recent regime entry (skip weekends)
        steps = 0
        while prior not in regime_by_day and steps < 7:
            prior -= timedelta(days=1)
            steps += 1
        if prior not in regime_by_day:
            continue
        if d in select_dates:
            select_pnl_by_regime[regime_by_day[prior]].append(float(pnl))

    print(f"\nSELECT-window per-regime PnL (used to pick tradable set):")
    tradable: set[int] = set()
    for j in range(n_states):
        bucket = select_pnl_by_regime.get(j, [])
        if not bucket:
            print(f"  regime {j}: 0 days, no data")
            continue
        mu = sum(bucket) / len(bucket)
        total = sum(bucket)
        print(f"  regime {j}: {len(bucket):>3} days  mean=${mu:+8.2f}/day  total=${total:+,.0f}")
        if mu > 0:
            tradable.add(j)
    print(f"  -> tradable regimes (SELECT mean > 0): {sorted(tradable)}")
    if not tradable:
        print("  no regime had positive mean PnL on SELECT -- the filter "
              "would block all trading. Falling back to the SELECT-best "
              "regime alone.")
        best_j = max(range(n_states),
                     key=lambda j: (sum(select_pnl_by_regime.get(j, []))
                                    / max(1, len(select_pnl_by_regime.get(j, [])))))
        tradable = {best_j}
        print(f"  -> fallback: keep regime {best_j}")

    # ----- 7) re-run ORB with regime filter, restricted to TEST ------
    def daily_filter(d: date) -> bool:
        prior = d - timedelta(days=1)
        steps = 0
        while prior not in regime_by_day and steps < 7:
            prior -= timedelta(days=1)
            steps += 1
        if prior not in regime_by_day:
            return False
        return regime_by_day[prior] in tradable

    print(f"\nRunning ORB on full series with regime filter; "
          f"measuring TEST window only...", flush=True)
    t0 = _time.time()
    filtered = run_orb(bars, daily_filter=daily_filter)
    print(f"  done in {_time.time()-t0:.1f}s, "
          f"{len(filtered.daily_pnl)} trading days emitted")

    # Restrict daily_pnl to TEST window
    test_daily = {d: v for d, v in filtered.daily_pnl.items() if d in test_dates}
    base_test_daily = {d: v for d, v in baseline.daily_pnl.items() if d in test_dates}
    print(f"  TEST trading days: filtered={len(test_daily)}, baseline={len(base_test_daily)}")

    if not test_daily:
        print("  filtered strategy produced no trading days in TEST window.")
        return

    print(f"\n=== TEST window results ===")
    print(f"{'metric':>28} {'baseline':>12} {'HMM-gated':>12}")
    perf_base = performance(base_test_daily,
                            [(d, sum(list(base_test_daily.values())[:i+1])
                              + RULES.starting_balance)
                             for i, d in enumerate(sorted(base_test_daily))],
                            starting_balance=RULES.starting_balance)
    perf_filt = performance(test_daily,
                            [(d, sum(list(test_daily.values())[:i+1])
                              + RULES.starting_balance)
                             for i, d in enumerate(sorted(test_daily))],
                            starting_balance=RULES.starting_balance)
    print(f"{'trading days':>28} {perf_base.n_periods:>12} {perf_filt.n_periods:>12}")
    print(f"{'total PnL ($)':>28} {float(perf_base.total_pnl):>12,.0f} "
          f"{float(perf_filt.total_pnl):>12,.0f}")
    print(f"{'Sharpe (annual)':>28} {perf_base.sharpe_annual:>12.2f} "
          f"{perf_filt.sharpe_annual:>12.2f}")
    print(f"{'Profit factor':>28} {perf_base.profit_factor:>12.2f} "
          f"{perf_filt.profit_factor:>12.2f}")
    print(f"{'win-rate (day)':>28} {perf_base.win_rate:>12.1%} "
          f"{perf_filt.win_rate:>12.1%}")
    print(f"{'Max DD ($)':>28} {float(perf_base.max_drawdown_dollars):>12,.0f} "
          f"{float(perf_filt.max_drawdown_dollars):>12,.0f}")
    print(f"{'best day ($)':>28} {float(perf_base.best_period):>12,.0f} "
          f"{float(perf_filt.best_period):>12,.0f}")
    print(f"{'worst day ($)':>28} {float(perf_base.worst_period):>12,.0f} "
          f"{float(perf_filt.worst_period):>12,.0f}")

    for window in (30, 45):
        if len(test_daily) >= window:
            rr_b = realized_pass_rate(base_test_daily, rules=RULES,
                                      starting_balance=RULES.starting_balance,
                                      window_days=window, stride_days=1)
            rr_f = realized_pass_rate(test_daily, rules=RULES,
                                      starting_balance=RULES.starting_balance,
                                      window_days=window, stride_days=1)
            print(f"{f'pass-rate ({window}d window)':>28} "
                  f"{rr_b.pass_rate:>11.1%} ({rr_b.n_passed}/{rr_b.n_windows})  "
                  f"{rr_f.pass_rate:>11.1%} ({rr_f.n_passed}/{rr_f.n_windows})")


if __name__ == "__main__":
    main()
