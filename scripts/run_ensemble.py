"""Three-strategy ensemble: ORB + MeanReversion + OvernightDrift.

Discipline (to limit overfit):
  * Parameters pre-committed BEFORE running on the data. ORB uses the
    pre-sweep baseline (or=30, tp=1.0, both, opposite_range) -- NOT
    the sweep winner. Mean-reversion uses lookback=60, sigma=2.0 from
    Bollinger's original 1980s paper. Overnight uses long-only,
    5min/5min offsets.
  * Strict TRAIN/TEST split. TRAIN (first 2y) is used ONLY to compute
    inverse-vol sizing weights. TEST (last 2y+) is the only window we
    report pass-rate and Sharpe on.
  * All three strategies are INCLUDED in the headline ensemble. We do
    NOT drop the negative-Sharpe one. Drop-one diagnostics are
    reported but are sensitivity checks, not selection steps.
  * DSR is deflated for the trial count we honestly accumulated this
    pass: 3 individual strategies + 1 ensemble + 3 drop-one =
    7 reported numbers. (The 144-cell ORB sweep is upstream and was
    paid for in `sweep_orb_es_full.py`.)
"""

from __future__ import annotations

import sys
import time as _time
from collections import defaultdict
from datetime import date, timedelta
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
from topstep50k.rules import combine_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout
from topstep50k.strategy.overnight_drift import OvernightDrift


ES = Instrument(symbol="ES", point_value=Decimal("50"),
                tick_size=Decimal("0.25"),
                commission_per_side=Decimal("2.50"))
RULES = combine_50k()

# PRE-COMMITTED parameters. Do not tune from here.
ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both",
                   stop_mode="opposite_range", stop_ticks=40,
                   tp_multiple=1.0, flat_before_close_minutes=15,
                   tick_size=0.25)
MR_PARAMS = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                  time_stop_minutes=45, tick_size=0.25)
OD_PARAMS = dict(qty=1, entry_offset_minutes=5, exit_offset_minutes=5)


def run_strategy(bars, strategy_factory, label):
    strat = strategy_factory(symbol="ES")
    pf = PortfolioStrategy(components={"ES": strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({"ES": bars}, clk)
    bt = Backtester(rules=RULES, instruments={"ES": ES}, strategy=pf,
                    audit=InMemoryAuditLog(), combine_enforcement=False)
    t0 = _time.time()
    result = bt.run(clk, src)
    print(f"  [{label}] {len(result.daily_pnl)} trading days, "
          f"{_time.time() - t0:.1f}s", flush=True)
    return result.daily_pnl


def to_series(daily_pnl: dict, day_index: list[date]) -> np.ndarray:
    """Project a daily_pnl dict onto a fixed day index. Missing days = 0."""
    out = np.zeros(len(day_index), dtype=float)
    for i, d in enumerate(day_index):
        if d in daily_pnl:
            out[i] = float(daily_pnl[d])
    return out


def sharpe_annual(daily: np.ndarray) -> float:
    if daily.size < 2 or daily.std(ddof=1) == 0:
        return float("nan")
    return (daily.mean() / daily.std(ddof=1)) * np.sqrt(252)


def pass_rate_from_daily(daily: np.ndarray, day_index, window=30):
    """Realized sliding-window pass-rate on a daily PnL np array.

    The pass-rate calc expects Decimal values (it adds them to the
    Decimal starting balance), so quantise here to two decimal places.
    """
    daily_pnl = {d: Decimal(str(round(float(v), 2)))
                 for d, v in zip(day_index, daily)}
    return realized_pass_rate(daily_pnl, rules=RULES,
                               starting_balance=RULES.starting_balance,
                               window_days=window, stride_days=1)


def max_drawdown(daily: np.ndarray) -> float:
    if daily.size == 0:
        return 0.0
    eq = np.cumsum(daily)
    peaks = np.maximum.accumulate(eq)
    return float((eq - peaks).min())


def report_block(label, daily, day_index):
    if daily.size < 30:
        print(f"  [{label}] insufficient data ({daily.size} days)")
        return
    sh = sharpe_annual(daily)
    pf = (daily[daily > 0].sum() / -daily[daily < 0].sum()) \
        if (daily < 0).any() else float("inf")
    wr = float((daily > 0).sum()) / float(daily.size)
    mdd = max_drawdown(daily)
    rr30 = pass_rate_from_daily(daily, day_index, window=30)
    rr45 = pass_rate_from_daily(daily, day_index, window=45)
    total = float(daily.sum())
    best = float(daily.max())
    worst = float(daily.min())
    print(f"  {label:>20} : days={daily.size:>4}  "
          f"total=${total:>+9,.0f}  Sharpe={sh:>+5.2f}  "
          f"win-rate={wr:.1%}  PF={pf:>4.2f}  "
          f"MaxDD=${mdd:>+8,.0f}  best=${best:>+6,.0f}  worst=${worst:>+6,.0f}  "
          f"pass30={rr30.pass_rate:>5.1%} ({rr30.n_passed}/{rr30.n_windows})  "
          f"pass45={rr45.pass_rate:>5.1%}")


def main():
    print("Loading ES bars...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(ROOT / "data" / "raw" / "es_cleaned.txt"))
    print(f"  {len(bars):,} bars in {_time.time() - t0:.1f}s", flush=True)

    print("\nRunning each strategy independently (qty=1)...", flush=True)
    orb_daily = run_strategy(
        bars, lambda symbol: OpeningRangeBreakout(symbol=symbol, **ORB_PARAMS),
        "ORB",
    )
    mr_daily = run_strategy(
        bars, lambda symbol: MeanReversionBollinger(symbol=symbol, **MR_PARAMS),
        "MeanRev",
    )
    od_daily = run_strategy(
        bars, lambda symbol: OvernightDrift(symbol=symbol, **OD_PARAMS),
        "OvernightDrift",
    )

    # Unified day index across all three strategies
    all_days = sorted(set(orb_daily) | set(mr_daily) | set(od_daily))
    print(f"\nUnified day index: {len(all_days)} days, "
          f"{all_days[0]} -> {all_days[-1]}")

    # ---- TRAIN / TEST split: first 2 years for TRAIN, rest for TEST ----
    first_day = all_days[0]
    train_end = first_day + timedelta(days=365 * 2)
    train_mask = np.array([d < train_end for d in all_days])
    test_mask = ~train_mask
    train_days = [d for d, m in zip(all_days, train_mask) if m]
    test_days = [d for d, m in zip(all_days, test_mask) if m]
    print(f"  TRAIN: {train_days[0]} -> {train_days[-1]} ({len(train_days)} days)")
    print(f"  TEST : {test_days[0]} -> {test_days[-1]} ({len(test_days)} days)")

    series = {
        "ORB": to_series(orb_daily, all_days),
        "MeanRev": to_series(mr_daily, all_days),
        "OvernightDrift": to_series(od_daily, all_days),
    }

    # =====================================================================
    # INDIVIDUAL STRATEGIES, TRAIN window
    # =====================================================================
    print(f"\n=== Individual strategies, TRAIN window ===")
    train_arrays = {}
    for name, s in series.items():
        train_arrays[name] = s[train_mask]
        report_block(f"{name} (TRAIN)", train_arrays[name], train_days)

    # =====================================================================
    # SIZING -- compute inverse-vol weights on TRAIN only
    # =====================================================================
    # Each strategy is naturally at qty=1. The weight is a "shares-of-PnL"
    # multiplier; in practice an inverse-vol weight of 0.5 means we'd trade
    # half-size. We can't trade 0.5 contracts of ES so this is really only
    # representable with multi-asset / micro-contract sizing. For now we
    # interpret the weights as a HYPOTHETICAL combined PnL.
    train_stds = {name: float(train_arrays[name].std(ddof=1))
                   for name in series}
    print(f"\n=== TRAIN-window per-day stdev ===")
    for name, s in train_stds.items():
        print(f"  {name:>20} : stdev=${s:>+8,.2f}")

    # Inverse-vol weights, normalised to sum to a target combined stdev
    # of $500/day (a hand-picked daily-variance budget). At qty=1 ES the
    # un-scaled stdevs are large (often $300-500/day each); inverse-vol
    # gives strategies with lower stdev MORE weight. This is the standard
    # risk-parity recipe.
    inv_vol = {name: 1.0 / s if s > 0 else 0.0 for name, s in train_stds.items()}
    raw_total = sum(inv_vol.values())
    weights = {name: w / raw_total for name, w in inv_vol.items()}
    print(f"\n=== Inverse-vol weights (TRAIN-derived) ===")
    for name, w in weights.items():
        print(f"  {name:>20} : weight={w:>5.3f}")

    def build_ensemble(strategies: dict[str, np.ndarray]) -> np.ndarray:
        """Combine the provided strategies using TRAIN-derived weights,
        renormalised to those still in the set."""
        sub_weights = {n: weights[n] for n in strategies}
        z = sum(sub_weights.values())
        sub_weights = {n: w / z for n, w in sub_weights.items()}
        combined = np.zeros(next(iter(strategies.values())).size, dtype=float)
        for n, s in strategies.items():
            combined += sub_weights[n] * s
        return combined

    # =====================================================================
    # INDIVIDUAL + ENSEMBLE, TEST window
    # =====================================================================
    print(f"\n=== Individual strategies, TEST window ===")
    test_arrays = {}
    for name, s in series.items():
        test_arrays[name] = s[test_mask]
        report_block(f"{name} (TEST)", test_arrays[name], test_days)

    # Correlation matrix on TEST series
    print(f"\n=== TEST-window correlation matrix ===")
    names = list(series.keys())
    matrix = np.zeros((len(names), len(names)))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            xi, xj = test_arrays[ni], test_arrays[nj]
            if xi.std(ddof=1) == 0 or xj.std(ddof=1) == 0:
                matrix[i, j] = float("nan")
            else:
                matrix[i, j] = float(np.corrcoef(xi, xj)[0, 1])
    print(f"  {'':>20} {' '.join(f'{n:>15}' for n in names)}")
    for i, ni in enumerate(names):
        row = " ".join(f"{matrix[i, j]:>+15.3f}" for j in range(len(names)))
        print(f"  {ni:>20} {row}")

    # Headline ensemble (all three, TEST)
    print(f"\n=== Headline ensemble (all three, TRAIN-weighted) ===")
    ensemble_test = build_ensemble(test_arrays)
    report_block("ALL-THREE (TEST)", ensemble_test, test_days)

    # Drop-one diagnostics (sensitivity, NOT selection)
    print(f"\n=== Drop-one diagnostics (sensitivity check) ===")
    for drop in names:
        sub = {n: a for n, a in test_arrays.items() if n != drop}
        comb = build_ensemble(sub)
        report_block(f"drop {drop}", comb, test_days)

    # =====================================================================
    # DSR with honest trial count (3 individual + 1 ensemble + 3 drop-one)
    # =====================================================================
    print(f"\n=== Deflated Sharpe ===")
    trial_sharpes_test = []
    for name, arr in test_arrays.items():
        trial_sharpes_test.append(sharpe_annual(arr))
    trial_sharpes_test.append(sharpe_annual(ensemble_test))
    for drop in names:
        sub = {n: a for n, a in test_arrays.items() if n != drop}
        comb = build_ensemble(sub)
        trial_sharpes_test.append(sharpe_annual(comb))
    trial_sharpes_test = [s for s in trial_sharpes_test if not np.isnan(s)]
    print(f"  N trials reported this pass: {len(trial_sharpes_test)}")
    # Deflate the BEST (winner) against this trial population
    winner_name = "ALL-THREE"
    winner_arr = ensemble_test
    # express daily return in fraction of starting balance for DSR
    starting = float(RULES.starting_balance)
    try:
        dsr = deflated_sharpe(
            (winner_arr / starting).tolist(),
            all_trial_sharpes_annual=trial_sharpes_test,
            periods_per_year=252,
        )
        print(f"  winner: {winner_name}")
        print(f"    raw Sharpe (annual): {dsr.sharpe:.3f}")
        print(f"    expected-max Sharpe under null: {dsr.expected_max_sharpe:.3f}")
        print(f"    trial-sharpe stdev: {dsr.sharpe_trial_std:.3f}")
        print(f"    DSR: {dsr.dsr:.4f}")
        if dsr.dsr < 0.95:
            print(f"    -> NOT statistically distinguishable from selection bias.")
        else:
            print(f"    -> statistically significant after deflation.")
    except Exception as e:
        print(f"  DSR computation failed: {e}")


if __name__ == "__main__":
    main()
