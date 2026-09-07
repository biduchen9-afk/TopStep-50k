"""Multi-asset gated ensemble: 3 strategies x 3 assets = 9 streams,
combined with pass-rate-aware sizing.

Same three strategies (ORB, MeanRev, OvernightDrift), same literature-
grounded gates, same pre-committed parameters as the single-asset
pipeline. We fan out across ES, NQ, GC and recombine with the same
pass-rate-aware weighting that delivered the $50K pass-rate lift.

Discipline (overfit safeguards):

  * Strategy parameters PRE-COMMITTED from the single-asset work.
    Adjusting them per-asset would be a fishing expedition.
  * RTH window: 09:30-16:00 US/Eastern uniformly across all three
    assets. Yes, gold's official COMEX session is 08:20-13:30 ET --
    we acknowledge that uniform RTH is a simplification, and the
    pass-rate-aware sizer should naturally downweight strategies
    that perform badly on the off-RTH assets.
  * The OvernightDrift gate (post-selloff RTH) was derived for
    equities (NY Fed SR #917). We apply it uniformly to GC too;
    if it generates noise on GC, the sizer will assign GC-OD a
    near-zero weight.
  * Pass-rate-aware weights derived on TRAIN window only ($25K
    criterion, 30-day window), then held fixed on TEST.
  * All nine streams ALWAYS included (no drop-the-loser).
  * Report on TEST only.

Instrument specs (Topstep + CME standards):
  ES: $50/pt, 0.25 tick (S&P 500 E-mini)
  NQ: $20/pt, 0.25 tick (Nasdaq 100 E-mini)
  GC: $100/pt, 0.10 tick (Gold)
  All: $2.50/side commission (Topstep standard estimate)
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


# Asset configs
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
RULESETS = {"$50K Combine": combine_50k(), "$25K Combine": combine_25k()}

# PRE-COMMITTED strategy parameters
ORB_PARAMS_BASE = dict(qty=1, or_minutes=30, direction="both",
                        stop_mode="opposite_range", stop_ticks=40,
                        tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS_BASE = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                       time_stop_minutes=45)
OD_PARAMS_BASE = dict(qty=1, entry_offset_minutes=5, exit_offset_minutes=5)

WF_LOOKBACK_DAYS = 252
WEIGHT_CRITERION_WINDOW = 30


def run_strategy(bars, build_strategy, asset_key, label):
    asset = ASSETS[asset_key]
    strat = build_strategy()
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: asset["instrument"]},
                    strategy=pf, audit=InMemoryAuditLog(),
                    combine_enforcement=False)
    t0 = _time.time()
    result = bt.run(clk, src)
    nz = sum(1 for v in result.daily_pnl.values() if v != 0)
    print(f"  [{label:>22}] {len(result.daily_pnl)} days, "
          f"{nz} nonzero, {_time.time() - t0:.1f}s", flush=True)
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
    pf_v = (daily[daily > 0].sum() / -daily[daily < 0].sum()) \
        if (daily < 0).any() else float("inf")
    wr = float((daily > 0).sum()) / float(daily.size)
    mdd = max_drawdown(daily)
    nz = int((daily != 0).sum())
    print(f"  {label:>36} : days={daily.size}  nz={nz:>4}  "
          f"total=${daily.sum():>+10,.0f}  Sharpe={sh:>+5.2f}  "
          f"WR={wr:.1%}  PF={pf_v:>4.2f}  MaxDD=${mdd:>+9,.0f}")
    for rname, rules in RULESETS.items():
        rr30 = proper_pass_rate(daily, day_index, 30, rules)
        rr45 = proper_pass_rate(daily, day_index, 45, rules)
        print(f"  {'':>36}   {rname:<14}  "
              f"pass30={rr30.pass_rate:>5.1%} ({rr30.n_passed}/{rr30.n_windows})  "
              f"pass45={rr45.pass_rate:>5.1%} ({rr45.n_passed}/{rr45.n_windows})")


def main():
    # ---- Load all three asset bar files ----
    all_bars = {}
    for key in ASSETS:
        print(f"Loading {key}...", flush=True)
        t0 = _time.time()
        bars = list(load_bars_csv(ASSETS[key]["data_path"]))
        print(f"  {key}: {len(bars):,} bars in {_time.time() - t0:.1f}s",
              flush=True)
        all_bars[key] = bars

    # ---- Build literature gates per asset (causal, asset-specific) ----
    print("\nBuilding per-asset regime conditioners...", flush=True)
    gates = {}
    for key, bars in all_bars.items():
        stats = per_day_session_stats(bars)
        gates[key] = {
            "orb": orb_expansion_gate(stats),
            "mr": meanrev_low_vol_gate(stats),
            "od": overnight_drift_post_selloff_gate(stats),
        }

    # ---- Run 3 strategies on each of 3 assets ----
    print(f"\nRunning {len(ASSETS) * 3} strategy-asset combinations...",
          flush=True)
    series_dict: dict[tuple[str, str], dict] = {}
    for asset_key in ASSETS:
        ts = ASSETS[asset_key]["tick_size_f"]
        bars = all_bars[asset_key]
        g = gates[asset_key]

        series_dict[(asset_key, "ORB")] = run_strategy(
            bars,
            lambda gg=g, tts=ts, ak=asset_key: OpeningRangeBreakout(
                symbol=ak, tick_size=tts, daily_filter=gg["orb"],
                **ORB_PARAMS_BASE),
            asset_key, f"{asset_key}-ORB",
        )
        series_dict[(asset_key, "MeanRev")] = run_strategy(
            bars,
            lambda gg=g, tts=ts, ak=asset_key: MeanReversionBollinger(
                symbol=ak, tick_size=tts, daily_filter=gg["mr"],
                **MR_PARAMS_BASE),
            asset_key, f"{asset_key}-MeanRev",
        )
        series_dict[(asset_key, "OvernightDrift")] = run_strategy(
            bars,
            lambda gg=g, ak=asset_key: OvernightDrift(
                symbol=ak, entry_filter=gg["od"], **OD_PARAMS_BASE),
            asset_key, f"{asset_key}-OvernightDrift",
        )

    # ---- Build unified day index, TRAIN/TEST split ----
    union_days = sorted(set().union(*[set(s.keys())
                                       for s in series_dict.values()]))
    first_day = union_days[0]
    train_end = first_day + timedelta(days=365 * 2)
    train_mask = np.array([d < train_end for d in union_days])
    test_mask = ~train_mask
    train_days = [d for d, m in zip(union_days, train_mask) if m]
    test_days = [d for d, m in zip(union_days, test_mask) if m]
    print(f"\nTRAIN: {train_days[0]} -> {train_days[-1]} ({len(train_days)} d)")
    print(f"TEST : {test_days[0]} -> {test_days[-1]} ({len(test_days)} d)")

    # Build aligned ndarrays
    series_arrays: dict[tuple[str, str], np.ndarray] = {}
    for k, daily_pnl in series_dict.items():
        series_arrays[k] = to_series(daily_pnl, union_days)

    # ---- Solo TRAIN pass-rate per stream ($25K rules, pre-committed) ----
    rules_for_weights = combine_25k()
    target = float(rules_for_weights.profit_target)
    mll = float(rules_for_weights.max_loss_limit_distance)
    print(f"\n{'='*78}")
    print(f"Solo TRAIN pass-rates (gated, 30d, $25K rules) per stream")
    print(f"{'='*78}")
    solo_pr = {}
    for k, arr in series_arrays.items():
        pr = fast_pass_rate(arr[train_mask], target=target,
                             mll_distance=mll,
                             window=WEIGHT_CRITERION_WINDOW)
        solo_pr[k] = pr
        print(f"  {k[0]:>3}/{k[1]:<16} : solo TRAIN pass30 = {pr:>5.1%}  "
              f"TRAIN total $={arr[train_mask].sum():>+10,.0f}  "
              f"TRAIN stdev=${arr[train_mask].std(ddof=1):>+6,.0f}")

    total_pr = sum(solo_pr.values())
    if total_pr <= 0:
        n = len(solo_pr)
        weights = {k: 1.0 / n for k in solo_pr}
    else:
        weights = {k: v / total_pr for k, v in solo_pr.items()}
    print(f"\nPass-rate-aware weights (normalised):")
    for k in solo_pr:
        print(f"  {k[0]:>3}/{k[1]:<16} : {weights[k]:.3f}")

    # ---- Multi-asset ensembles ----
    def ensemble(weights_dict: dict, series: dict) -> np.ndarray:
        out = np.zeros(next(iter(series.values())).size, dtype=float)
        z = sum(weights_dict.values())
        if z <= 0:
            return out
        for k, w in weights_dict.items():
            out += (w / z) * series[k]
        return out

    # Per-asset 3-strategy ensembles (asset-internal pass-rate weighting)
    print(f"\n{'='*78}")
    print(f"Per-asset ensembles on TEST window")
    print(f"{'='*78}")
    test_arrays = {k: a[test_mask] for k, a in series_arrays.items()}
    for ak in ASSETS:
        asset_streams = {k: v for k, v in series_arrays.items() if k[0] == ak}
        # Re-derive within-asset pass-rate weights for clean comparison
        asset_solo = {k: solo_pr[k] for k in asset_streams}
        ze = sum(asset_solo.values())
        if ze > 0:
            asset_w = {k: v / ze for k, v in asset_solo.items()}
        else:
            asset_w = {k: 1.0 / len(asset_solo) for k in asset_solo}
        comb = ensemble(asset_w, {k: v[test_mask] for k, v in asset_streams.items()})
        report(f"{ak}-only ensemble (3-strategy)", comb, test_days)
        print()

    # The HEADLINE multi-asset ensemble
    print(f"{'='*78}")
    print(f"HEADLINE: 9-stream multi-asset ensemble on TEST")
    print(f"{'='*78}")
    multi_test = ensemble(weights, test_arrays)
    report("multi-asset pass-rate-weighted", multi_test, test_days)

    # Equal-weighted baseline as a sensitivity check
    eq_w = {k: 1.0 / len(test_arrays) for k in test_arrays}
    multi_eq = ensemble(eq_w, test_arrays)
    print()
    report("multi-asset EQUAL-weighted", multi_eq, test_days)

    # ---- Drop-asset diagnostics ----
    print(f"\n{'='*78}")
    print(f"Drop-asset sensitivity (post-hoc, NOT a recommendation)")
    print(f"{'='*78}")
    for ak in ASSETS:
        sub = {k: v for k, v in test_arrays.items() if k[0] != ak}
        sub_w = {k: weights[k] for k in sub}
        if sum(sub_w.values()) <= 0:
            continue
        report(f"drop asset {ak}", ensemble(sub_w, sub), test_days)

    # ---- DSR ----
    print(f"\n{'='*78}")
    print(f"Deflated Sharpe")
    print(f"{'='*78}")
    trials = []
    for arr in test_arrays.values():
        trials.append(sharpe_annual(arr))
    trials.append(sharpe_annual(multi_test))
    trials.append(sharpe_annual(multi_eq))
    for ak in ASSETS:
        sub = {k: v for k, v in test_arrays.items() if k[0] != ak}
        sub_w = {k: weights[k] for k in sub}
        if sum(sub_w.values()) <= 0:
            continue
        trials.append(sharpe_annual(ensemble(sub_w, sub)))
    trials = [s for s in trials if not np.isnan(s)]
    starting = float(RULES.starting_balance)
    try:
        dsr = deflated_sharpe(
            (multi_test / starting).tolist(),
            all_trial_sharpes_annual=trials,
            periods_per_year=252,
        )
        print(f"  candidate winner: multi-asset pass-rate")
        print(f"    raw Sharpe (annual): {dsr.sharpe:.3f}")
        print(f"    expected-max under null: {dsr.expected_max_sharpe:.3f}")
        print(f"    DSR: {dsr.dsr:.4f}")
        print(f"    trials this pass: {len(trials)}")
        v = ("statistically significant" if dsr.dsr >= 0.95
             else "NOT statistically distinguishable from selection bias")
        print(f"    -> {v}")
    except Exception as e:
        print(f"  DSR failed: {e}")

    # Correlation matrix across the 9 streams (TEST) -- diversification check
    print(f"\n{'='*78}")
    print(f"Cross-stream correlation on TEST (diversification check)")
    print(f"{'='*78}")
    keys = list(test_arrays.keys())
    cols = np.stack([test_arrays[k] for k in keys], axis=1)
    # Mask: only days where stream had nonzero PnL
    corr = np.zeros((len(keys), len(keys)))
    for i in range(len(keys)):
        for j in range(len(keys)):
            xi, xj = cols[:, i], cols[:, j]
            if xi.std(ddof=1) == 0 or xj.std(ddof=1) == 0:
                corr[i, j] = float("nan")
            else:
                corr[i, j] = float(np.corrcoef(xi, xj)[0, 1])
    # Compact print
    print(f"  {'':>20}" + "".join(f"{f'{k[0]}/{k[1][:6]}':>11}" for k in keys))
    for i, ki in enumerate(keys):
        row = "".join(f"{corr[i, j]:>+11.2f}" for j in range(len(keys)))
        print(f"  {f'{ki[0]}/{ki[1][:14]}':>20}{row}")


if __name__ == "__main__":
    main()
