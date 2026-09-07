"""Vol-targeting and Kelly-fractional sizing on the 9-stream gated ensemble.

Same 9 streams, same gates, same pass-rate-aware weights. We test two
classical sizing rules that do NOT depend on the within-window buffer
(unlike path-aware sizing):

  1. CONSTANT KELLY-MULTIPLIER: scale the ensemble's daily PnL by a
     constant factor k. The "full Kelly" k* is derived from TRAIN
     statistics:  k* = mu_train * B / sigma_train^2  (with B = starting
     balance). Fractional Kelly variants test {0.25, 0.5, 0.75, 1.0}*k*.
     k* is computed ONCE from TRAIN data, then held fixed across TEST.

  2. VOL-TARGETING: at day t, scale by target_vol / realized_vol_{t-L:t-1}
     where target_vol is the TRAIN realised stdev (pre-committed) and L
     is the rolling lookback. Pre-commit L=20; sensitivity L=10, 60.
     Strictly causal (window ends at day t-1). Scale clipped to
     [0.25, 4.0] to prevent extreme moves.

Discipline:
  * Strategy code and weights unchanged from `run_ensemble_multi_asset.py`.
  * k* derived from TRAIN only. Vol target from TRAIN only. Held fixed
    across TEST.
  * All evaluations under PROPER TopStep rules (simulate_combine_window).
  * Single hyperparameter per family (Kelly fraction, vol lookback).

Hypothesis (informed by path-aware sizing null result):
The trailing MLL geometry punishes peak ratcheting. Constant multiplier
>1 will inflate peaks. Vol-targeting raises size on calm days (also
inflates peaks). Both are expected to fail for the same reason.
This test FALSIFIES or confirms that hypothesis on real data.
"""

from __future__ import annotations

import sys
import time as _time
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.dsr import deflated_sharpe
from topstep50k.analysis.passrate import realized_pass_rate, simulate_combine_window
from topstep50k.rules import combine_25k, combine_50k


RULES = combine_50k()
RULESETS = {"$50K Combine": combine_50k(), "$25K Combine": combine_25k()}
WEIGHT_CRITERION_WINDOW = 30

CACHE_PATH = ROOT / "results" / "cache" / "multi_asset_streams.npz"


def load_cached_streams():
    with np.load(CACHE_PATH, allow_pickle=False) as z:
        days = [date.fromisoformat(s) for s in z["days"]]
        keys = [tuple(s.split("|", 1)) for s in z["keys"]]
        data = z["data"]
    arrays = {k: data[i] for i, k in enumerate(keys)}
    return days, arrays


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


def pass_rate_under_rules(daily_arr, day_index, window, rules):
    """Proper-rules pass-rate using simulate_combine_window per window."""
    n = daily_arr.size
    if n < window:
        return {"n_passed": 0, "n_windows": 0, "pass_rate": 0.0,
                "outcomes": Counter()}
    total = n - window + 1
    passes = 0
    outcomes = Counter()
    for start in range(total):
        pnls = [(day_index[start + i],
                  Decimal(str(round(float(daily_arr[start + i]), 2))))
                 for i in range(window)]
        result = simulate_combine_window(
            pnls, rules=rules, starting_balance=rules.starting_balance)
        outcomes[result.outcome] += 1
        if result.outcome == "pass":
            passes += 1
    return {"n_passed": passes, "n_windows": total,
            "pass_rate": passes / total, "outcomes": outcomes}


def vol_targeted_series(daily: np.ndarray, target_vol: float,
                         lookback: int, scale_min: float = 0.25,
                         scale_max: float = 4.0) -> np.ndarray:
    """Causal vol-targeting: scale at day t uses [t-L:t-1] realised stdev.

    For the first `lookback` days, scale = 1.0 (no data yet).
    """
    n = daily.size
    out = np.zeros(n, dtype=float)
    for t in range(n):
        if t < lookback:
            scale = 1.0
        else:
            window = daily[t - lookback : t]
            sd = float(window.std(ddof=1)) if window.size > 1 else 0.0
            if sd <= 1e-9:
                scale = 1.0
            else:
                scale = target_vol / sd
                scale = max(scale_min, min(scale_max, scale))
        out[t] = scale * daily[t]
    return out


def main():
    print(f"Loading cached streams from {CACHE_PATH}...", flush=True)
    union_days, arrays = load_cached_streams()
    print(f"  {len(arrays)} streams over {len(union_days)} days", flush=True)

    first_day = union_days[0]
    train_end = first_day + timedelta(days=365 * 2)
    train_mask = np.array([d < train_end for d in union_days])
    test_mask = ~train_mask
    test_days = [d for d, m in zip(union_days, test_mask) if m]
    print(f"TRAIN days: {int(train_mask.sum())}  TEST days: {int(test_mask.sum())}")

    # ---- Reproduce pass-rate-aware weights from TRAIN ----
    rules_for_weights = combine_25k()
    target_w = float(rules_for_weights.profit_target)
    mll_w = float(rules_for_weights.max_loss_limit_distance)
    solo_pr = {}
    for k, arr in arrays.items():
        pr = fast_pass_rate(arr[train_mask], target=target_w,
                             mll_distance=mll_w,
                             window=WEIGHT_CRITERION_WINDOW)
        solo_pr[k] = pr
    z = sum(solo_pr.values())
    weights = ({k: v / z for k, v in solo_pr.items()} if z > 0
               else {k: 1.0 / len(solo_pr) for k in solo_pr})

    # ---- Build the 9-stream weighted ensemble ----
    multi_full = np.zeros(len(union_days), dtype=float)
    for k, w in weights.items():
        multi_full += w * arrays[k]
    multi_train = multi_full[train_mask]
    multi_test = multi_full[test_mask]
    print(f"\nEnsemble TRAIN stats: mu=${multi_train.mean():+.2f}/d  "
          f"sigma=${multi_train.std(ddof=1):.2f}/d  "
          f"Sharpe={sharpe_annual(multi_train):+.2f}")
    print(f"Ensemble TEST  stats: mu=${multi_test.mean():+.2f}/d  "
          f"sigma=${multi_test.std(ddof=1):.2f}/d  "
          f"Sharpe={sharpe_annual(multi_test):+.2f}")

    # ---- (1) Kelly multiplier on TRAIN ----
    mu_tr = float(multi_train.mean())
    sd_tr = float(multi_train.std(ddof=1))
    B = float(RULES.starting_balance)
    k_full = (mu_tr * B / (sd_tr * sd_tr)) if sd_tr > 0 else 0.0
    print(f"\n{'='*78}")
    print(f"Kelly multiplier derivation (TRAIN, $50K bankroll)")
    print(f"{'='*78}")
    print(f"  mu_train = ${mu_tr:.2f}/d  sigma_train = ${sd_tr:.2f}/d")
    print(f"  full-Kelly multiplier k* = mu*B/sigma^2 = "
          f"${mu_tr:.2f} * {B:,.0f} / {sd_tr:.2f}^2 = {k_full:.3f}")
    print(f"  (interpretation: at k*={k_full:.2f}, expected log-growth "
          f"per day is maximised given TRAIN distribution.)")

    kelly_variants = [
        ("baseline qty=1     ", 1.0),
        ("0.25*k*            ", 0.25 * k_full),
        ("0.5*k* (half-Kelly)", 0.5 * k_full),
        ("0.75*k*            ", 0.75 * k_full),
        ("1.0*k* (full Kelly)", 1.0 * k_full),
    ]

    print(f"\n{'='*78}")
    print(f"CONSTANT KELLY-MULTIPLIER pass-rates on TEST (proper rules)")
    print(f"{'='*78}")
    print(f"  {'variant':<22} {'mult':>6} | "
          f"{'$50K p30':>9} {'$50K p45':>9} | "
          f"{'$25K p30':>9} {'$25K p45':>9} | Sharpe")
    kelly_results = {}
    for tag, mult in kelly_variants:
        scaled = mult * multi_test
        row = f"  {tag:<22} {mult:>6.2f} | "
        for rname, rules in RULESETS.items():
            for window in (30, 45):
                pr = pass_rate_under_rules(scaled, test_days, window, rules)
                row += f"{pr['pass_rate']:>8.1%}  "
                kelly_results[(tag, rname, window)] = pr["pass_rate"]
            row += "| " if rname == "$50K Combine" else ""
        row += f" {sharpe_annual(scaled):+.2f}"
        print(row)

    # ---- (2) Vol-targeting ----
    target_vol = sd_tr  # pre-committed: target = TRAIN realised stdev
    print(f"\n{'='*78}")
    print(f"VOL-TARGETING (target = TRAIN realised stdev = ${target_vol:.2f}/d)")
    print(f"{'='*78}")
    vol_variants = [
        ("baseline qty=1     ", None),
        ("vol-target L=10    ", 10),
        ("vol-target L=20 *  ", 20),  # pre-committed MAIN
        ("vol-target L=60    ", 60),
    ]
    print(f"  {'variant':<22} | {'$50K p30':>9} {'$50K p45':>9} | "
          f"{'$25K p30':>9} {'$25K p45':>9} | Sharpe  scale_mean")
    vol_results = {}
    for tag, L in vol_variants:
        if L is None:
            scaled = multi_test
            sm = 1.0
        else:
            # Need to use full series for causal lookback (TRAIN warms up)
            scaled_full = vol_targeted_series(multi_full,
                                                target_vol=target_vol,
                                                lookback=L)
            scaled = scaled_full[test_mask]
            # Effective scale mean on TEST
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(multi_test) > 1e-6,
                                  scaled / multi_test, np.nan)
            sm = float(np.nanmean(ratio))
        row = f"  {tag:<22} | "
        for rname, rules in RULESETS.items():
            for window in (30, 45):
                pr = pass_rate_under_rules(scaled, test_days, window, rules)
                row += f"{pr['pass_rate']:>8.1%}  "
                vol_results[(tag, rname, window)] = pr["pass_rate"]
            row += "| " if rname == "$50K Combine" else ""
        row += f" {sharpe_annual(scaled):+.2f}   {sm:>5.2f}"
        print(row)

    # ---- Headline summary ----
    print(f"\n{'='*78}")
    print(f"HEADLINE: every advanced sizing variant vs baseline on $50K pass45")
    print(f"{'='*78}")
    baseline_p45 = kelly_results[("baseline qty=1     ", "$50K Combine", 45)]
    print(f"  {'variant':<25}  {'$50K p45':>9}  {'vs baseline':>12}")
    print(f"  {'baseline (qty=1)':<25}  {baseline_p45:>8.1%}  "
          f"{'(reference)':>12}")
    for tag, mult in kelly_variants[1:]:
        v = kelly_results[(tag, "$50K Combine", 45)]
        print(f"  {tag:<25}  {v:>8.1%}  {v - baseline_p45:>+11.1%}")
    for tag, L in vol_variants[1:]:
        v = vol_results[(tag, "$50K Combine", 45)]
        print(f"  {tag:<25}  {v:>8.1%}  {v - baseline_p45:>+11.1%}")

    # ---- DSR -- count all new trials honestly ----
    print(f"\n{'='*78}")
    print(f"Deflated Sharpe -- {len(kelly_variants)-1 + len(vol_variants)-1} "
          f"new trials this pass")
    print(f"{'='*78}")
    trials = [sharpe_annual(multi_test)]
    series_for_dsr = {}
    for tag, mult in kelly_variants[1:]:
        s = mult * multi_test
        series_for_dsr[tag] = s
        trials.append(sharpe_annual(s))
    for tag, L in vol_variants[1:]:
        scaled_full = vol_targeted_series(multi_full,
                                            target_vol=target_vol,
                                            lookback=L)
        s = scaled_full[test_mask]
        series_for_dsr[tag] = s
        trials.append(sharpe_annual(s))
    trials = [s for s in trials if not np.isnan(s)]

    # Winner = whichever variant scored highest on $50K p45
    all_p45 = {tag: kelly_results[(tag, "$50K Combine", 45)]
                for tag, _ in kelly_variants[1:]}
    all_p45.update({tag: vol_results[(tag, "$50K Combine", 45)]
                     for tag, _ in vol_variants[1:]})
    winner_tag = max(all_p45, key=lambda k: all_p45[k])
    winner = series_for_dsr[winner_tag]

    starting = float(RULES.starting_balance)
    try:
        dsr = deflated_sharpe(
            (winner / starting).tolist(),
            all_trial_sharpes_annual=trials,
            periods_per_year=252,
        )
        print(f"  winner (by $50K p45): {winner_tag}  "
              f"(pass45={all_p45[winner_tag]:.1%})")
        print(f"    raw Sharpe (annual): {dsr.sharpe:.3f}")
        print(f"    expected-max under null: {dsr.expected_max_sharpe:.3f}")
        print(f"    DSR: {dsr.dsr:.4f}")
        print(f"    trials this pass: {len(trials)}  "
              f"(cumulative since project start: 19+{len(trials)}="
              f"{19 + len(trials)})")
    except Exception as e:
        print(f"  DSR failed: {e}")


if __name__ == "__main__":
    main()
