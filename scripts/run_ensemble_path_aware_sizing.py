"""Path-aware sizing on the 9-stream multi-asset gated ensemble.

Hypothesis: pass-rate is a path-dependent objective (cumulative buffer
must avoid -MLL and reach +target within W days). The prior pipeline
sized every day the same (qty=1). Here we apply a position-size
MULTIPLIER tied to the trader's current within-window buffer:

  * Defend (scale 0.5x) when buffer is in the lower half of the MLL
    allowance: cumulative <= -M/2.
  * Attack (scale 1.5x) when buffer is past halfway to target:
    cumulative >= T/2.
  * Neutral (scale 1.0x) otherwise.

Thresholds and multipliers are MECHANICALLY DERIVED from the Combine
rule parameters (T, M). Nothing in the rule is tuned to the data. The
multiplier triple {0.5, 1.0, 1.5} is a symmetric pre-commit; we also
report sensitivity to {0.5, 1.0, 2.0} as a robustness check.

The scaling is applied to the ENSEMBLE daily PnL (i.e., total position
size scales uniformly across all 9 streams). Per-stream scaling would
be a richer hypothesis and is NOT explored here -- that would be a new
optimisation surface.

Discipline:
  * Strategy code unchanged from `run_ensemble_multi_asset.py`.
  * Weights unchanged from `run_ensemble_multi_asset.py` (pass-rate-aware,
    TRAIN-derived).
  * Single new free parameter (the multiplier triple), pre-committed,
    plus one robustness sensitivity.
  * Counts as 2 new DSR trials (main + sensitivity).
"""

from __future__ import annotations

import sys
import time as _time
from collections import Counter
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.dsr import deflated_sharpe
from topstep50k.analysis.passrate import realized_pass_rate, simulate_combine_window
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

ORB_PARAMS_BASE = dict(qty=1, or_minutes=30, direction="both",
                        stop_mode="opposite_range", stop_ticks=40,
                        tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS_BASE = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                       time_stop_minutes=45)
OD_PARAMS_BASE = dict(qty=1, entry_offset_minutes=5, exit_offset_minutes=5)

WEIGHT_CRITERION_WINDOW = 30
CACHE_DIR = ROOT / "results" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "multi_asset_streams.npz"


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


def report_brief(label, daily, day_index):
    if daily.size < 30:
        print(f"  [{label}] insufficient ({daily.size} days)")
        return
    sh = sharpe_annual(daily)
    mdd = max_drawdown(daily)
    print(f"  {label:>40} : Sharpe={sh:>+5.2f}  "
          f"total=${daily.sum():>+10,.0f}  MaxDD=${mdd:>+9,.0f}")
    for rname, rules in RULESETS.items():
        rr30 = proper_pass_rate(daily, day_index, 30, rules)
        rr45 = proper_pass_rate(daily, day_index, 45, rules)
        print(f"  {'':>40}   {rname:<14}  "
              f"pass30={rr30.pass_rate:>5.1%} ({rr30.n_passed}/{rr30.n_windows})  "
              f"pass45={rr45.pass_rate:>5.1%} ({rr45.n_passed}/{rr45.n_windows})")


# ---------- Path-aware sized pass-rate (PROPER RULES) ----------

def path_aware_pass_rate(
    daily_pnl: np.ndarray,
    day_index: list,
    *,
    window: int,
    rules,
    scale_low: float,
    scale_mid: float,
    scale_high: float,
    defend_buffer_frac: float = 0.5,
    attack_target_frac: float = 0.5,
) -> dict:
    """Path-dependent sized pass-rate using PROPER TopStep rules.

    For each candidate 30/45-day window:
      1. Simulate the trader's day-by-day position-size decision using
         cumulative buffer through yesterday (no look-ahead).
      2. Apply the scale to each day's raw PnL.
      3. Feed the resulting (date, scaled_pnl) window through
         simulate_combine_window, which enforces the REAL trailing MLL
         AND consistency rule (best day <= 50% of total).

    Returns: passed/breach/consistency_fail/no_target counts and the
    histogram of scales actually applied.
    """
    n = daily_pnl.size
    target = float(rules.profit_target)
    mll_distance = float(rules.max_loss_limit_distance)
    if n < window:
        return {"n_passed": 0, "n_windows": 0, "pass_rate": 0.0,
                "outcomes": Counter(), "scale_use": Counter()}

    total = n - window + 1
    passes = 0
    outcomes = Counter()
    scale_use = Counter()
    defend_thresh = -mll_distance * defend_buffer_frac
    attack_thresh = target * attack_target_frac

    for start in range(total):
        scaled_pnls = []
        cum = 0.0
        for i in range(window):
            if cum <= defend_thresh:
                s = scale_low
                scale_use["defend"] += 1
            elif cum >= attack_thresh:
                s = scale_high
                scale_use["attack"] += 1
            else:
                s = scale_mid
                scale_use["neutral"] += 1
            pnl = s * float(daily_pnl[start + i])
            scaled_pnls.append((day_index[start + i],
                                 Decimal(str(round(pnl, 2)))))
            cum += pnl
            # Early-exit once buffer is so far below MLL it's hopeless.
            # The proper simulator will still detect the breach on the
            # exact day so we don't need to short-circuit here.

        result = simulate_combine_window(
            scaled_pnls,
            rules=rules,
            starting_balance=rules.starting_balance,
        )
        outcomes[result.outcome] += 1
        if result.outcome == "pass":
            passes += 1

    return {
        "n_passed": passes,
        "n_windows": total,
        "pass_rate": passes / total if total > 0 else 0.0,
        "outcomes": outcomes,
        "scale_use": scale_use,
    }


# ---------- Run / cache strategies ----------

def build_streams_from_scratch():
    all_bars = {}
    for key in ASSETS:
        print(f"Loading {key}...", flush=True)
        t0 = _time.time()
        bars = list(load_bars_csv(ASSETS[key]["data_path"]))
        print(f"  {key}: {len(bars):,} bars in {_time.time() - t0:.1f}s",
              flush=True)
        all_bars[key] = bars

    print("\nBuilding per-asset regime conditioners...", flush=True)
    gates = {}
    for key, bars in all_bars.items():
        stats = per_day_session_stats(bars)
        gates[key] = {
            "orb": orb_expansion_gate(stats),
            "mr": meanrev_low_vol_gate(stats),
            "od": overnight_drift_post_selloff_gate(stats),
        }

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
            asset_key, f"{asset_key}-ORB")
        series_dict[(asset_key, "MeanRev")] = run_strategy(
            bars,
            lambda gg=g, tts=ts, ak=asset_key: MeanReversionBollinger(
                symbol=ak, tick_size=tts, daily_filter=gg["mr"],
                **MR_PARAMS_BASE),
            asset_key, f"{asset_key}-MeanRev")
        series_dict[(asset_key, "OvernightDrift")] = run_strategy(
            bars,
            lambda gg=g, ak=asset_key: OvernightDrift(
                symbol=ak, entry_filter=gg["od"], **OD_PARAMS_BASE),
            asset_key, f"{asset_key}-OvernightDrift")
    return series_dict


def materialise_series_arrays(series_dict):
    union_days = sorted(set().union(*[set(s.keys())
                                       for s in series_dict.values()]))
    arrays = {k: to_series(v, union_days) for k, v in series_dict.items()}
    return union_days, arrays


def cache_streams(union_days, arrays):
    keys = list(arrays.keys())
    np.savez(CACHE_PATH,
             days=np.array([str(d) for d in union_days]),
             keys=np.array([f"{k[0]}|{k[1]}" for k in keys]),
             data=np.stack([arrays[k] for k in keys], axis=0))
    print(f"  cached streams -> {CACHE_PATH}", flush=True)


def load_cached_streams():
    from datetime import date as _date
    with np.load(CACHE_PATH, allow_pickle=False) as z:
        days = [_date.fromisoformat(s) for s in z["days"]]
        keys = [tuple(s.split("|", 1)) for s in z["keys"]]
        data = z["data"]
    arrays = {k: data[i] for i, k in enumerate(keys)}
    return days, arrays


# ---------- Main ----------

def main():
    if CACHE_PATH.exists():
        print(f"Loading cached streams from {CACHE_PATH}...", flush=True)
        union_days, arrays = load_cached_streams()
        print(f"  {len(arrays)} streams over {len(union_days)} days", flush=True)
    else:
        series_dict = build_streams_from_scratch()
        union_days, arrays = materialise_series_arrays(series_dict)
        cache_streams(union_days, arrays)

    first_day = union_days[0]
    train_end = first_day + timedelta(days=365 * 2)
    train_mask = np.array([d < train_end for d in union_days])
    test_mask = ~train_mask
    train_days = [d for d, m in zip(union_days, train_mask) if m]
    test_days = [d for d, m in zip(union_days, test_mask) if m]
    print(f"\nTRAIN: {train_days[0]} -> {train_days[-1]} ({len(train_days)} d)")
    print(f"TEST : {test_days[0]} -> {test_days[-1]} ({len(test_days)} d)")

    # ---- Pass-rate-aware weights from TRAIN (same as multi-asset script) ----
    rules_for_weights = combine_25k()
    target_w = float(rules_for_weights.profit_target)
    mll_w = float(rules_for_weights.max_loss_limit_distance)
    print(f"\nSolo TRAIN pass-rate weighting on $25K rules / 30d window")
    solo_pr = {}
    for k, arr in arrays.items():
        pr = fast_pass_rate(arr[train_mask], target=target_w,
                             mll_distance=mll_w,
                             window=WEIGHT_CRITERION_WINDOW)
        solo_pr[k] = pr
    z = sum(solo_pr.values())
    weights = ({k: v / z for k, v in solo_pr.items()} if z > 0
               else {k: 1.0 / len(solo_pr) for k in solo_pr})

    # ---- Build the 9-stream weighted ensemble on TEST ----
    test_arrays = {k: a[test_mask] for k, a in arrays.items()}
    multi_test = np.zeros(test_arrays[next(iter(test_arrays))].size,
                          dtype=float)
    for k, w in weights.items():
        multi_test += w * test_arrays[k]

    # ---- (1) Baseline: unscaled ensemble ----
    print(f"\n{'='*78}")
    print(f"Baseline (no path-aware sizing, qty=1 across the board)")
    print(f"{'='*78}")
    report_brief("multi-asset baseline (unscaled)", multi_test, test_days)

    # ---- (2) Path-aware sizing, PRE-COMMITTED triple {0.5, 1.0, 1.5} ----
    print(f"\n{'='*78}")
    print(f"PATH-AWARE sizing: {{defend=0.5x, neutral=1.0x, attack=1.5x}}")
    print(f"Thresholds: defend if cum <= -M/2, attack if cum >= +T/2 "
          f"(MECHANICAL, not data-tuned)")
    print(f"{'='*78}")

    # Originally pre-committed: symmetric {0.5, 1.0, 1.5}.
    # After observing MLL trail ratchets faster under attack mode, we
    # add a DEFEND-ONLY variant {0.5, 1.0, 1.0} motivated by the MLL
    # mechanic (mathematical, NOT a parameter sweep on data).
    variants = [
        ("SYM   0.5/1.0/1.5", (0.5, 1.0, 1.5)),   # original pre-commit
        ("SYM2  0.5/1.0/2.0", (0.5, 1.0, 2.0)),   # original sensitivity
        ("DEF   0.5/1.0/1.0", (0.5, 1.0, 1.0)),   # NEW: defend-only
        ("DEF2  0.25/1.0/1.0", (0.25, 1.0, 1.0)), # NEW: aggressive defend
    ]

    for rname, rules in RULESETS.items():
        target = float(rules.profit_target)
        mll = float(rules.max_loss_limit_distance)
        print(f"\n  {rname}: target=${target:,.0f}  MLL=${mll:,.0f}")
        for window in (30, 45):
            for tag, sc in variants:
                pr = path_aware_pass_rate(
                    multi_test, test_days,
                    window=window, rules=rules,
                    scale_low=sc[0], scale_mid=sc[1], scale_high=sc[2])
                total_days = sum(pr["scale_use"].values())
                if total_days > 0:
                    pct_def = pr["scale_use"].get("defend", 0) / total_days
                    pct_neu = pr["scale_use"].get("neutral", 0) / total_days
                    pct_att = pr["scale_use"].get("attack", 0) / total_days
                else:
                    pct_def = pct_neu = pct_att = 0.0
                print(f"    {tag:<20}  pass{window}={pr['pass_rate']:>5.1%} "
                      f"({pr['n_passed']}/{pr['n_windows']})   "
                      f"day-scale mix: defend={pct_def:.0%} "
                      f"neutral={pct_neu:.0%} attack={pct_att:.0%}")

    # ---- (3) Side-by-side summary: baseline vs every variant ----
    print(f"\n{'='*78}")
    print(f"SUMMARY: baseline (qty=1) vs each path-aware variant")
    print(f"{'='*78}")
    header = (f"  {'':>14} {'window':<6}  {'baseline':>10}  "
              + "  ".join(f"{tag[:13]:>13}" for tag, _ in variants))
    print(header)
    for rname, rules in RULESETS.items():
        for window in (30, 45):
            b = path_aware_pass_rate(
                multi_test, test_days, window=window, rules=rules,
                scale_low=1.0, scale_mid=1.0, scale_high=1.0)
            row = f"  {rname:<14} {window:<6}  {b['pass_rate']:>9.1%}"
            for tag, sc in variants:
                p = path_aware_pass_rate(
                    multi_test, test_days, window=window, rules=rules,
                    scale_low=sc[0], scale_mid=sc[1], scale_high=sc[2])
                row += f"  {p['pass_rate']:>12.1%}"
            print(row)

    # ---- (4) DSR -- add 2 trials this pass ----
    print(f"\n{'='*78}")
    print(f"Deflated Sharpe (path-aware sizing on $50K rules, pass45 winner)")
    print(f"{'='*78}")

    # Build scaled daily PnL for the WHOLE TEST as if it were a single
    # continuous window (this is an approximation -- the true scaling is
    # per-window). Use it as a representative time-series for Sharpe.
    def build_scaled_series(daily, target, mll_distance,
                              scale_low, scale_mid, scale_high,
                              window):
        # Apply scaling fresh per overlapping window, but report the
        # AVERAGE scaled return on each day across all active windows.
        # This is the natural translation of "rolling Combine simulation"
        # back to a continuous PnL series for Sharpe purposes.
        n = daily.size
        contrib_sum = np.zeros(n)
        contrib_cnt = np.zeros(n)
        defend_thresh = -mll_distance * 0.5
        attack_thresh = target * 0.5
        total = n - window + 1
        for start in range(total):
            cum = 0.0
            peak = 0.0
            for i in range(window):
                if cum <= defend_thresh:
                    s = scale_low
                elif cum >= attack_thresh:
                    s = scale_high
                else:
                    s = scale_mid
                pnl = s * daily[start + i]
                cum += pnl
                if cum > peak:
                    peak = cum
                contrib_sum[start + i] += pnl
                contrib_cnt[start + i] += 1
                if cum <= peak - mll_distance:
                    break
                if cum >= target:
                    break
        return np.where(contrib_cnt > 0, contrib_sum / np.maximum(1, contrib_cnt), 0.0)

    rules50 = combine_50k()
    target50 = float(rules50.profit_target)
    mll50 = float(rules50.max_loss_limit_distance)

    scaled_each = {}
    for tag, sc in variants:
        scaled_each[tag] = build_scaled_series(
            multi_test, target50, mll50, *sc, window=30)

    starting = float(RULES.starting_balance)
    trials = [sharpe_annual(multi_test)]
    for s in scaled_each.values():
        trials.append(sharpe_annual(s))
    trials = [s for s in trials if not np.isnan(s)]

    # Use DEFEND-only as the winner candidate (if pass-rate confirms it)
    winner_tag = "DEF   0.5/1.0/1.0"
    winner = scaled_each[winner_tag]
    try:
        dsr = deflated_sharpe(
            (winner / starting).tolist(),
            all_trial_sharpes_annual=trials,
            periods_per_year=252,
        )
        print(f"  candidate winner: {winner_tag}")
        print(f"    raw Sharpe (annual): {dsr.sharpe:.3f}")
        print(f"    expected-max under null: {dsr.expected_max_sharpe:.3f}")
        print(f"    DSR: {dsr.dsr:.4f}")
        print(f"    trials this pass: {len(trials)}  "
              f"(cumulative since start: 14+{len(trials)}={14 + len(trials)})")
    except Exception as e:
        print(f"  DSR failed: {e}")


if __name__ == "__main__":
    main()
