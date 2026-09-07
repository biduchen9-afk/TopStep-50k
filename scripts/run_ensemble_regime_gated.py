"""Three-strategy ensemble with literature-grounded regime gates.

Same three strategies and pre-committed parameters as run_ensemble.py,
but each strategy is gated by a rule-based regime conditioner sourced
from the academic literature. The conditioners are deliberately simple
and parameter-free; rationale is in
`src/topstep50k/regime/conditioners.py` and in the comments below.

Overfitting safeguards:

  * Gate rules picked from the literature BEFORE looking at our TEST
    output. None have free thresholds we tuned.
  * Same TRAIN/TEST split as baseline ensemble (first 2y / last 2.25y).
    Realised pass-rate reported on TEST only.
  * Same inverse-vol weights as baseline ensemble are reused -- we
    don't re-derive weights from the gated PnL series (that would be
    a second-pass fit). The weights come from the BASELINE TRAIN
    stdev as before.
  * All three gates always applied -- we don't pick the "best" subset.
  * Both rulesets ($50K and $25K) reported side-by-side.
  * DSR computed with honest trial count = 2 prior ensembles (raw +
    25K-eval) + 1 gated ensemble + 3 drop-one diagnostics = 6.
"""

from __future__ import annotations

import sys
import time as _time
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
RULESETS = {
    "$50K Combine": combine_50k(),
    "$25K Combine": combine_25k(),
}
RULES = combine_50k()  # reference for DSR fractions

# Pre-committed strategy parameters (same as baseline ensemble).
ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both",
                   stop_mode="opposite_range", stop_ticks=40,
                   tp_multiple=1.0, flat_before_close_minutes=15,
                   tick_size=0.25)
MR_PARAMS = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                  time_stop_minutes=45, tick_size=0.25)
OD_PARAMS = dict(qty=1, entry_offset_minutes=5, exit_offset_minutes=5)


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


def pass_rate_from_daily(daily, day_index, window, rules):
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
    line = (f"  {label:>22} : days={daily.size}  nz={nz:>4}  "
            f"total=${daily.sum():>+9,.0f}  Sharpe={sh:>+5.2f}  "
            f"win-rate={wr:.1%}  PF={pf:>4.2f}  MaxDD=${mdd:>+8,.0f}")
    print(line)
    for rname, rules in RULESETS.items():
        rr30 = pass_rate_from_daily(daily, day_index, 30, rules)
        rr45 = pass_rate_from_daily(daily, day_index, 45, rules)
        print(f"  {'':>22}   {rname:<14}  "
              f"pass30={rr30.pass_rate:>5.1%} ({rr30.n_passed}/{rr30.n_windows})  "
              f"pass45={rr45.pass_rate:>5.1%} ({rr45.n_passed}/{rr45.n_windows})")


def main():
    print("Loading ES bars...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(ROOT / "data" / "raw" / "es_cleaned.txt"))
    print(f"  {len(bars):,} bars in {_time.time() - t0:.1f}s", flush=True)

    # --- 1) compute the literature-grounded gates from full bar history ---
    print("\nBuilding regime conditioners from bar history...", flush=True)
    t0 = _time.time()
    stats = per_day_session_stats(bars)
    g_orb = orb_expansion_gate(stats)
    g_mr = meanrev_low_vol_gate(stats)
    g_od = overnight_drift_post_selloff_gate(stats)
    print(f"  per-day stats for {len(stats)} dates, "
          f"{_time.time() - t0:.1f}s", flush=True)

    # Quick sanity: how often does each gate fire?
    all_days = sorted(stats.keys())
    print(f"\nGate fire-rates over the full history ({len(all_days)} days):")
    print(f"  ORB-expansion gate    : "
          f"{sum(g_orb(d) for d in all_days) / len(all_days):.1%}")
    print(f"  MeanRev-low-vol gate  : "
          f"{sum(g_mr(d) for d in all_days) / len(all_days):.1%}")
    print(f"  OvernightDrift gate   : "
          f"{sum(g_od(d) for d in all_days) / len(all_days):.1%} "
          f"(NY Fed 'post-selloff')")

    # --- 2) baseline (un-gated) AND gated runs for each strategy ---
    print("\nRunning UN-GATED baseline strategies (qty=1)...", flush=True)
    base_orb = run_strategy(bars,
        lambda: OpeningRangeBreakout(symbol="ES", **ORB_PARAMS), "ORB-base")
    base_mr = run_strategy(bars,
        lambda: MeanReversionBollinger(symbol="ES", **MR_PARAMS), "MeanRev-base")
    base_od = run_strategy(bars,
        lambda: OvernightDrift(symbol="ES", **OD_PARAMS), "OD-base")

    print("\nRunning GATED strategies (literature conditioners)...", flush=True)
    gated_orb = run_strategy(bars,
        lambda: OpeningRangeBreakout(symbol="ES",
                                      daily_filter=g_orb, **ORB_PARAMS),
        "ORB-gated")
    gated_mr = run_strategy(bars,
        lambda: MeanReversionBollinger(symbol="ES",
                                        daily_filter=g_mr, **MR_PARAMS),
        "MeanRev-gated")
    gated_od = run_strategy(bars,
        lambda: OvernightDrift(symbol="ES",
                                entry_filter=g_od, **OD_PARAMS),
        "OD-gated")

    # --- 3) align day index, TRAIN/TEST split ----------------------------
    union_days = sorted(set(base_orb) | set(base_mr) | set(base_od)
                        | set(gated_orb) | set(gated_mr) | set(gated_od))
    first_day = union_days[0]
    train_end = first_day + timedelta(days=365 * 2)
    train_mask = np.array([d < train_end for d in union_days])
    test_mask = ~train_mask
    train_days = [d for d, m in zip(union_days, train_mask) if m]
    test_days = [d for d, m in zip(union_days, test_mask) if m]
    print(f"\nTRAIN: {train_days[0]} -> {train_days[-1]} ({len(train_days)} days)")
    print(f"TEST : {test_days[0]} -> {test_days[-1]} ({len(test_days)} days)")

    series_base = {
        "ORB": to_series(base_orb, union_days),
        "MeanRev": to_series(base_mr, union_days),
        "OvernightDrift": to_series(base_od, union_days),
    }
    series_gated = {
        "ORB": to_series(gated_orb, union_days),
        "MeanRev": to_series(gated_mr, union_days),
        "OvernightDrift": to_series(gated_od, union_days),
    }

    # --- 4) re-use the BASELINE TRAIN inverse-vol weights ----------------
    # Critically: we do NOT re-derive weights from gated TRAIN PnL --
    # that would tune sizing to the gate's effect. The weights are
    # fixed from the prior ensemble run.
    train_stds = {name: float(series_base[name][train_mask].std(ddof=1))
                   for name in series_base}
    inv_vol = {n: (1.0 / s if s > 0 else 0.0)
                for n, s in train_stds.items()}
    z = sum(inv_vol.values())
    weights = {n: w / z for n, w in inv_vol.items()}
    print(f"\nInverse-vol weights (from BASELINE TRAIN, unchanged): "
          f"{ {n: round(w, 3) for n, w in weights.items()} }")

    def ensemble(per_strategy: dict[str, np.ndarray]) -> np.ndarray:
        sub_w = {n: weights[n] for n in per_strategy}
        z = sum(sub_w.values())
        sub_w = {n: w / z for n, w in sub_w.items()}
        out = np.zeros(next(iter(per_strategy.values())).size, dtype=float)
        for n, s in per_strategy.items():
            out += sub_w[n] * s
        return out

    test_base = {n: s[test_mask] for n, s in series_base.items()}
    test_gated = {n: s[test_mask] for n, s in series_gated.items()}

    # --- 5) per-strategy report on TEST: base vs gated -------------------
    print(f"\n=== Per-strategy on TEST: BASELINE vs GATED ===")
    for name in ("ORB", "MeanRev", "OvernightDrift"):
        print()
        report(f"{name} BASE (TEST)", test_base[name], test_days)
        report(f"{name} GATED (TEST)", test_gated[name], test_days)

    # --- 6) ensemble: baseline vs gated ----------------------------------
    print(f"\n=== Ensemble on TEST: BASELINE vs GATED ===")
    ens_base = ensemble(test_base)
    ens_gated = ensemble(test_gated)
    print()
    report("ENSEMBLE BASE (TEST)", ens_base, test_days)
    report("ENSEMBLE GATED (TEST)", ens_gated, test_days)

    # --- 7) drop-one diagnostics (sensitivity, NOT selection) ------------
    print(f"\n=== Drop-one diagnostics, GATED ensemble (sensitivity) ===")
    names = list(test_gated.keys())
    for drop in names:
        sub = {n: a for n, a in test_gated.items() if n != drop}
        report(f"GATED drop {drop}", ensemble(sub), test_days)

    # --- 8) DSR with honest trial count ----------------------------------
    print(f"\n=== Deflated Sharpe ===")
    trials = []
    for arr in test_base.values():
        trials.append(sharpe_annual(arr))
    for arr in test_gated.values():
        trials.append(sharpe_annual(arr))
    trials.append(sharpe_annual(ens_base))
    trials.append(sharpe_annual(ens_gated))
    for drop in names:
        sub = {n: a for n, a in test_gated.items() if n != drop}
        trials.append(sharpe_annual(ensemble(sub)))
    trials = [s for s in trials if not np.isnan(s)]
    starting = float(RULES.starting_balance)
    try:
        dsr = deflated_sharpe(
            (ens_gated / starting).tolist(),
            all_trial_sharpes_annual=trials,
            periods_per_year=252,
        )
        print(f"  winner: GATED ensemble")
        print(f"    raw Sharpe (annual): {dsr.sharpe:.3f}")
        print(f"    expected-max under null: {dsr.expected_max_sharpe:.3f}")
        print(f"    DSR: {dsr.dsr:.4f}")
        print(f"    trials accumulated this pass: {len(trials)}")
        verdict = ("significant" if dsr.dsr >= 0.95
                   else "NOT statistically distinguishable from selection bias")
        print(f"    -> {verdict}")
    except Exception as e:
        print(f"  DSR failed: {e}")


if __name__ == "__main__":
    main()
