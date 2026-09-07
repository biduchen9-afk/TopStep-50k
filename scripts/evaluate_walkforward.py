"""Rolling walk-forward validation of ORB+MeanRev, instead of trusting
one arbitrary 70/30 IS/OOS split point.

Every result this session (v1 through v12, all screens) used a SINGLE
70/30 split: derive weights on the first 70% of history, touch the
last 30% exactly once. That's the right discipline against look-ahead,
but it's still only ONE draw of "which period happened to be OOS" --
if the original OOS window (2025-02-26 to 2026-07-03) was unusually
favorable (or unfavorable), the single-split number doesn't reveal
that.

This uses an EXPANDING walk-forward instead: minimum 500-day IS
window, then 5 sequential OOS folds of ~133 days each, each fold's
weights re-derived from ONLY the data before that fold (never after --
still zero look-ahead, just more touches of "the next chunk" rather
than one big one). The 5 folds' OOS segments are then CONCATENATED
into one long walk-forward-OOS series (~665 days, nearly double the
original 350-day one-touch OOS) and run through the same sequential-
account simulation + Monte Carlo distribution as every other result
this session, for a much more robust read on whether the edge (and
the weight-derivation process itself) is stable across different
historical periods or whether the original split just got lucky.

Run with: python scripts/evaluate_walkforward.py
"""

from __future__ import annotations

import sys
import time as _time
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.montecarlo import monte_carlo_pass_rate
from topstep50k.analysis.passrate import realized_pass_rate, simulate_sequential_accounts
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import meanrev_low_vol_gate, orb_expansion_gate, per_day_session_stats
from topstep50k.rules import combine_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout


ASSETS = {
    "ES": {
        "instrument": Instrument(symbol="ES", point_value=Decimal("50"),
                                  tick_size=Decimal("0.25"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.25,
        "data_path": ROOT / "data" / "raw" / "es_databento.txt",
    },
    "NQ": {
        "instrument": Instrument(symbol="NQ", point_value=Decimal("20"),
                                  tick_size=Decimal("0.25"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.25,
        "data_path": ROOT / "data" / "raw" / "nq_databento.txt",
    },
    "GC": {
        "instrument": Instrument(symbol="GC", point_value=Decimal("100"),
                                  tick_size=Decimal("0.10"),
                                  commission_per_side=Decimal("2.50")),
        "tick_size_f": 0.10,
        "data_path": ROOT / "data" / "raw" / "gc_databento.txt",
    },
}
RULES = combine_50k()
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)
ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both",
                   stop_mode="opposite_range", stop_ticks=40,
                   tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS  = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                   time_stop_minutes=45)

MIN_IS_DAYS = 500
N_FOLDS = 5


def run_stream(bars, build_fn, asset_key) -> dict[date, Decimal]:
    asset = ASSETS[asset_key]
    strat = build_fn()
    pf = PortfolioStrategy(components={asset_key: strat})
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: asset["instrument"]},
                    strategy=pf, audit=InMemoryAuditLog(), combine_enforcement=False)
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


def pass_rate_30(arr: np.ndarray, days: list[date]):
    pnl = {d: Decimal(str(round(float(v), 2))) for d, v in zip(days, arr)}
    return realized_pass_rate(pnl, rules=RULES, starting_balance=RULES.starting_balance,
                               window_days=30, stride_days=1)


def main():
    print("=" * 78)
    print(f"WALK-FORWARD VALIDATION -- ORB+MeanRev, {N_FOLDS} expanding folds")
    print("=" * 78)

    t0 = _time.time()
    all_bars, gates = {}, {}
    for ak in ASSETS:
        print(f"\nLoading {ak} bars...", flush=True)
        tl = _time.time()
        bars = [b for b in load_bars_csv(ASSETS[ak]["data_path"]) if b.ts >= RECENT_START]
        print(f"  {len(bars):,} bars in {_time.time()-tl:.1f}s", flush=True)
        all_bars[ak] = bars
        stats = per_day_session_stats(bars)
        gates[ak] = {"orb": orb_expansion_gate(stats), "mr": meanrev_low_vol_gate(stats)}

    print("\nRunning 6 ORB+MeanRev streams...", flush=True)
    streams = {}
    for ak in ASSETS:
        ts = ASSETS[ak]["tick_size_f"]
        bars = all_bars[ak]
        g = gates[ak]
        streams[(ak, "ORB")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: OpeningRangeBreakout(
                symbol=a, tick_size=tts, daily_filter=gg["orb"], **ORB_PARAMS), ak)
        streams[(ak, "MeanRev")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: MeanReversionBollinger(
                symbol=a, tick_size=tts, daily_filter=gg["mr"], **MR_PARAMS), ak)
    all_bars.clear()

    union_days = sorted(set().union(*[set(s.keys()) for s in streams.values()]))
    full_arrays = {k: to_series(pnl, union_days) for k, pnl in streams.items()}
    n_total = len(union_days)
    print(f"\nTotal history: {union_days[0]} -> {union_days[-1]} ({n_total} d)")

    remaining = n_total - MIN_IS_DAYS
    fold_size = remaining // N_FOLDS
    print(f"MIN_IS_DAYS={MIN_IS_DAYS}, {N_FOLDS} folds of ~{fold_size}d each\n")

    print(f"{'='*78}\nPER-FOLD RESULTS\n{'='*78}")
    all_oos_days: list[date] = []
    all_oos_vals: list[float] = []
    fold_weights_history = []

    for fold in range(N_FOLDS):
        is_end = MIN_IS_DAYS + fold_size * fold
        oos_end = MIN_IS_DAYS + fold_size * (fold + 1) if fold < N_FOLDS - 1 else n_total
        is_days_f = union_days[:is_end]
        oos_days_f = union_days[is_end:oos_end]
        if not oos_days_f:
            continue

        is_slice = {k: arr[:is_end] for k, arr in full_arrays.items()}
        oos_slice = {k: arr[is_end:oos_end] for k, arr in full_arrays.items()}

        is_pr30, is_ev_pos = {}, {}
        for k, arr in is_slice.items():
            rr30 = pass_rate_30(arr, is_days_f)
            is_pr30[k] = rr30.pass_rate
            is_ev_pos[k] = float(arr.mean()) > 0
        gated = {k: (v if is_ev_pos[k] else 0.0) for k, v in is_pr30.items()}
        total_pr = sum(gated.values())
        weights = ({k: v / total_pr for k, v in gated.items()} if total_pr > 0
                   else {k: 1.0 / len(is_pr30) for k in is_pr30})
        fold_weights_history.append(weights)

        def ensemble(arrays_dict, wt):
            z = sum(wt.values())
            n = next(iter(arrays_dict.values())).size
            out = np.zeros(n, dtype=float)
            for k, w in wt.items():
                out += (w / z) * arrays_dict[k]
            return out

        ens_is_f  = ensemble(is_slice, weights)
        ens_oos_f = ensemble(oos_slice, weights)
        is_sh = sharpe(ens_is_f)
        oos_sh = sharpe(ens_oos_f)
        rr30_oos_f = pass_rate_30(ens_oos_f, oos_days_f)

        top3 = sorted(weights.items(), key=lambda x: -x[1])[:3]
        top3_s = ", ".join(f"{k[0]}/{k[1]}={w:.2f}" for k, w in top3)
        print(f"Fold {fold+1}: IS={is_days_f[0]}->{is_days_f[-1]} ({len(is_days_f)}d)  "
              f"OOS={oos_days_f[0]}->{oos_days_f[-1]} ({len(oos_days_f)}d)")
        print(f"  weights(top3): {top3_s}")
        print(f"  IS Sharpe={is_sh:+.2f}  OOS Sharpe={oos_sh:+.2f}  "
              f"OOS pass30={rr30_oos_f.pass_rate:.1%}  OOS total=${ens_oos_f.sum():+,.0f}")

        all_oos_days.extend(oos_days_f)
        all_oos_vals.extend(ens_oos_f.tolist())

    print(f"\n{'='*78}\nCONCATENATED WALK-FORWARD OOS ({len(all_oos_days)} days total)\n{'='*78}")
    wf_oos_arr = np.array(all_oos_vals)
    wf_oos_daily = {d: Decimal(str(round(float(v), 2))) for d, v in zip(all_oos_days, wf_oos_arr)}
    wf_sh = sharpe(wf_oos_arr)
    rr30_wf = pass_rate_30(wf_oos_arr, all_oos_days)
    print(f"  Walk-forward OOS Sharpe={wf_sh:+.2f}  pass30={rr30_wf.pass_rate:.1%}  "
          f"total=${wf_oos_arr.sum():+,.0f}")

    seq = simulate_sequential_accounts(wf_oos_daily, rules=RULES,
                                        starting_balance=RULES.starting_balance,
                                        checkpoint=Decimal("1500"))
    print(f"\n  Sequential (walk-forward OOS): {seq.n_accounts} accounts, "
          f"{seq.count('pass')} pass ({seq.pass_rate:.1%})")

    t1 = _time.time()
    mc = monte_carlo_pass_rate(wf_oos_daily, rules=RULES, starting_balance=RULES.starting_balance,
                                n_sims=2000, block_len=10, checkpoint=Decimal("1500"), seed=42)
    print(f"  Monte Carlo (walk-forward OOS, {_time.time()-t1:.1f}s): "
          f"mean={mc.mean_pass_rate:.1%}  median={mc.median_pass_rate:.1%}  "
          f"90%CI=[{mc.p05:.1%},{mc.p95:.1%}]  P(>50%)={mc.prob_above_50pct:.1%}")

    print(f"\n{'='*78}\nWEIGHT STABILITY ACROSS FOLDS\n{'='*78}")
    all_keys = sorted(set().union(*[set(w.keys()) for w in fold_weights_history]))
    for k in all_keys:
        vals = [w.get(k, 0.0) for w in fold_weights_history]
        print(f"  {k[0]}/{k[1]:<10}: " + "  ".join(f"{v:.2f}" for v in vals) +
              f"   (std={np.std(vals):.3f})")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
