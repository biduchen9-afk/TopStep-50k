"""Autonomous loop, phase 1: ensemble combinations of the session's
near-miss candidates, evaluated efficiently against ONE shared load of
ES/NQ/GC bars and ONE computation of every candidate stream.

Motivation: ThreeDayReversal(low-vol gated), PowerHourContinuation, and
IntradayMomentumTP(vol-expansion gated) each failed solo on a NARROW
gate (t-stat or max_loss/avg_win -- a tail-risk/sample-size issue, not
a broad "no edge" failure) when tested one at a time (v12, and the
individual evaluate_*.py checklists). Combined together, their
individual tail-risk outliers may not coincide on the same days,
which would show up as reduced ensemble-level max_loss/avg_win and
improved t-stat even if no single one of them would pass alone --
exactly the mechanism that already let ORB clear Gate 2 only once
diversified. This tests every combination systematically instead of
guessing at one.

Streams computed ONCE per asset: ORB, MeanRev, ThreeDayRev(gated),
PowerHour, IntraMomTP(gated) -- 5 strategies x 3 assets = 15 streams.
Combinations evaluated: all non-empty subsets of the 3 "new" strategies
(ThreeDayRev, PowerHour, IntraMomTP) added on top of the ORB+MeanRev
base (7 combinations x full Gate 2/3/4 + DSR + correlation + sequential
+ Monte Carlo each).

Run with: python scripts/evaluate_ensemble_combinations.py
"""

from __future__ import annotations

import math
import sys
import time as _time
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.benchmark import buy_and_hold_daily_pnl, to_aligned_array
from topstep50k.analysis.correlation import print_stream_correlation_report, stream_correlation_report
from topstep50k.analysis.dsr import deflated_sharpe
from topstep50k.analysis.montecarlo import monte_carlo_pass_rate
from topstep50k.analysis.passrate import (
    realized_pass_rate,
    simulate_combine_window,
    simulate_sequential_accounts,
)
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.evaluation.harness import is_oos_split
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import (
    meanrev_low_vol_gate,
    orb_expansion_gate,
    per_day_session_stats,
)
from topstep50k.rules import combine_50k
from topstep50k.strategy.intraday_momentum_tp import IntradayMomentumTP
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout
from topstep50k.strategy.power_hour import PowerHourContinuation
from topstep50k.strategy.three_day_reversal import ThreeDayReversal


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

BASE_STRATS = ("ORB", "MeanRev")
NEW_STRATS = ("ThreeDayRev", "PowerHour", "IntraMomTP")

# All non-empty subsets of the 3 new strategies, on top of the ORB+MeanRev base.
COMBINATIONS = []
for r in range(1, len(NEW_STRATS) + 1):
    for subset in combinations(NEW_STRATS, r):
        COMBINATIONS.append(subset)

# Third-leg search trials from this whole session (see v12's script for the
# raw/gated screen numbers) -- used for DSR deflation on every combination
# tried here too, since these combinations are themselves additional trials
# on top of the 36 already searched.
THIRD_LEG_SEARCH_SHARPES = [
    -0.97, -0.16, -0.08, -2.51, +0.40, -0.10,
    -0.35, +0.54, +0.53, -1.76, -0.61, +0.06,
    -1.32, +0.04, +0.50, -1.40, +0.23, +0.25,
    -1.99, +0.01, -1.51, -0.63, -1.01, -0.05,
    -1.66, -0.86, +0.61, -0.10,
    -1.24, -0.56, +0.72, +0.35,
    -0.40, -1.69, +0.91, +0.32,
    # ORB-TrendHold (v10-adjacent research) + PowerHour solo screen
    +0.88, +1.01, +0.24, +0.44, +1.03, -0.25,
]


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
    rr30 = realized_pass_rate(pnl, rules=RULES, starting_balance=RULES.starting_balance,
                               window_days=30, stride_days=1)
    rr45 = realized_pass_rate(pnl, rules=RULES, starting_balance=RULES.starting_balance,
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
        r = simulate_combine_window(pnls, rules=RULES, starting_balance=RULES.starting_balance)
        cnt[r.outcome] += 1
    return cnt["mll_breach"] / total if total > 0 else 0.0


def gate_check(name: str, observed, threshold, hard: bool, note: str = "") -> bool:
    passed = observed > threshold if hard else observed >= threshold
    tick = "OK" if passed else ("FAIL" if hard else "~")
    kind = "HARD" if hard else "adv."
    obs_s = f"{observed:.4f}" if isinstance(observed, float) else str(observed)
    thr_s = f"{threshold:.4f}" if isinstance(threshold, float) else str(threshold)
    print(f"  [{tick:>4}] ({kind}) {name:<40} obs={obs_s:<10} thr={thr_s}  {note}")
    return passed


def evaluate_combination(strat_names, streams, union_days, is_mask, oos_mask,
                          is_days, oos_days, bh_basket, label):
    keys = [(ak, sn) for ak in ASSETS for sn in strat_names if (ak, sn) in streams]
    full_arrays = {k: to_series(streams[k], union_days) for k in keys}
    is_arrays  = {k: a[is_mask]  for k, a in full_arrays.items()}
    oos_arrays = {k: a[oos_mask] for k, a in full_arrays.items()}

    is_pr30, is_ev_positive = {}, {}
    for k, arr in is_arrays.items():
        rr30, _ = pass_rate_30_45(arr, is_days)
        is_pr30[k] = rr30.pass_rate
        is_ev_positive[k] = float(arr.mean()) > 0

    gated_pr30 = {k: (v if is_ev_positive[k] else 0.0) for k, v in is_pr30.items()}
    total_pr = sum(gated_pr30.values())
    weights = ({k: v / total_pr for k, v in gated_pr30.items()}
               if total_pr > 0 else {k: 1.0 / len(is_pr30) for k in is_pr30})

    def ensemble(arrays_dict, wt):
        z = sum(wt.values())
        if z <= 0:
            return np.zeros(next(iter(arrays_dict.values())).size)
        n = next(iter(arrays_dict.values())).size
        out = np.zeros(n, dtype=float)
        for k, w in wt.items():
            out += (w / z) * arrays_dict[k]
        return out

    ens_is  = ensemble(is_arrays,  weights)
    ens_oos = ensemble(oos_arrays, weights)

    is_sh = sharpe(ens_is)
    is_ev = float(ens_is.mean())
    is_pf = (ens_is[ens_is > 0].sum() / -ens_is[ens_is < 0].sum()
             if (ens_is < 0).any() else float("inf"))
    rr30_is, rr45_is = pass_rate_30_45(ens_is, is_days)
    n_is_years = len(is_days) / 252.0
    nz_is = int((ens_is != 0).sum())
    trades_per_year = nz_is / n_is_years if n_is_years > 0 else 0.0
    ts_ = ens_is.mean() / (ens_is.std(ddof=1) / math.sqrt(ens_is.size)) if ens_is.size > 1 else 0.0
    pos = ens_is[ens_is > 0]; neg = ens_is[ens_is < 0]
    avg_win = float(pos.mean()) if pos.size > 0 else 0.0
    max_loss_ratio = abs(float(neg.min())) / avg_win if (pos.size > 0 and neg.size > 0) else 0.0

    print(f"\n{'='*78}\nCOMBINATION: {label}\n{'='*78}")
    print("Weights:", {f"{k[0]}/{k[1]}": round(w, 3) for k, w in sorted(weights.items(), key=lambda x: -x[1])})

    corr_report = stream_correlation_report(is_arrays, threshold=0.6)
    if corr_report.high_corr_pairs:
        pairs_s = ", ".join(f"{a[0]}/{a[1]}<->{b[0]}/{b[1]}:{c:+.2f}"
                             for a, b, c in corr_report.high_corr_pairs)
        print(f"  Flagged correlation pairs: {pairs_s}")
    else:
        print("  No flagged correlation pairs.")

    g2_results = [
        gate_check("trades_per_year>=20", trades_per_year, 20.0, hard=True),
        gate_check("EV per day > 0", is_ev, 0.0, hard=True),
        gate_check("profit_factor > 1.0", is_pf, 1.0, hard=True),
        gate_check("t_stat (IS) >= 1.5", ts_, 1.5, hard=True),
        gate_check("sharpe_annual (IS) >= 0.3", is_sh, 0.3, hard=True),
        gate_check("IS pass30 >= 15%", rr30_is.pass_rate, 0.15, hard=True),
    ]
    gate_check("max_loss / avg_win < 3.0", max_loss_ratio, 3.0, hard=False)
    g2_passed = all(g2_results)

    dsr = deflated_sharpe(ens_is.tolist(), all_trial_sharpes_annual=THIRD_LEG_SEARCH_SHARPES)
    print(f"  DSR (deflated for {dsr.n_trials} trials): {dsr.dsr:.1%}  IS Sharpe={dsr.sharpe:+.2f}")

    if not g2_passed:
        print(f"  IS  pass30={rr30_is.pass_rate:.1%}  Sharpe={is_sh:+.2f}  -> RETIRED (failed Gate 2/3)")
        return {"label": label, "promoted": False, "dsr": dsr.dsr, "seq_pass_rate": None,
                "mc_mean": None, "mc_p_above_50": None}

    oos_sh = sharpe(ens_oos)
    oos_ev = float(ens_oos.mean())
    rr30_oos, rr45_oos = pass_rate_30_45(ens_oos, oos_days)
    sh_ratio = (oos_sh / is_sh) if (not math.isnan(is_sh) and is_sh > 0) else 0.0
    pr_ratio = (rr30_oos.pass_rate / rr30_is.pass_rate) if rr30_is.pass_rate > 0 else 0.0

    is_mll = mll_breach_rate(ens_is, is_days)
    oos_mll = mll_breach_rate(ens_oos, oos_days)
    mll_ok = math.isnan(oos_mll) or math.isnan(is_mll) or oos_mll <= is_mll * 1.5

    bh_oos_total = float(bh_basket[oos_mask].sum())
    rr30_bh, _ = pass_rate_30_45(bh_basket[oos_mask], oos_days)
    beats_bh = rr30_oos.pass_rate > rr30_bh.pass_rate

    g4_hard = [
        gate_check("OOS EV > 0 ($)", oos_ev, 0.0, hard=True),
        gate_check("OOS Sharpe >= 0.5xIS", sh_ratio, 0.50, hard=True,
                   note=f"OOS={oos_sh:.3f} IS={is_sh:.3f}"),
    ]
    gate_check("OOS pass30 >= 0.5xIS", pr_ratio, 0.50, hard=False)
    gate_check("OOS MLL rate <= 1.5xIS", mll_ok, True, hard=False)
    gate_check("OOS beats naive buy-and-hold pass30", beats_bh, True, hard=False,
               note=f"ens={rr30_oos.pass_rate:.1%} bh={rr30_bh.pass_rate:.1%}")

    g4_passed = (oos_ev > 0) and (sh_ratio >= 0.50)
    promoted = g2_passed and g4_passed

    print(f"  IS  pass30={rr30_is.pass_rate:.1%}  Sharpe={is_sh:+.2f}   "
          f"OOS pass30={rr30_oos.pass_rate:.1%}  Sharpe={oos_sh:+.2f}")
    print(f"  -> {'OOS_PROMOTED' if promoted else 'RETIRED (failed Gate 4)'}")

    seq_pass_rate = mc_mean = mc_p_above_50 = None
    if promoted:
        oos_daily = {d: Decimal(str(round(float(v), 2))) for d, v in zip(oos_days, ens_oos)}
        seq = simulate_sequential_accounts(oos_daily, rules=RULES,
                                            starting_balance=RULES.starting_balance,
                                            checkpoint=Decimal("1500"))
        mc = monte_carlo_pass_rate(oos_daily, rules=RULES, starting_balance=RULES.starting_balance,
                                    n_sims=2000, block_len=10, checkpoint=Decimal("1500"), seed=42)
        seq_pass_rate = seq.pass_rate
        mc_mean = mc.mean_pass_rate
        mc_p_above_50 = mc.prob_above_50pct
        print(f"  Sequential: {seq.n_accounts} accounts, {seq.count('pass')} pass "
              f"({seq_pass_rate:.1%})")
        print(f"  Monte Carlo: mean={mc_mean:.1%}  90%CI=[{mc.p05:.1%},{mc.p95:.1%}]  "
              f"P(>50%)={mc_p_above_50:.1%}")

    return {"label": label, "promoted": promoted, "dsr": dsr.dsr,
            "seq_pass_rate": seq_pass_rate, "mc_mean": mc_mean, "mc_p_above_50": mc_p_above_50,
            "oos_sharpe": oos_sh, "is_sharpe": is_sh}


def main():
    print("=" * 78)
    print("ENSEMBLE COMBINATIONS -- near-miss candidates, shared data load")
    print("=" * 78)

    t0 = _time.time()
    all_bars = {}
    gates = {}
    bh_daily = {}
    for ak in ASSETS:
        print(f"\nLoading {ak} bars...", flush=True)
        tl = _time.time()
        bars = [b for b in load_bars_csv(ASSETS[ak]["data_path"]) if b.ts >= RECENT_START]
        print(f"  {len(bars):,} bars in {_time.time()-tl:.1f}s", flush=True)
        all_bars[ak] = bars
        bh_daily[ak] = buy_and_hold_daily_pnl(bars, ASSETS[ak]["instrument"].point_value, qty=1)
        stats = per_day_session_stats(bars)
        gates[ak] = {
            "orb": orb_expansion_gate(stats),
            "mr":  meanrev_low_vol_gate(stats),
        }

    print("\nComputing all 5 strategies x 3 assets = 15 streams...", flush=True)
    streams: dict[tuple[str, str], dict] = {}
    for ak in ASSETS:
        ts = ASSETS[ak]["tick_size_f"]
        bars = all_bars[ak]
        g = gates[ak]
        t1 = _time.time()
        streams[(ak, "ORB")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: OpeningRangeBreakout(
                symbol=a, tick_size=tts, daily_filter=gg["orb"], **ORB_PARAMS), ak)
        streams[(ak, "MeanRev")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: MeanReversionBollinger(
                symbol=a, tick_size=tts, daily_filter=gg["mr"], **MR_PARAMS), ak)
        streams[(ak, "ThreeDayRev")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: ThreeDayReversal(
                symbol=a, tick_size=tts, daily_filter=gg["mr"]), ak)
        streams[(ak, "PowerHour")] = run_stream(
            bars, lambda tts=ts, a=ak: PowerHourContinuation(symbol=a, tick_size=tts), ak)
        streams[(ak, "IntraMomTP")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: IntradayMomentumTP(
                symbol=a, tick_size=tts, daily_filter=gg["orb"]), ak)
        print(f"  {ak}: 5 streams in {_time.time()-t1:.1f}s", flush=True)

    all_bars.clear()

    union_days = sorted(set().union(*[set(s.keys()) for s in streams.values()]))
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
    bh_basket = sum(to_aligned_array(bh_daily[ak], union_days) for ak in ASSETS)
    print(f"\nIS : {is_days[0]} -> {is_days[-1]} ({len(is_days)} d)")
    print(f"OOS: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d)")
    print(f"Streams computed in {_time.time()-t0:.0f}s total\n")

    results = []
    for new_subset in COMBINATIONS:
        strat_names = list(BASE_STRATS) + list(new_subset)
        label = "ORB+MeanRev+" + "+".join(new_subset)
        r = evaluate_combination(strat_names, streams, union_days, is_mask, oos_mask,
                                  is_days, oos_days, bh_basket, label)
        results.append(r)

    print(f"\n{'='*78}\nFINAL SUMMARY -- ALL COMBINATIONS\n{'='*78}")
    for r in results:
        if r["promoted"]:
            print(f"  {r['label']:<32} OOS_PROMOTED  DSR={r['dsr']:.1%}  "
                  f"seq_pass={r['seq_pass_rate']:.1%}  MC_mean={r['mc_mean']:.1%}  "
                  f"P(>50%)={r['mc_p_above_50']:.1%}")
        else:
            print(f"  {r['label']:<32} retired       DSR={r['dsr']:.1%}")

    promoted = [r for r in results if r["promoted"]]
    if promoted:
        best = max(promoted, key=lambda r: r["mc_mean"] or 0)
        print(f"\nBEST BY MONTE CARLO MEAN PASS RATE: {best['label']} "
              f"({best['mc_mean']:.1%}, DSR={best['dsr']:.1%})")
    else:
        print("\nNONE of the combinations reached OOS_PROMOTED.")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
