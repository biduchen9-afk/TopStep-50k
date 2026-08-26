"""9-stream regime-gated multi-asset ensemble:
ORB + MeanRev + ThreeDayReversal(gated) x ES + NQ + GC.

v12: replaces IntradayMomentumTP (v11, failed -- OOS Sharpe negative on
all 3 assets, a real overfitting signature) with ThreeDayReversal,
low-vol gated (meanrev_low_vol_gate, same gate as Bollinger MeanRev --
mean-reversion mechanism, same reasoning). Of everything screened this
session, gated ThreeDayReversal was the closest to solo promotion: on
its own it showed positive Sharpe AND profit_factor>1.0 on ALL THREE
assets simultaneously (Sharpe +0.61/+0.72/+0.91), and its solo Gate
2/3/4 failures were narrow (t_stat / max_loss-avg_win, driven by a
reduced sample from gating -- 61-66 trades/yr) rather than broad
multi-metric failures like IntraMomTP's. Same test that already worked
for ORB: does it clear the bar once diversified into a multi-stream
ensemble, not whether it stands alone (see evaluate_gated_candidates_v2.py
for the solo checklist that motivated this).

Adds two things v11 didn't have: a pairwise stream-correlation report
(analysis.correlation.stream_correlation_report) and a DSR check
(analysis.dsr.deflated_sharpe) on the ensemble's IS Sharpe, deflated
for the number of DISTINCT strategy candidates actually tried this
session (12: ORB raw/gated, MeanRev, OD-retired, IntraMomTP raw/gated,
GapFill raw/gated, VolProfileMR raw/gated(x2 incl. bugfix),
VwapStdFractal raw/gated, ThreeDayRev raw/gated -- counted at the
ensemble-construction decision, not per-asset). Direct response to the
concern that OD's decorrelation from the rest of the ensemble was
automatic (disjoint overnight time window) and every replacement
candidate trades the same RTH window, so decorrelation and overfitting
both need to be checked explicitly, not assumed from the strategies
looking mechanistically different on paper.

Same 70/30 IS/OOS harness, same causal regime gates, same EV-gated
pass-rate-aware weight derivation, same buy-and-hold Gate 4 check
(pass30 vs pass30, not $ vs $) as v1/v10/v11. Also includes the
sequential (non-overlapping) account simulation and Monte Carlo
block-bootstrap pass-rate resample in one run.

Run with: python scripts/evaluate_ensemble_databento_v12_threedayrev.py
"""

from __future__ import annotations

import math
import sys
import time as _time
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
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
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout
from topstep50k.strategy.three_day_reversal import ThreeDayReversal


# Every per-asset IS Sharpe actually observed while searching for a third
# leg to replace OD, across screen_new_daysession_strategies.py (raw),
# screen_vwap_gated_databento.py (pre-bugfix gated), and
# screen_gated_candidates_v2.py (post-bugfix gated) -- 6 strategies x 3
# assets raw, + 2 x 3 pre-bugfix-gated, + 4 x 3 post-bugfix-gated = 36
# trials. Used below for a DSR check on the ensemble's IS Sharpe: how
# much should it be deflated for how much searching actually happened
# before landing on ThreeDayReversal?
THIRD_LEG_SEARCH_SHARPES = [
    -0.97, -0.16, -0.08, -2.51, +0.40, -0.10,   # ES raw
    -0.35, +0.54, +0.53, -1.76, -0.61, +0.06,   # NQ raw
    -1.32, +0.04, +0.50, -1.40, +0.23, +0.25,   # GC raw
    -1.99, +0.01, -1.51, -0.63, -1.01, -0.05,   # ES/NQ/GC VolProfileMR+VwapFractal, pre-bugfix gated
    -1.66, -0.86, +0.61, -0.10,                  # ES post-bugfix gated (VolProfileMR/GapFill/ThreeDayRev/IntraMomTP)
    -1.24, -0.56, +0.72, +0.35,                  # NQ post-bugfix gated
    -0.40, -1.69, +0.91, +0.32,                  # GC post-bugfix gated
]


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
TDR_PARAMS = dict(qty=1)  # pre-committed defaults, no tuning (screen used the same)


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
    print(f"  [{tick:>4}] ({kind}) {name:<32} obs={obs_s:<10} thr={thr_s}  {note}")
    return passed


def main():
    print("=" * 72)
    print("9-STREAM ENSEMBLE v12 -- ThreeDayReversal(gated) replaces retired OD")
    print("Strategies: ORB + MeanRev + ThreeDayReversal(low-vol gated) x ES + NQ + GC")
    print(f"Bars filtered to ts >= {RECENT_START}")
    print("=" * 72)

    all_bars = {}
    gates = {}
    bh_daily = {}
    for ak in ASSETS:
        print(f"\nLoading {ak} bars...", flush=True)
        t0 = _time.time()
        bars = [b for b in load_bars_csv(ASSETS[ak]["data_path"]) if b.ts >= RECENT_START]
        print(f"  {len(bars):,} bars (>= {RECENT_START.date()}) in {_time.time() - t0:.1f}s",
              flush=True)
        all_bars[ak] = bars
        bh_daily[ak] = buy_and_hold_daily_pnl(bars, ASSETS[ak]["instrument"].point_value, qty=1)
        stats = per_day_session_stats(bars)
        gates[ak] = {
            "orb": orb_expansion_gate(stats),
            "mr":  meanrev_low_vol_gate(stats),
        }

    print("\nRunning 9 strategy-asset streams...", flush=True)
    streams: dict[tuple[str, str], dict] = {}
    for ak in ASSETS:
        ts = ASSETS[ak]["tick_size_f"]
        bars = all_bars[ak]
        g = gates[ak]
        t0 = _time.time()
        streams[(ak, "ORB")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: OpeningRangeBreakout(
                symbol=a, tick_size=tts, daily_filter=gg["orb"], **ORB_PARAMS), ak)
        streams[(ak, "MeanRev")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: MeanReversionBollinger(
                symbol=a, tick_size=tts, daily_filter=gg["mr"], **MR_PARAMS), ak)
        streams[(ak, "ThreeDayRev")] = run_stream(
            bars, lambda gg=g, tts=ts, a=ak: ThreeDayReversal(
                symbol=a, tick_size=tts, daily_filter=gg["mr"], **TDR_PARAMS), ak)
        print(f"  {ak}: 3 streams in {_time.time() - t0:.1f}s", flush=True)

    all_bars.clear()

    union_days = sorted(set().union(*[set(s.keys()) for s in streams.values()]))
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})
    bh_basket = sum(to_aligned_array(bh_daily[ak], union_days) for ak in ASSETS)
    print(f"\nIS : {is_days[0]} -> {is_days[-1]} ({len(is_days)} d)")
    print(f"OOS: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d)")

    full_arrays = {k: to_series(pnl, union_days) for k, pnl in streams.items()}
    is_arrays  = {k: a[is_mask]  for k, a in full_arrays.items()}
    oos_arrays = {k: a[oos_mask] for k, a in full_arrays.items()}

    print(f"\n{'─'*72}")
    print("PER-STREAM IS STATS (for weight derivation)")
    print(f"{'─'*72}")
    is_pr30 = {}
    is_ev_positive = {}
    for k, arr in is_arrays.items():
        rr30, _ = pass_rate_30_45(arr, is_days)
        is_pr30[k] = rr30.pass_rate
        is_ev_positive[k] = float(arr.mean()) > 0
        nz = int((arr != 0).sum())
        sh = sharpe(arr)
        print(f"  {k[0]}/{k[1]:<14}: nz={nz:>4} IS-total=${arr.sum():>+9,.0f} "
              f"Sharpe={sh:>+5.2f}  IS-pass30={rr30.pass_rate:.1%}"
              f"{'' if is_ev_positive[k] else '  [EV<=0 -> excluded]'}")

    gated_pr30 = {k: (v if is_ev_positive[k] else 0.0) for k, v in is_pr30.items()}
    total_pr = sum(gated_pr30.values())
    weights = ({k: v / total_pr for k, v in gated_pr30.items()}
               if total_pr > 0 else {k: 1.0 / len(is_pr30) for k in is_pr30})
    print(f"\nIS pass-rate-aware weights (EV-gated):")
    for k, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {k[0]}/{k[1]:<14}: {w:.3f}")

    # ── Pairwise stream correlation -- are these actually decorrelated? ───
    # OD's decorrelation was automatic (disjoint overnight time window).
    # ThreeDayReversal trades the same RTH window as ORB and MeanRev, so
    # that has to be checked, not assumed from the mechanisms looking
    # different on paper.
    print(f"\n{'─'*72}")
    print("PAIRWISE STREAM CORRELATION (IS, all 9 streams)")
    print(f"{'─'*72}")
    corr_report = stream_correlation_report(is_arrays, threshold=0.6)
    print_stream_correlation_report(corr_report)

    def ensemble(arrays_dict: dict, wt: dict) -> np.ndarray:
        z = sum(wt.values())
        n = next(iter(arrays_dict.values())).size
        out = np.zeros(n, dtype=float)
        for k, w in wt.items():
            out += (w / z) * arrays_dict[k]
        return out

    ens_is  = ensemble(is_arrays,  weights)
    ens_oos = ensemble(oos_arrays, weights)

    print(f"\n{'─'*72}")
    print("ENSEMBLE IS STATS (pass-rate-weighted)")
    print(f"{'─'*72}")
    is_sh = sharpe(ens_is)
    is_ev = float(ens_is.mean())
    is_pf = (ens_is[ens_is > 0].sum() / -ens_is[ens_is < 0].sum()
             if (ens_is < 0).any() else float("inf"))
    is_eq = np.cumsum(ens_is)
    is_mdd = float((is_eq - np.maximum.accumulate(is_eq)).min())
    rr30_is, rr45_is = pass_rate_30_45(ens_is, is_days)
    print(f"  IS total=${ens_is.sum():>+10,.0f}  Sharpe={is_sh:>+5.2f}  "
          f"EV=${is_ev:>+7.2f}/d  PF={is_pf:.2f}  MaxDD=${is_mdd:>+8,.0f}")
    print(f"  IS pass30={rr30_is.pass_rate:.1%} ({rr30_is.n_passed}/{rr30_is.n_windows})  "
          f"pass45={rr45_is.pass_rate:.1%} ({rr45_is.n_passed}/{rr45_is.n_windows})")

    # ── DSR: deflate the ensemble's IS Sharpe for how much searching it
    # took to find ThreeDayReversal as the third leg (36 trial Sharpes,
    # see THIRD_LEG_SEARCH_SHARPES above).
    dsr = deflated_sharpe(ens_is.tolist(), all_trial_sharpes_annual=THIRD_LEG_SEARCH_SHARPES)
    print(f"\n  DSR (deflated for {dsr.n_trials} third-leg search trials): "
          f"{dsr.dsr:.1%}  (E[max Sharpe|noise]={dsr.expected_max_sharpe:+.2f}, "
          f"IS Sharpe={dsr.sharpe:+.2f})")

    print(f"\n{'─'*72}")
    print("GATE 2/3 -- IS BASIC HEALTH + STATISTICAL QUALITY (ensemble)")
    print(f"{'─'*72}")
    n_is_years = len(is_days) / 252.0
    nz_is = int((ens_is != 0).sum())
    trades_per_year = nz_is / n_is_years if n_is_years > 0 else 0.0
    ts_ = ens_is.mean() / (ens_is.std(ddof=1) / math.sqrt(ens_is.size)) if ens_is.size > 1 else 0.0
    g2 = [
        gate_check("trades_per_year>=20",    trades_per_year, 20.0, hard=True),
        gate_check("EV per day > 0",          is_ev,           0.0,  hard=True),
        gate_check("profit_factor > 1.0",     is_pf,           1.0,  hard=True),
        gate_check("t_stat (IS) >= 1.5",      ts_,             1.5,  hard=True),
        gate_check("sharpe_annual (IS) >= 0.3", is_sh,         0.3,  hard=True),
        gate_check("IS pass30 >= 15%",        rr30_is.pass_rate, 0.15, hard=True),
    ]
    g2_passed = all(g2)

    print(f"\n{'─'*72}")
    print("ENSEMBLE OOS STATS (one-touch)")
    print(f"{'─'*72}")
    oos_sh  = sharpe(ens_oos)
    oos_ev  = float(ens_oos.mean())
    oos_pf  = (ens_oos[ens_oos > 0].sum() / -ens_oos[ens_oos < 0].sum()
               if (ens_oos < 0).any() else float("inf"))
    oos_eq  = np.cumsum(ens_oos)
    oos_mdd = float((oos_eq - np.maximum.accumulate(oos_eq)).min())
    rr30_oos, rr45_oos = pass_rate_30_45(ens_oos, oos_days)
    print(f"  OOS total=${ens_oos.sum():>+10,.0f}  Sharpe={oos_sh:>+5.2f}  "
          f"EV=${oos_ev:>+7.2f}/d  PF={oos_pf:.2f}  MaxDD=${oos_mdd:>+8,.0f}")
    print(f"  OOS pass30={rr30_oos.pass_rate:.1%} ({rr30_oos.n_passed}/{rr30_oos.n_windows})  "
          f"pass45={rr45_oos.pass_rate:.1%} ({rr45_oos.n_passed}/{rr45_oos.n_windows})")

    print(f"\n  Per-stream OOS breakdown:")
    for k in sorted(oos_arrays, key=lambda x: -is_pr30[x]):
        arr = oos_arrays[k]
        sh_k = sharpe(arr)
        rr_k, _ = pass_rate_30_45(arr, oos_days)
        print(f"    {k[0]}/{k[1]:<14}: total=${arr.sum():>+9,.0f}  "
              f"Sharpe={sh_k:>+5.2f}  OOS-pass30={rr_k.pass_rate:.1%}  w={weights[k]:.3f}")

    print(f"\nComputing MLL breach rates...", flush=True)
    t0 = _time.time()
    is_mll  = mll_breach_rate(ens_is,  is_days)
    oos_mll = mll_breach_rate(ens_oos, oos_days)
    print(f"  IS MLL breach rate  = {is_mll:.1%}")
    print(f"  OOS MLL breach rate = {oos_mll:.1%}  ({_time.time() - t0:.1f}s)")

    print(f"\n{'='*72}")
    print("GATE 4 -- OOS VALIDITY")
    print(f"{'='*72}")
    if not g2_passed:
        print("  Gate 2/3 already FAILED on IS -- Gate 4 is informational only below.")
    sh_ratio = (oos_sh / is_sh) if (not math.isnan(is_sh) and is_sh > 0) else 0.0
    pr_ratio = (rr30_oos.pass_rate / rr30_is.pass_rate) if rr30_is.pass_rate > 0 else 0.0
    mll_ok   = math.isnan(oos_mll) or math.isnan(is_mll) or oos_mll <= is_mll * 1.5

    g4_hard = [
        gate_check("OOS EV > 0 ($)",        oos_ev,   0.0,  hard=True),
        gate_check("OOS Sharpe >= 0.5xIS",  sh_ratio, 0.50, hard=True,
                   note=f"OOS={oos_sh:.3f} IS={is_sh:.3f}"),
    ]
    bh_oos_total = float(bh_basket[oos_mask].sum())
    rr30_bh, _ = pass_rate_30_45(bh_basket[oos_mask], oos_days)
    beats_bh = rr30_oos.pass_rate > rr30_bh.pass_rate
    g4_adv = [
        gate_check("OOS pass30 >= 0.5xIS",  pr_ratio, 0.50, hard=False,
                   note=f"OOS={rr30_oos.pass_rate:.1%} IS={rr30_is.pass_rate:.1%}"),
        gate_check("OOS MLL rate <= 1.5xIS", mll_ok,  True, hard=False,
                   note=f"IS={is_mll:.1%} OOS={oos_mll:.1%}"),
        gate_check("OOS pass30 beats naive buy-and-hold pass30", beats_bh, True, hard=False,
                   note=(f"ensemble pass30={rr30_oos.pass_rate:.1%} vs "
                         f"naive 1x ES+NQ+GC pass30={rr30_bh.pass_rate:.1%} "
                         f"(buy&hold $ total=${bh_oos_total:+,.0f}, unbounded risk)")),
    ]

    if g2_passed and all(g4_hard):
        promoted = "OOS_PROMOTED"
    elif g2_passed:
        promoted = "RETIRED (failed Gate 4)"
    else:
        promoted = "RETIRED (failed Gate 2/3)"

    print(f"\n{'='*72}")
    print(f"FINAL VERDICT: >>> {promoted} <<<")
    print(f"{'='*72}")
    print(f"  IS  period: {is_days[0]}  -> {is_days[-1]}  ({len(is_days)} d)")
    print(f"  OOS period: {oos_days[0]} -> {oos_days[-1]} ({len(oos_days)} d)")
    print(f"  IS  pass30={rr30_is.pass_rate:.1%}   pass45={rr45_is.pass_rate:.1%}   Sharpe={is_sh:+.2f}")
    print(f"  OOS pass30={rr30_oos.pass_rate:.1%}   pass45={rr45_oos.pass_rate:.1%}   Sharpe={oos_sh:+.2f}")
    print(f"  OOS/IS Sharpe ratio = {sh_ratio:.2f}  (>=0.50 required)")
    print(f"  OOS/IS pass30 ratio = {pr_ratio:.2f}  (>=0.50 advisory)")
    print(f"  IS MLL rate={is_mll:.1%}  OOS MLL rate={oos_mll:.1%}")
    print(f"  OOS ensemble pass30={rr30_oos.pass_rate:.1%}  vs  "
          f"naive 1x ES+NQ+GC buy-and-hold OOS pass30={rr30_bh.pass_rate:.1%}  -> "
          f"ensemble {'BEATS' if beats_bh else 'LOSES TO'} naive pass rate")

    # ── Sequential (non-overlapping) account simulation, OOS only ─────────
    print(f"\n{'='*72}")
    print("SEQUENTIAL ACCOUNT SIMULATION (OOS only, one account at a time)")
    print(f"{'='*72}")
    oos_daily = {d: Decimal(str(round(float(v), 2))) for d, v in zip(oos_days, ens_oos)}
    seq = simulate_sequential_accounts(oos_daily, rules=RULES,
                                        starting_balance=RULES.starting_balance,
                                        checkpoint=Decimal("1500"))
    print(f"  {seq.n_accounts} accounts opened over the {len(oos_days)}-day OOS period\n")
    for a in seq.accounts:
        cp = " [reached $1,500+]" if a.reached_checkpoint else ""
        print(f"  #{a.account_no:<2} {a.start_day} -> {a.end_day} ({a.n_days:>3}d)  "
              f"{a.outcome:<17} peak=${a.peak_profit:>+8,.0f}  final=${a.final_pnl:>+8,.0f}{cp}")
    print(f"\n  Outcome breakdown:")
    for outcome in ("pass", "mll_breach", "consistency_fail", "no_target"):
        n = seq.count(outcome)
        pct = n / seq.n_accounts * 100 if seq.n_accounts else 0
        print(f"    {outcome:<17}: {n:>2}/{seq.n_accounts}  ({pct:.0f}%)")
    print(f"\n  Unconditional pass rate (fresh account): {seq.pass_rate:.1%}")
    n_cp = len(seq.checkpoint_accounts)
    print(f"  Accounts that ever reached $1,500+ profit: {n_cp}/{seq.n_accounts}")
    print(f"  Of those, went on to actually pass: {seq.checkpoint_pass_rate:.1%}")

    # ── Monte Carlo block-bootstrap resample of the sequential pass rate ──
    print(f"\n{'='*72}")
    print("MONTE CARLO -- BLOCK-BOOTSTRAP RESAMPLE OF THE SEQUENTIAL PASS RATE")
    print(f"{'='*72}")
    t0 = _time.time()
    mc = monte_carlo_pass_rate(oos_daily, rules=RULES, starting_balance=RULES.starting_balance,
                                n_sims=2000, block_len=10, checkpoint=Decimal("1500"), seed=42)
    print(f"  {mc.n_sims} resamples of the {mc.n_days}-day OOS series "
          f"(block_len={mc.block_len}) in {_time.time()-t0:.1f}s")
    print(f"  single realized-path pass rate : {seq.pass_rate:.1%}  ({seq.count('pass')}/{seq.n_accounts})")
    print(f"  Monte Carlo mean pass rate     : {mc.mean_pass_rate:.1%}")
    print(f"  Monte Carlo median pass rate   : {mc.median_pass_rate:.1%}")
    print(f"  Monte Carlo 90% interval       : [{mc.p05:.1%}, {mc.p95:.1%}]")
    print(f"  P(resampled pass rate > 50%)   : {mc.prob_above_50pct:.1%}")
    print(f"  avg accounts per resample      : {mc.mean_n_accounts:.1f}")


if __name__ == "__main__":
    main()
