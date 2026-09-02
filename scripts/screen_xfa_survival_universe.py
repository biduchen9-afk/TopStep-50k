"""Re-screen the full legal day-session strategy universe against the
criterion the XFA-phase risk analysis actually calls for: funded-account
SURVIVAL probability (Monte Carlo, via the trailing-floor lifecycle
sim), not Combine pass-rate and not raw Sharpe.

Why this is a different screen from everything else in this session:
evaluate_xfa_sizing_overlays.py's sensitivity sweep showed breach
probability against a trailing floor tracks the classical reflected-
random-walk ruin formula P(breach) ~ exp(-2*mu*D/sigma^2) -- governed
by mean/VARIANCE, not Sharpe (mean/std) alone. Since portfolio mean is
additive across independent streams but portfolio VARIANCE shrinks
with genuine decorrelation, the lever this screen is built around is
"which combination of streams minimizes portfolio sigma^2 for a given
portfolio mu" -- not "which single strategy has the best pass30 rate,"
which was every prior screen's criterion.

Universe: every day-session (RTH, flat-before-close, no overnight
hold) strategy built this session, x ES/NQ/GC = up to 27 streams.
EXCLUDED on legality grounds (TopStep disallows overnight holding at
every account stage -- see docs/rules_sources.md): OvernightDrift
(already retired), MacroOvernightDrift, PreFomcDrift (explicitly
overnight), TrendChannel (max_hold_days=15, carries positions across
sessions). SweepMSS is a different instrument/session-window port
(Bangkok-session Pine Script strategy) not adapted to this codebase's
RTH/no-overnight framework -- out of scope here, not re-litigated.

Per-strategy parameters are the SAME pre-committed values already used
elsewhere this session (ORB/MeanRev from live_portfolio.py; ThreeDayRev
low-vol-gated, PowerHourContinuation, IntraMomTP orb-gated from
evaluate_ensemble_combinations.py; GapFill from evaluate_gap_fill.py;
VolumeProfileMeanReversion/VwapStdFractal defaults from
screen_new_daysession_strategies.py) -- no new parameter tuning here,
this changes the SELECTION criterion, not the candidate strategies
themselves.

Two stages:
  1. Solo stats for all 27 streams: mean/std/Sharpe, analytic ruin
     exponent, and a Monte Carlo standalone survival probability.
  2. Greedy MINIMUM-VARIANCE (inverse-solo-variance-weighted) forward
     selection over the Sharpe>0 subset, scored by ensemble XFA
     survival (Monte Carlo, ground truth -- the analytic formula is an
     approximation used for intuition/cross-checking only). Two earlier
     weighting choices were tried and rejected here, in order: (a)
     naive equal-DOLLAR weighting collapsed to ~0% survival by round 2
     -- the pool's daily-std spans a ~30x range (VwapFractal ~$50-230/
     day vs. IntraMom/ORB ~$700-4,600/day), so a high-vol stream just
     swamps a low-vol one the instant it's added; (b) inverse-STD
     ("risk parity" in the equal-risk-contribution sense) collapsed
     almost as fast, because equal-risk-CONTRIBUTION weights give every
     added stream the SAME variance contribution as what's already in
     the portfolio, so total variance grows with portfolio size instead
     of shrinking -- the right tool for budgeting risk across sleeves
     of similar importance, not for minimizing total variance, which is
     the actual goal (sigma^2 sits in the ruin exponent's denominator).
     Inverse-VARIANCE weighting (1/std^2, normalized) is the classical
     closed-form minimizer of portfolio variance for independent
     series -- it can only match or beat the quietest single stream.
     All three weighting schemes are stream-level statistics, not
     parameters fit on any particular combination.

Run with: python scripts/screen_xfa_survival_universe.py
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

from topstep50k.analysis.xfa_economics import monte_carlo_xfa_economics, xfa_full_size
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.regime import meanrev_low_vol_gate, orb_expansion_gate, per_day_session_stats
from topstep50k.rules import combine_50k
from topstep50k.rules.topstep_xfa import xfa_50k
from topstep50k.strategy.gap_fill import GapFill
from topstep50k.strategy.intraday_momentum import IntradayMomentum
from topstep50k.strategy.intraday_momentum_tp import IntradayMomentumTP
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout
from topstep50k.strategy.power_hour import PowerHourContinuation
from topstep50k.strategy.three_day_reversal import ThreeDayReversal
from topstep50k.strategy.volume_profile_mr import VolumeProfileMeanReversion
from topstep50k.strategy.vwap_std_fractal import VwapStdFractal

ASSETS = {
    "ES": {"instrument": Instrument(symbol="ES", point_value=Decimal("50"), tick_size=Decimal("0.25"),
                                     commission_per_side=Decimal("2.50")),
           "tick_size_f": 0.25, "data_path": ROOT / "data" / "raw" / "es_databento.txt"},
    "NQ": {"instrument": Instrument(symbol="NQ", point_value=Decimal("20"), tick_size=Decimal("0.25"),
                                     commission_per_side=Decimal("2.50")),
           "tick_size_f": 0.25, "data_path": ROOT / "data" / "raw" / "nq_databento.txt"},
    "GC": {"instrument": Instrument(symbol="GC", point_value=Decimal("100"), tick_size=Decimal("0.10"),
                                     commission_per_side=Decimal("2.50")),
           "tick_size_f": 0.10, "data_path": ROOT / "data" / "raw" / "gc_databento.txt"},
}
RULES = combine_50k()
XFA = xfa_50k()
RECENT_START = datetime(2021, 12, 31, tzinfo=timezone.utc)

ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both", stop_mode="opposite_range",
                   stop_ticks=40, tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15, time_stop_minutes=45)
GAP_PARAMS = dict(qty=1, gap_threshold_pct=0.0015, max_gap_pct=0.0150, stop_multiple=1.0,
                   time_stop_minutes=90, flat_before_close_minutes=15)

XFA_HORIZON_DAYS = 252
N_SIMS_SCREEN = 300
N_SIMS_FINAL = 2000
BLOCK_LEN = 10
SEED = 42
MAX_PORTFOLIO_SIZE = 8


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


def xfa_mc(values: np.ndarray, n_sims: int):
    daily = [Decimal(str(round(float(v), 2))) for v in values]
    return monte_carlo_xfa_economics(daily, xfa=XFA, horizon_days=XFA_HORIZON_DAYS,
                                      n_sims=n_sims, block_len=BLOCK_LEN, seed=SEED,
                                      sizing_fn=xfa_full_size)


def main():
    print("=" * 96)
    print("XFA SURVIVAL SCREEN -- full legal strategy universe, scored by breach risk not pass-rate")
    print("=" * 96)
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

    STRATS = {
        "ORB":        lambda a, ts, g: OpeningRangeBreakout(symbol=a, tick_size=ts, daily_filter=g["orb"], **ORB_PARAMS),
        "MeanRev":    lambda a, ts, g: MeanReversionBollinger(symbol=a, tick_size=ts, daily_filter=g["mr"], **MR_PARAMS),
        "ThreeDayRev": lambda a, ts, g: ThreeDayReversal(symbol=a, tick_size=ts, daily_filter=g["mr"]),
        "PowerHour":  lambda a, ts, g: PowerHourContinuation(symbol=a, tick_size=ts),
        "IntraMomTP": lambda a, ts, g: IntradayMomentumTP(symbol=a, tick_size=ts, daily_filter=g["orb"]),
        "IntraMom":   lambda a, ts, g: IntradayMomentum(symbol=a, tick_size=ts),
        "GapFill":    lambda a, ts, g: GapFill(symbol=a, tick_size=ts, daily_filter=None, **GAP_PARAMS),
        "VolProfileMR": lambda a, ts, g: VolumeProfileMeanReversion(symbol=a, tick_size=ts),
        "VwapFractal": lambda a, ts, g: VwapStdFractal(symbol=a, tick_size=ts),
    }

    print(f"\nBuilding {len(STRATS)} strategies x {len(ASSETS)} assets = "
          f"{len(STRATS)*len(ASSETS)} streams...", flush=True)
    t1 = _time.time()
    streams = {}
    for sk, build_fn in STRATS.items():
        for ak in ASSETS:
            ts = ASSETS[ak]["tick_size_f"]
            g = gates[ak]
            streams[(sk, ak)] = run_stream(all_bars[ak], lambda b=build_fn, a=ak, tts=ts, gg=g: b(a, tts, gg), ak)
    all_bars.clear()
    print(f"  built in {_time.time()-t1:.0f}s")

    union_days = sorted(set().union(*[set(s.keys()) for s in streams.values()]))
    arrays = {k: to_series(pnl, union_days) for k, pnl in streams.items()}
    n_days = len(union_days)
    print(f"\nFull history: {union_days[0]} -> {union_days[-1]} ({n_days} d).  "
          f"Setup: {_time.time()-t0:.0f}s")

    # ── STAGE 1: solo stats per stream ─────────────────────────────────
    print(f"\n{'='*96}\nSTAGE 1 -- solo stream stats  [{N_SIMS_SCREEN} sims each]\n{'='*96}")
    print(f"{'stream':<22}{'mean/d':>9}{'std/d':>10}{'Sharpe':>8}{'ruin-exp':>10}{'survive':>10}{'income':>10}")
    t1 = _time.time()
    solo = {}
    for k, arr in arrays.items():
        mean, std = arr.mean(), arr.std()
        sharpe = mean / std * np.sqrt(252) if std > 0 else 0.0
        ruin_exp = 2 * mean * float(XFA.mll_distance) / (std ** 2) if std > 0 else 0.0
        r = xfa_mc(arr, N_SIMS_SCREEN)
        solo[k] = dict(mean=mean, std=std, sharpe=sharpe, ruin_exp=ruin_exp,
                        survive=r.prob_survive, income=r.mean_income)
        print(f"{k[0]+'/'+k[1]:<22}{mean:>9.1f}{std:>10.1f}{sharpe:>8.2f}{ruin_exp:>10.3f}"
              f"{r.prob_survive:>9.1%} ${r.mean_income:>8,.0f}")
    print(f"  ({_time.time()-t1:.0f}s)")

    viable = [k for k, v in solo.items() if v["sharpe"] > 0]
    viable.sort(key=lambda k: -solo[k]["survive"])
    print(f"\n{len(viable)}/{len(solo)} streams have positive Sharpe -- candidate pool for diversification.")

    def risk_parity_combine(keys: list) -> np.ndarray:
        """MINIMUM-VARIANCE combination: weight each stream inversely
        to its OWN solo daily VARIANCE (1/std^2), normalized to sum to
        1. This is the classical closed-form minimizer of Var(sum w_i
        X_i) subject to sum w_i = 1 for independent series -- exactly
        what the ruin exponent 2*mu*D/sigma^2 rewards, since it's
        sigma^2 in the denominator, not sigma.

        A first pass here used naive equal-DOLLAR weights and collapsed
        to ~0% survival by round 2 (a high-vol stream swamping the one
        low-vol stream). A second pass used inverse-STD weights ("risk
        parity" in the equal-risk-contribution sense) and STILL
        collapsed almost as fast -- because equal-risk-contribution
        weights literally give each added stream the SAME variance
        contribution as the ones already in the portfolio, so total
        variance grows roughly linearly with portfolio size instead of
        shrinking. That's the right tool for risk-budgeting across
        similarly-important sleeves, not for minimizing total variance,
        which is what actually matters here. Inverse-variance weighting
        is the one that can only ever match or beat the quietest single
        stream (for independent series) -- both are stream-level
        statistics, not parameters fit on any particular combination.
        """
        inv_var = np.array([1.0 / solo[k]["std"] ** 2 for k in keys])
        w = inv_var / inv_var.sum()
        return sum(w[i] * arrays[k] for i, k in enumerate(keys))

    # ── STAGE 2: greedy risk-parity forward selection ───────────────────
    print(f"\n{'='*96}\nSTAGE 2 -- greedy minimum-variance (inverse-variance-weighted) diversification search  "
          f"[{N_SIMS_SCREEN} sims/step]\n{'='*96}")
    t1 = _time.time()
    portfolio: list = []
    remaining = list(viable)
    trajectory = []
    for round_no in range(1, min(MAX_PORTFOLIO_SIZE, len(viable)) + 1):
        best_candidate, best_result = None, None
        for cand in remaining:
            trial = portfolio + [cand]
            combined = risk_parity_combine(trial)
            r = xfa_mc(combined, N_SIMS_SCREEN)
            if best_result is None or r.prob_survive > best_result.prob_survive:
                best_candidate, best_result = cand, r
        portfolio.append(best_candidate)
        remaining.remove(best_candidate)
        trajectory.append((round_no, best_candidate, best_result.prob_survive, best_result.mean_income))
        print(f"  round {round_no}: +{best_candidate[0]+'/'+best_candidate[1]:<20} "
              f"portfolio={[k[0]+'/'+k[1] for k in portfolio]}")
        print(f"           survive={best_result.prob_survive:6.1%}  mean_income=${best_result.mean_income:>7,.0f}")
    print(f"  ({_time.time()-t1:.0f}s)")

    best_round = max(trajectory, key=lambda t: t[2])
    best_size = best_round[0]
    best_portfolio = portfolio[:best_size]
    print(f"\nBest point on the trajectory: {best_size} streams, "
          f"survive={best_round[2]:.1%}, mean_income=${best_round[3]:,.0f}")
    print(f"  Composition: {[k[0]+'/'+k[1] for k in best_portfolio]}")

    # ── FINAL CONFIRMATION at higher n_sims ─────────────────────────────
    print(f"\n{'='*96}\nFINAL CONFIRMATION  [{N_SIMS_FINAL} sims]\n{'='*96}")
    combined = risk_parity_combine(best_portfolio)
    mean, std = combined.mean(), combined.std()
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0.0
    r = xfa_mc(combined, N_SIMS_FINAL)
    print(f"  Best diversified portfolio: {[k[0]+'/'+k[1] for k in best_portfolio]}")
    print(f"  mean/day=${mean:.1f}  std/day=${std:.1f}  Sharpe(ann)={sharpe:.2f}")
    print(f"  survive={r.prob_survive:.1%}  mean_income=${r.mean_income:,.0f}  "
          f"median=${r.median_income:,.0f}  payouts/yr={r.mean_n_payouts:.2f}")

    print(f"\nTotal wall time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
