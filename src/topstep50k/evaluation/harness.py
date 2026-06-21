"""Strategy promotion checklist — 70/30 IS/OOS framework.

Each new strategy is evaluated through 6 gates in order. Gates are
enforced mechanically; no human override. DSR trial count is tracked
across the project: every strategy that reaches Gate 4 (OOS evaluated)
adds one trial, regardless of outcome.

Gate sequence
─────────────
  Gate 1  Theory (verified by human before running harness)
  Gate 2  IS basic health  — 6 hard checks
  Gate 3  IS statistical quality — 2 hard checks + 1 advisory
  Gate 4  OOS validity (ONE-TOUCH) — 2 hard checks + 2 advisory
  Gate 5  Ensemble value — 4 advisory (influence weighting, not gate)
  Gate 6  Regime fit — descriptive/placement only

Promotion levels
────────────────
  retired        : failed Gate 2 or Gate 3 (hard checks)
  is_only        : passed Gate 2+3 but Gate 4 not yet run
  oos_promoted   : passed Gate 2+3+4 — enters ensemble candidate pool
  fully_promoted : also passed Gate 5 — added to routing table
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Sequence

import numpy as np

from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.rules.topstep import TopstepRules


IS_FRACTION = 0.70
IS_HARD_GATES = {
    "trades_per_year":     (">=", 20.0),
    "ev_per_day":          (">",   0.0),
    "profit_factor":       (">",   1.0),
    "t_stat":              (">=",  1.5),
    "sharpe_annual":       (">=",  0.3),
    "max_loss_vs_avg_win": ("<",   3.0),  # |max_single_loss| / avg_win < 3
}
IS_HARD_PASS_RATE      = 0.15   # Gate 3: IS pass30 >= 15%
IS_PASS_RATE_FRAC      = 0.50   # Gate 3: IS pass30 >= 50% of best existing
OOS_EV_REQUIRED        = True   # Gate 4 hard
OOS_SHARPE_FRAC        = 0.50   # Gate 4 hard: OOS Sharpe >= 50% of IS Sharpe
OOS_PASS_RATE_FRAC     = 0.50   # Gate 4 advisory
ENSEMBLE_CORR_MAX      = 0.60   # Gate 5 advisory


@dataclass
class GateResult:
    name: str
    passed: bool
    observed: float | None
    threshold: float | None
    hard: bool = True
    note: str = ""

    def __str__(self) -> str:
        tick = "✓" if self.passed else ("✗" if self.hard else "~")
        obs = f"{self.observed:.3f}" if self.observed is not None else "—"
        thr = f"{self.threshold:.3f}" if self.threshold is not None else "—"
        return f"  [{tick}] {self.name:<30} obs={obs:<10} thr={thr}  {self.note}"


@dataclass
class ChecklistResult:
    label: str
    is_days: int
    oos_days: int
    gate2: list[GateResult] = field(default_factory=list)
    gate3: list[GateResult] = field(default_factory=list)
    gate4: list[GateResult] = field(default_factory=list)
    gate5: list[GateResult] = field(default_factory=list)
    promoted: str = "pending"   # retired | is_only | oos_promoted | fully_promoted
    is_pass30: float = 0.0
    oos_pass30: float = 0.0
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0

    def print_report(self) -> None:
        print(f"\n{'='*72}")
        print(f"STRATEGY CHECKLIST: {self.label}")
        print(f"IS={self.is_days}d  OOS={self.oos_days}d  "
              f"outcome=>>> {self.promoted.upper()} <<<")
        print(f"{'='*72}")
        for gate_num, gate_list in [("2 IS basic health", self.gate2),
                                     ("3 IS quality",     self.gate3),
                                     ("4 OOS validity",   self.gate4),
                                     ("5 Ensemble value", self.gate5)]:
            if not gate_list:
                continue
            print(f"\nGate {gate_num}:")
            for g in gate_list:
                print(str(g))


def _sharpe(arr: np.ndarray) -> float:
    if arr.size < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    return float((arr.mean() / arr.std(ddof=1)) * np.sqrt(252))


def _profit_factor(arr: np.ndarray) -> float:
    pos = arr[arr > 0].sum()
    neg = -arr[arr < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")


def _t_stat(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 0.0
    se = arr.std(ddof=1) / math.sqrt(arr.size)
    return float(arr.mean() / se) if se > 0 else 0.0


def _block_bootstrap_ci_lower(arr: np.ndarray, n_samples: int = 1000,
                               block_len: int = 10,
                               pct: float = 2.5) -> float:
    """95% lower bound on the mean via stationary block bootstrap."""
    rng = np.random.default_rng(42)
    n = arr.size
    means = np.empty(n_samples)
    blocks = max(1, n // block_len)
    for i in range(n_samples):
        starts = rng.integers(0, n, size=blocks)
        sample = np.concatenate([
            arr[s : s + block_len] if s + block_len <= n else arr[s:]
            for s in starts
        ])[:n]
        means[i] = sample.mean()
    return float(np.percentile(means, pct))


def _pass_rate_30(arr: np.ndarray, day_index: Sequence[date],
                   rules: TopstepRules) -> float:
    if arr.size < 30:
        return 0.0
    pnl = {d: Decimal(str(round(float(v), 2)))
            for d, v in zip(day_index, arr)}
    rr = realized_pass_rate(pnl, rules=rules,
                             starting_balance=rules.starting_balance,
                             window_days=30, stride_days=1)
    return rr.pass_rate


def is_oos_split(
    union_days: list[date],
    arrays: dict,
    is_fraction: float = IS_FRACTION,
) -> tuple[np.ndarray, np.ndarray, list[date], list[date]]:
    """Return (is_mask, oos_mask, is_days, oos_days) from the full day index."""
    n = len(union_days)
    is_n = int(n * is_fraction)
    is_mask = np.zeros(n, dtype=bool)
    is_mask[:is_n] = True
    oos_mask = ~is_mask
    is_days = union_days[:is_n]
    oos_days = union_days[is_n:]
    return is_mask, oos_mask, is_days, oos_days


def evaluate_strategy(
    label: str,
    daily_full: np.ndarray,          # full-history daily PnL (len = all days)
    is_mask: np.ndarray,
    oos_mask: np.ndarray,
    is_days: list[date],
    oos_days: list[date],
    rules: TopstepRules,
    best_existing_is_pass30: float = 0.0,
    existing_is_arrays: dict[str, np.ndarray] | None = None,
    run_oos: bool = True,
) -> ChecklistResult:
    """Run the full 6-gate checklist. Gate 4 (OOS) is optional (run_oos=True
    by default; set False if you want IS-only screening first).

    `daily_full` must be aligned to the same union day index used to
    produce is_mask / oos_mask.
    """
    is_arr  = daily_full[is_mask]
    oos_arr = daily_full[oos_mask]
    n_is_years = len(is_days) / 252.0
    result = ChecklistResult(label=label, is_days=len(is_days),
                              oos_days=len(oos_days))

    # ------------------------------------------------------------------ Gate 2
    nz_is = int((is_arr != 0).sum())
    trades_per_year = nz_is / n_is_years if n_is_years > 0 else 0.0
    ev = float(is_arr.mean())
    pf = _profit_factor(is_arr)
    ts = _t_stat(is_arr)
    sh = _sharpe(is_arr)
    pos = is_arr[is_arr > 0]
    neg = is_arr[is_arr < 0]
    avg_win = float(pos.mean()) if pos.size > 0 else 0.0
    max_loss_ratio = abs(float(neg.min())) / avg_win if (pos.size > 0 and neg.size > 0) else 0.0

    g2 = [
        GateResult("trades_per_year",     trades_per_year >= 20,     trades_per_year, 20.0),
        GateResult("EV per day ($)",       ev > 0,                   ev,              0.0),
        GateResult("profit_factor",        pf > 1.0,                 pf,              1.0),
        GateResult("t_stat (IS)",          ts >= 1.5,                ts,              1.5),
        GateResult("sharpe_annual (IS)",   sh >= 0.3,                sh,              0.3),
        GateResult("max_loss / avg_win",   max_loss_ratio < 3.0,     max_loss_ratio,  3.0,
                   note="(lower is better)"),
    ]
    result.gate2 = g2
    result.is_sharpe = sh
    g2_passed = all(g.passed for g in g2 if g.hard)
    if not g2_passed:
        result.promoted = "retired"
        return result

    # ------------------------------------------------------------------ Gate 3
    ci_lower = _block_bootstrap_ci_lower(is_arr)
    is_pr30 = _pass_rate_30(is_arr, is_days, rules)
    result.is_pass30 = is_pr30
    pr_vs_existing = (is_pr30 / best_existing_is_pass30
                      if best_existing_is_pass30 > 0 else float("inf"))

    g3 = [
        GateResult("bootstrap CI lower > 0", ci_lower > 0, ci_lower, 0.0,
                   note="block-10, 1000 samples"),
        GateResult("IS pass30 >= 15%",   is_pr30 >= IS_HARD_PASS_RATE,
                   is_pr30, IS_HARD_PASS_RATE),
        GateResult("IS pass30 vs best",  pr_vs_existing >= IS_PASS_RATE_FRAC,
                   pr_vs_existing, IS_PASS_RATE_FRAC, hard=False,
                   note=f"best_existing={best_existing_is_pass30:.1%}"),
    ]
    result.gate3 = g3
    g3_passed = all(g.passed for g in g3 if g.hard)
    if not g3_passed:
        result.promoted = "retired"
        return result

    result.promoted = "is_only"
    if not run_oos:
        return result

    # ------------------------------------------------------------------ Gate 4
    oos_ev    = float(oos_arr.mean())
    oos_sh    = _sharpe(oos_arr)
    oos_pr30  = _pass_rate_30(oos_arr, oos_days, rules)
    result.oos_pass30 = oos_pr30
    result.oos_sharpe = oos_sh

    sh_ratio  = (oos_sh / sh) if (not math.isnan(sh) and sh > 0) else 0.0
    pr_ratio  = (oos_pr30 / is_pr30) if is_pr30 > 0 else 0.0

    # MLL breach rate comparison (advisory)
    def _mll_rate(arr, dix, rules):
        if arr.size < 30:
            return float("nan")
        from topstep50k.analysis.passrate import simulate_combine_window
        from collections import Counter
        total = arr.size - 30 + 1
        cnt = Counter()
        for s in range(total):
            pnls = [(dix[s+i], Decimal(str(round(float(arr[s+i]), 2))))
                     for i in range(30)]
            r = simulate_combine_window(pnls, rules=rules,
                                         starting_balance=rules.starting_balance)
            cnt[r.outcome] += 1
        return cnt["mll_breach"] / total if total > 0 else 0.0

    is_mll_rate  = _mll_rate(is_arr,  is_days,  rules)
    oos_mll_rate = _mll_rate(oos_arr, oos_days, rules)
    mll_ok = (math.isnan(oos_mll_rate) or math.isnan(is_mll_rate)
              or oos_mll_rate <= is_mll_rate * 1.5)

    g4 = [
        GateResult("OOS EV > 0",            oos_ev > 0,        oos_ev,   0.0),
        GateResult("OOS Sharpe >= 0.5×IS",  sh_ratio >= 0.50,  sh_ratio, 0.50),
        GateResult("OOS pass30 >= 0.5×IS",  pr_ratio >= 0.50,  pr_ratio, 0.50,
                   hard=False),
        GateResult("OOS MLL rate <= 1.5×IS", mll_ok,           oos_mll_rate,
                   is_mll_rate * 1.5, hard=False,
                   note=f"IS={is_mll_rate:.1%} OOS={oos_mll_rate:.1%}"),
    ]
    result.gate4 = g4
    g4_passed = all(g.passed for g in g4 if g.hard)
    if not g4_passed:
        result.promoted = "retired"
        return result
    result.promoted = "oos_promoted"

    # ------------------------------------------------------------------ Gate 5
    g5 = []
    if existing_is_arrays:
        corrs = {}
        for k, arr in existing_is_arrays.items():
            xi, xj = is_arr, arr
            if xi.std() > 0 and xj.std() > 0:
                corrs[k] = float(np.corrcoef(xi, xj)[0, 1])
        max_corr = max(corrs.values()) if corrs else 0.0
        g5.append(GateResult(
            "max corr w/ existing < 0.6",
            max_corr < ENSEMBLE_CORR_MAX,
            max_corr, ENSEMBLE_CORR_MAX, hard=False,
            note=f"worst={max(corrs, key=corrs.get) if corrs else 'n/a'}"
        ))

    result.gate5 = g5
    if g5 and all(g.passed for g in g5):
        result.promoted = "fully_promoted"

    return result
