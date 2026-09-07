"""EV-gated fractional-Kelly sizing.

The rule: don't scale up until the strategy's OOS edge is statistically
positive. We check two independent gates before non-zero sizing:

  1. Deflated Sharpe Ratio for the strategy on the previous walk-forward
     test fold must exceed `min_dsr` (typically 0.95).
     [ref:bailey_dsr]
  2. Bootstrap pass-probability on the same OOS daily-PnL series must
     exceed `min_pass_rate`. [ref:politis_romano]

If both gates pass, the recommended position multiplier is a FRACTION of
Kelly (default 0.25), capped at `max_multiplier`. Fractional Kelly is
the practitioner standard -- full Kelly is too aggressive for
non-stationary returns and small samples. [ref:thorp_2006],
[ref:maclean_thorp_zionts]

If either gate fails, the recommended multiplier is 0. This is by
design: the EV check is what stops the system from bidding into
backtest-mined edges.

This module does NOT modify positions itself. It returns a
SizingDecision; the caller (typically a portfolio risk_gate) is
responsible for translating it into TargetPosition.qty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from topstep50k.analysis.bootstrap import topstep_pass_probability
from topstep50k.analysis.dsr import deflated_sharpe
from topstep50k.rules.topstep import TopstepRules


@dataclass(frozen=True)
class EVGateResult:
    passed: bool
    dsr: float
    pass_rate: float
    pass_rate_ci_low: float
    expected_max_sharpe: float
    detail: str = ""


@dataclass(frozen=True)
class SizingDecision:
    multiplier: float           # what to scale the base position by
    gate: EVGateResult
    raw_kelly_fraction: float   # full-Kelly fraction (before damping)
    fractional_kelly_used: float


def fractional_kelly(
    mean_return: float,
    var_return: float,
    *,
    fraction: float = 0.25,
    cap: float = 1.0,
) -> float:
    """Fractional Kelly fraction = `fraction` * mu / sigma^2, capped at `cap`.

    Returns 0 if mean <= 0 or var <= 0 (no edge / degenerate variance).
    """
    if mean_return <= 0 or var_return <= 0:
        return 0.0
    full = mean_return / var_return
    return float(max(0.0, min(cap, fraction * full)))


@dataclass
class EVGate:
    """Two-check gate evaluated against an OOS daily-PnL series.

    Typical wiring in a walk-forward harness:

        for fold in folds:
            oos = run_fold(strategy, fold.test_days)  # returns daily PnL
            gate = EVGate(rules=...).evaluate(
                oos_daily_pnl=oos,
                all_trial_sharpes=trial_sharpes,
                starting_balance=...,
            )
            sizing = gate.size(mean_oos, var_oos)
            # apply sizing.multiplier in fold+1
    """

    rules: TopstepRules
    min_dsr: float = 0.95
    min_pass_rate: float = 0.55
    min_pass_rate_ci_low: float = 0.40
    kelly_fraction: float = 0.25
    max_multiplier: float = 1.0
    bootstrap_draws: int = 2000
    bootstrap_block_mean: float = 5.0
    bootstrap_target_days: int = 60
    seed: int | None = 42

    def evaluate(
        self,
        *,
        oos_daily_pnl: Sequence[Decimal],
        all_trial_sharpes_annual: Sequence[float],
        starting_balance: Decimal,
        periods_per_year: int = 252,
    ) -> EVGateResult:
        if not oos_daily_pnl:
            return EVGateResult(False, 0.0, 0.0, 0.0, 0.0, "empty OOS series")
        if len(oos_daily_pnl) < 4:
            return EVGateResult(False, 0.0, 0.0, 0.0, 0.0,
                                "OOS too short for inference (<4 days)")

        oos_returns = [float(v) / float(starting_balance) for v in oos_daily_pnl]
        dsr_res = deflated_sharpe(
            oos_returns,
            all_trial_sharpes_annual=all_trial_sharpes_annual or [0.0],
            periods_per_year=periods_per_year,
        )
        boot = topstep_pass_probability(
            daily_pnl=list(oos_daily_pnl),
            rules=self.rules,
            starting_balance=starting_balance,
            n_draws=self.bootstrap_draws,
            target_n_days=self.bootstrap_target_days,
            block_mean_length=self.bootstrap_block_mean,
            seed=self.seed,
        )
        dsr_ok = dsr_res.dsr >= self.min_dsr
        rate_ok = boot.pass_rate >= self.min_pass_rate
        ci_ok = boot.ci_low >= self.min_pass_rate_ci_low
        passed = dsr_ok and rate_ok and ci_ok
        detail = (
            f"dsr={dsr_res.dsr:.3f}(>={self.min_dsr}? {dsr_ok}), "
            f"pass_rate={boot.pass_rate:.3f}(>={self.min_pass_rate}? {rate_ok}), "
            f"ci_low={boot.ci_low:.3f}(>={self.min_pass_rate_ci_low}? {ci_ok})"
        )
        return EVGateResult(
            passed=passed,
            dsr=dsr_res.dsr,
            pass_rate=boot.pass_rate,
            pass_rate_ci_low=boot.ci_low,
            expected_max_sharpe=dsr_res.expected_max_sharpe,
            detail=detail,
        )

    def size(
        self,
        mean_oos_return: float,
        var_oos_return: float,
        *,
        gate: EVGateResult,
    ) -> SizingDecision:
        """Combine the gate result with a Kelly fraction.

        If gate.passed is False, multiplier = 0. Else
        multiplier = min(max_multiplier, fractional_kelly(mu, var)).
        """
        if not gate.passed:
            return SizingDecision(
                multiplier=0.0, gate=gate,
                raw_kelly_fraction=0.0, fractional_kelly_used=0.0,
            )
        full_k = mean_oos_return / var_oos_return if var_oos_return > 0 else 0.0
        frac_k = fractional_kelly(
            mean_oos_return, var_oos_return,
            fraction=self.kelly_fraction, cap=self.max_multiplier,
        )
        return SizingDecision(
            multiplier=frac_k,
            gate=gate,
            raw_kelly_fraction=float(max(0.0, full_k)),
            fractional_kelly_used=frac_k,
        )
