"""Performance statistics.

All ratios operate on a SERIES OF PERIOD RETURNS (e.g. per-day pnl /
sod_equity). Annualisation uses an explicit periods_per_year so the
caller decides the convention (252 daily, 52 weekly, 12 monthly, etc.).

Drawdown is reported on the EQUITY CURVE in dollars — not on returns —
because TopStep rules are dollar-denominated and "drawdown in returns"
is misleading on a growing/shrinking account.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Sequence


@dataclass(frozen=True)
class PerformanceStats:
    n_periods: int
    total_pnl: Decimal
    mean_return: float
    stdev_return: float
    sharpe_annual: float
    sortino_annual: float
    max_drawdown_dollars: Decimal
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    best_period: Decimal
    worst_period: Decimal


def equity_to_returns(equity_curve: Sequence[tuple[object, Decimal]]) -> list[float]:
    """Convert equity points to simple returns. First point has no prior,
    so it is dropped (the returns list is length N-1)."""
    out: list[float] = []
    for i in range(1, len(equity_curve)):
        prev = float(equity_curve[i - 1][1])
        cur = float(equity_curve[i][1])
        if prev == 0:
            out.append(0.0)
        else:
            out.append((cur - prev) / prev)
    return out


def sharpe(returns: Sequence[float], periods_per_year: int = 252, rf: float = 0.0) -> float:
    if not returns:
        return 0.0
    mu = sum(returns) / len(returns) - rf / periods_per_year
    if len(returns) < 2:
        return 0.0
    var = sum((r - sum(returns) / len(returns)) ** 2 for r in returns) / (len(returns) - 1)
    sigma = math.sqrt(var)
    if sigma < 1e-12:
        return 0.0
    return (mu / sigma) * math.sqrt(periods_per_year)


def sortino(returns: Sequence[float], periods_per_year: int = 252, rf: float = 0.0) -> float:
    if not returns:
        return 0.0
    mu = sum(returns) / len(returns) - rf / periods_per_year
    downside = [r for r in returns if r < 0]
    if not downside:
        return float("inf") if mu > 0 else 0.0
    # population stdev of downside returns
    var = sum(r ** 2 for r in downside) / len(downside)
    sigma_d = math.sqrt(var)
    if sigma_d == 0:
        return 0.0
    return (mu / sigma_d) * math.sqrt(periods_per_year)


def max_drawdown(equity_curve: Sequence[tuple[object, Decimal]]) -> tuple[Decimal, float]:
    """Returns (peak-to-trough dollars, peak-to-trough pct)."""
    if not equity_curve:
        return Decimal("0"), 0.0
    peak = equity_curve[0][1]
    max_dd_dollars = Decimal("0")
    max_dd_pct = 0.0
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd_dollars:
            max_dd_dollars = dd
            if peak > 0:
                max_dd_pct = float(dd / peak)
    return max_dd_dollars, max_dd_pct


def drawdown_curve(equity_curve: Sequence[tuple[object, Decimal]]) -> list[tuple[object, Decimal]]:
    out = []
    peak = equity_curve[0][1] if equity_curve else Decimal("0")
    for ts, eq in equity_curve:
        if eq > peak:
            peak = eq
        out.append((ts, eq - peak))
    return out


def profit_factor(daily_pnl: dict[date, Decimal]) -> float:
    gross_win = sum((v for v in daily_pnl.values() if v > 0), start=Decimal(0))
    gross_loss = sum((-v for v in daily_pnl.values() if v < 0), start=Decimal(0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return float(gross_win / gross_loss)


def performance(
    daily_pnl: dict[date, Decimal],
    equity_curve: Sequence[tuple[object, Decimal]],
    starting_balance: Decimal,
    periods_per_year: int = 252,
) -> PerformanceStats:
    pnl_values = list(daily_pnl.values())
    n = len(pnl_values)
    total = sum(pnl_values, start=Decimal(0))
    if n == 0:
        return PerformanceStats(
            n_periods=0, total_pnl=Decimal(0), mean_return=0.0, stdev_return=0.0,
            sharpe_annual=0.0, sortino_annual=0.0,
            max_drawdown_dollars=Decimal(0), max_drawdown_pct=0.0,
            win_rate=0.0, profit_factor=0.0,
            best_period=Decimal(0), worst_period=Decimal(0),
        )
    daily_returns = [float(v) / float(starting_balance) for v in pnl_values]
    mean_r = sum(daily_returns) / n
    if n > 1:
        var = sum((r - mean_r) ** 2 for r in daily_returns) / (n - 1)
        std_r = math.sqrt(var)
    else:
        std_r = 0.0
    dd_dollars, dd_pct = max_drawdown(equity_curve)
    return PerformanceStats(
        n_periods=n,
        total_pnl=total,
        mean_return=mean_r,
        stdev_return=std_r,
        sharpe_annual=sharpe(daily_returns, periods_per_year),
        sortino_annual=sortino(daily_returns, periods_per_year),
        max_drawdown_dollars=dd_dollars,
        max_drawdown_pct=dd_pct,
        win_rate=sum(1 for v in pnl_values if v > 0) / n,
        profit_factor=profit_factor(daily_pnl),
        best_period=max(pnl_values),
        worst_period=min(pnl_values),
    )
