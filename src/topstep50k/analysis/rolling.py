"""Rolling-window performance stats over an equity / return series.

All windows are CAUSAL: the value at index t uses only points <= t. We
never centre, smooth backwards, or peek. This lets a caller emit a
rolling-Sharpe series alongside the equity curve in the audit log
without violating no-look-ahead.

Implemented stats:
  * rolling Sharpe  (annualised, periods_per_year configurable)
  * rolling Sortino (downside-only stdev)
  * rolling max drawdown (peak-to-trough within the window)
  * rolling hit rate
  * rolling profit factor

See [ref:bailey_psr] for the case for time-varying Sharpe rather than a
single point estimate, and [ref:lopez_aml_ch11] for why rolling-window
diagnostics are the minimum viable check on stationarity assumptions
that the bootstrap depends on.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence


@dataclass(frozen=True)
class RollingSeries:
    """One causal series aligned to the input index. nan in positions
    where the window isn't yet full."""

    values: list[float]
    window: int

    def __len__(self) -> int:
        return len(self.values)


def _stdev(buf: Sequence[float]) -> float:
    n = len(buf)
    if n < 2:
        return 0.0
    mu = sum(buf) / n
    var = sum((x - mu) ** 2 for x in buf) / (n - 1)
    return math.sqrt(var)


def rolling_sharpe(
    returns: Sequence[float],
    window: int,
    periods_per_year: int = 252,
) -> RollingSeries:
    """Causal rolling Sharpe. Output[t] uses returns[t-window+1 .. t]."""
    if window < 2:
        raise ValueError("rolling Sharpe needs window >= 2")
    out: list[float] = []
    buf: deque[float] = deque(maxlen=window)
    s = 0.0
    s2 = 0.0
    for r in returns:
        if len(buf) == window:
            old = buf[0]
            s -= old
            s2 -= old * old
        buf.append(r)
        s += r
        s2 += r * r
        if len(buf) < window:
            out.append(float("nan"))
            continue
        mu = s / window
        var = max((s2 - window * mu * mu) / (window - 1), 0.0)
        sigma = math.sqrt(var)
        if sigma < 1e-12:
            out.append(float("nan"))
        else:
            out.append((mu / sigma) * math.sqrt(periods_per_year))
    return RollingSeries(out, window)


def rolling_sortino(
    returns: Sequence[float],
    window: int,
    periods_per_year: int = 252,
) -> RollingSeries:
    """Causal rolling Sortino using downside semivariance (negative-only)."""
    if window < 2:
        raise ValueError("rolling Sortino needs window >= 2")
    out: list[float] = []
    buf: deque[float] = deque(maxlen=window)
    for r in returns:
        buf.append(r)
        if len(buf) < window:
            out.append(float("nan"))
            continue
        mu = sum(buf) / window
        downside = [x for x in buf if x < 0]
        if not downside:
            out.append(float("nan"))
            continue
        d_var = sum(x * x for x in downside) / len(downside)
        d_sigma = math.sqrt(d_var)
        if d_sigma < 1e-12:
            out.append(float("nan"))
        else:
            out.append((mu / d_sigma) * math.sqrt(periods_per_year))
    return RollingSeries(out, window)


def rolling_max_drawdown(
    equity_curve: Sequence[tuple[object, Decimal]],
    window: int,
) -> RollingSeries:
    """Causal rolling dollar-MaxDD inside a trailing equity window."""
    if window < 2:
        raise ValueError("rolling MaxDD needs window >= 2")
    eq = [float(e) for _, e in equity_curve]
    out: list[float] = []
    buf: deque[float] = deque(maxlen=window)
    for v in eq:
        buf.append(v)
        if len(buf) < window:
            out.append(float("nan"))
            continue
        peak = buf[0]
        max_dd = 0.0
        for x in buf:
            if x > peak:
                peak = x
            dd = peak - x
            if dd > max_dd:
                max_dd = dd
        out.append(max_dd)
    return RollingSeries(out, window)


def rolling_hit_rate(returns: Sequence[float], window: int) -> RollingSeries:
    if window < 1:
        raise ValueError("rolling hit-rate needs window >= 1")
    out: list[float] = []
    buf: deque[float] = deque(maxlen=window)
    for r in returns:
        buf.append(r)
        if len(buf) < window:
            out.append(float("nan"))
            continue
        wins = sum(1 for x in buf if x > 0)
        out.append(wins / window)
    return RollingSeries(out, window)


def rolling_profit_factor(returns: Sequence[float], window: int) -> RollingSeries:
    if window < 1:
        raise ValueError("rolling PF needs window >= 1")
    out: list[float] = []
    buf: deque[float] = deque(maxlen=window)
    for r in returns:
        buf.append(r)
        if len(buf) < window:
            out.append(float("nan"))
            continue
        gw = sum(x for x in buf if x > 0)
        gl = -sum(x for x in buf if x < 0)
        if gl <= 0:
            out.append(float("inf") if gw > 0 else float("nan"))
        else:
            out.append(gw / gl)
    return RollingSeries(out, window)
