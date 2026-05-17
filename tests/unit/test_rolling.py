"""Properties of the rolling-window stats:

  * causality: changing returns[t] never changes output[s] for s < t
  * length: output length equals input length
  * window fill: positions before the window is full are nan
  * Sharpe sign agrees with cumulative returns sign on monotone series
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from topstep50k.analysis.rolling import (
    rolling_hit_rate,
    rolling_max_drawdown,
    rolling_profit_factor,
    rolling_sharpe,
    rolling_sortino,
)


def _all_nan(xs):
    return all(math.isnan(x) for x in xs)


def test_rolling_sharpe_length_and_warmup():
    r = [0.001 * i for i in range(20)]
    s = rolling_sharpe(r, window=5)
    assert len(s) == len(r)
    assert _all_nan(s.values[:4])
    assert all(not math.isnan(x) for x in s.values[4:])


def test_rolling_sharpe_causal():
    r = [0.01, -0.02, 0.005, 0.003, -0.001, 0.004, 0.002]
    base = rolling_sharpe(r, window=3).values
    # Mutate a future point; ensure earlier output is unchanged.
    r2 = r.copy()
    r2[5] = 99.0
    mut = rolling_sharpe(r2, window=3).values
    for i in range(5):
        if math.isnan(base[i]):
            assert math.isnan(mut[i])
        else:
            assert base[i] == mut[i], f"causality violated at i={i}"


def test_rolling_sortino_handles_no_downside():
    r = [0.001] * 10
    s = rolling_sortino(r, window=4)
    # All upside -> nan once buffer fills
    assert _all_nan(s.values[3:])


def test_rolling_max_drawdown_simple():
    eq = [(i, Decimal(str(v))) for i, v in enumerate([100, 110, 105, 95, 100])]
    dd = rolling_max_drawdown(eq, window=3)
    # window=3 first non-nan at index 2: [100,110,105] -> max DD = 110-105 = 5
    assert math.isnan(dd.values[1])
    assert dd.values[2] == 5.0
    # next window [110,105,95] -> 110-95 = 15
    assert dd.values[3] == 15.0


def test_rolling_hit_rate_bounds():
    r = [0.1, -0.1, 0.1, -0.1, 0.1, 0.1]
    h = rolling_hit_rate(r, window=2)
    for v in h.values[1:]:
        assert 0.0 <= v <= 1.0


def test_rolling_profit_factor_inf_when_no_losses():
    r = [0.01, 0.02, 0.03]
    pf = rolling_profit_factor(r, window=3)
    assert math.isinf(pf.values[2])


@pytest.mark.parametrize("window", [0, 1])
def test_rolling_sharpe_rejects_tiny_window(window):
    with pytest.raises(ValueError):
        rolling_sharpe([0.0] * 10, window=window)
