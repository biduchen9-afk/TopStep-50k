from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from topstep50k.analysis import (
    drawdown_curve,
    max_drawdown,
    performance,
    profit_factor,
    sharpe,
    sortino,
    walk_forward_folds,
)
from topstep50k.analysis.walkforward import WalkForwardFold


class TestStats:
    def test_max_drawdown(self):
        curve = [
            ("a", Decimal("50000")),
            ("b", Decimal("51000")),  # peak
            ("c", Decimal("49000")),  # -2000 from peak
            ("d", Decimal("50500")),
            ("e", Decimal("47500")),  # -3500 from peak
            ("f", Decimal("48000")),
        ]
        dd_dollars, dd_pct = max_drawdown(curve)
        assert dd_dollars == Decimal("3500")
        assert dd_pct == pytest.approx(3500 / 51000)

    def test_drawdown_curve_non_positive(self):
        curve = [("a", Decimal("100")), ("b", Decimal("120")), ("c", Decimal("90"))]
        dd = drawdown_curve(curve)
        assert dd[0][1] == Decimal("0")
        assert dd[1][1] == Decimal("0")
        assert dd[2][1] == Decimal("-30")

    def test_profit_factor(self):
        pnl = {
            date(2026, 1, 1): Decimal("200"),
            date(2026, 1, 2): Decimal("-50"),
            date(2026, 1, 3): Decimal("100"),
            date(2026, 1, 4): Decimal("-50"),
        }
        assert profit_factor(pnl) == pytest.approx(3.0)  # 300/100

    def test_sharpe_zero_when_no_variance(self):
        assert sharpe([0.001] * 10) == 0.0

    def test_sortino_inf_when_no_downside(self):
        import math
        s = sortino([0.01, 0.02, 0.005])
        assert math.isinf(s)

    def test_performance_record(self):
        pnl = {
            date(2026, 1, 1): Decimal("500"),
            date(2026, 1, 2): Decimal("-200"),
            date(2026, 1, 3): Decimal("700"),
        }
        curve = [
            ("a", Decimal("50000")),
            ("b", Decimal("50500")),
            ("c", Decimal("50300")),
            ("d", Decimal("51000")),
        ]
        stats = performance(pnl, curve, starting_balance=Decimal("50000"))
        assert stats.n_periods == 3
        assert stats.total_pnl == Decimal("1000")
        assert stats.win_rate == pytest.approx(2 / 3)
        assert stats.best_period == Decimal("700")
        assert stats.worst_period == Decimal("-200")


class TestWalkForward:
    def setup_method(self):
        self.days = [date(2026, 1, 1) + timedelta(days=i) for i in range(365)]

    def test_anchored_folds(self):
        folds = walk_forward_folds(self.days, train=180, test=30, anchored=True)
        # First fold: train [0..179], test [180..209]
        assert folds[0].train_start == self.days[0]
        assert folds[0].train_end == self.days[179]
        assert folds[0].test_start == self.days[180]
        assert folds[0].test_end == self.days[209]
        # All anchored folds share train_start
        for f in folds:
            assert f.train_start == self.days[0]

    def test_rolling_folds_advance_train_start(self):
        folds = walk_forward_folds(self.days, train=180, test=30, anchored=False)
        assert folds[1].train_start > folds[0].train_start

    def test_no_overlap_default_step(self):
        folds = walk_forward_folds(self.days, train=180, test=30)
        # Adjacent test windows touch but don't overlap.
        for a, b in zip(folds, folds[1:]):
            assert b.test_start > a.test_end

    def test_train_test_strictly_ordered(self):
        folds = walk_forward_folds(self.days, train=180, test=30)
        for f in folds:
            assert f.train_end < f.test_start

    def test_rejects_unsorted_days(self):
        with pytest.raises(ValueError, match="ascending"):
            walk_forward_folds(
                [date(2026, 1, 2), date(2026, 1, 1), date(2026, 1, 3)] * 100,
                train=2, test=1,
            )

    def test_rejects_short_input(self):
        with pytest.raises(ValueError, match="at least"):
            walk_forward_folds(self.days[:50], train=180, test=30)
