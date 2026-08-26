"""Buy-and-hold benchmark helper + harness Gate 4 wiring."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np

from topstep50k.analysis.benchmark import buy_and_hold_daily_pnl, to_aligned_array
from topstep50k.engine.types import Bar
from topstep50k.evaluation.harness import evaluate_strategy, is_oos_split
from topstep50k.rules import combine_50k


def _bar(ts: datetime, close: float) -> Bar:
    return Bar(ts=ts, open=close, high=close, low=close, close=close, volume=1)


def test_buy_and_hold_first_day_is_zero_then_tracks_close_deltas():
    # 3 trading days, well before the 16:00 CT rollover so each bar lands
    # on its own calendar trading day.
    bars = [
        _bar(datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc), 4000.0),
        _bar(datetime(2026, 1, 6, 15, 0, tzinfo=timezone.utc), 4010.0),
        _bar(datetime(2026, 1, 7, 15, 0, tzinfo=timezone.utc), 3990.0),
    ]
    daily = buy_and_hold_daily_pnl(bars, point_value=Decimal("50"), qty=1)
    days = sorted(daily)
    assert daily[days[0]] == Decimal("0")
    assert daily[days[1]] == Decimal("500.00")   # +10.0 pts * $50
    assert daily[days[2]] == Decimal("-1000.00")  # -20.0 pts * $50


def test_to_aligned_array_zero_fills_missing_days():
    bars = [
        _bar(datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc), 100.0),
        _bar(datetime(2026, 1, 6, 15, 0, tzinfo=timezone.utc), 101.0),
    ]
    daily = buy_and_hold_daily_pnl(bars, point_value=Decimal("1"), qty=1)
    days = sorted(daily)
    padded_index = days + [days[-1].replace(day=days[-1].day + 1)]
    arr = to_aligned_array(daily, padded_index)
    assert arr[-1] == 0.0


def test_harness_benchmark_gate_is_advisory_only():
    from datetime import date, timedelta

    rules = combine_50k()
    n = 200
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(n)]
    is_mask, oos_mask, is_days, oos_days = is_oos_split(days, {})

    # Deterministic, clean-edge series designed to clear every hard gate:
    # a 5-day cycle of four +150 days and one -100 day. Any 30-day window
    # (a multiple of the 5-day period) sums to exactly the same $3,000
    # regardless of phase, comfortably clearing the profit target with a
    # tiny best-day share (no consistency violation) and no meaningful
    # drawdown (no MLL breach) -- this is purely to reach Gate 4, not a
    # claim about realistic strategy behaviour.
    cycle = np.array([150.0, 150.0, 150.0, 150.0, -100.0])
    daily_full = np.tile(cycle, n // len(cycle) + 1)[:n]
    assert (daily_full[is_mask] != 0).all()

    # Benchmark that clearly outperforms the strategy OOS -- should NOT
    # flip promotion, only annotate Gate 4.
    benchmark_daily = np.full(n, 10_000.0)

    result = evaluate_strategy(
        "test", daily_full, is_mask, oos_mask, is_days, oos_days, rules,
        benchmark_daily=benchmark_daily,
    )
    assert result.promoted in ("oos_promoted", "retired")  # reached Gate 4 either way
    bench_gates = [g for g in result.gate4 if "buy-and-hold" in g.name]
    assert len(bench_gates) == 1
    assert bench_gates[0].hard is False
    assert bench_gates[0].passed is False  # strategy nets ~$500/wk, benchmark nets $10k/day
