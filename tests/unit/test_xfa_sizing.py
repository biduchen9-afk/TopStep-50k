"""Candidate XFA risk-adjustment policies: reactive (cushion, cooldown)
and tenure-based (hard stop, time decay)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from topstep50k.analysis.xfa_economics import XFAAccountState
from topstep50k.analysis.xfa_sizing import (
    combined_xfa_scaling,
    constant_scale,
    cushion_proportional_scaling,
    hard_stop_after,
    post_payout_cooldown,
    time_decay_scaling,
)


def _state(**kwargs) -> XFAAccountState:
    defaults = dict(
        balance=Decimal("50000"), floor=Decimal("48000"), cushion=Decimal("0"),
        locked=False, days_since_funding=0, days_since_last_payout=None,
        n_payouts_so_far=0, yesterday_pnl=None,
    )
    defaults.update(kwargs)
    return XFAAccountState(**defaults)


def test_cushion_proportional_scaling_ramps_linearly():
    fn = cushion_proportional_scaling(Decimal("2000"), floor_frac=0.2, ramp_frac=1.0)
    assert fn(_state(cushion=Decimal("0"))) == 0.2
    assert fn(_state(cushion=Decimal("1000"))) == pytest.approx(0.6)
    assert fn(_state(cushion=Decimal("2000"))) == pytest.approx(1.0)


def test_cushion_proportional_scaling_always_full_size_once_locked():
    fn = cushion_proportional_scaling(Decimal("2000"), floor_frac=0.0, ramp_frac=1.0)
    assert fn(_state(cushion=Decimal("0"), locked=True)) == 1.0


def test_post_payout_cooldown_windows_correctly():
    fn = post_payout_cooldown(cooldown_days=10, scale_during=0.3)
    assert fn(_state(days_since_last_payout=None)) == 1.0     # never paid out
    assert fn(_state(days_since_last_payout=0)) == 0.3         # day of payout
    assert fn(_state(days_since_last_payout=9)) == 0.3         # last cooled day
    assert fn(_state(days_since_last_payout=10)) == 1.0        # cooldown over


def test_hard_stop_after_bounds_tenure():
    fn = hard_stop_after(max_days=63, scale_after=0.0)
    assert fn(_state(days_since_funding=0)) == 1.0
    assert fn(_state(days_since_funding=62)) == 1.0
    assert fn(_state(days_since_funding=63)) == 0.0
    assert fn(_state(days_since_funding=200)) == 0.0


def test_time_decay_scaling_decays_toward_floor():
    fn = time_decay_scaling(half_life_days=60, floor_frac=0.1)
    assert fn(_state(days_since_funding=0)) == pytest.approx(1.0)
    assert fn(_state(days_since_funding=60)) == pytest.approx(0.1 + 0.9 * 0.5)
    assert fn(_state(days_since_funding=100_000)) == pytest.approx(0.1, abs=1e-6)


def test_combined_xfa_scaling_takes_the_minimum():
    fn = combined_xfa_scaling(
        cushion_proportional_scaling(Decimal("2000"), floor_frac=0.5, ramp_frac=1.0),
        hard_stop_after(max_days=10, scale_after=0.0),
    )
    # Before the stop: min(cushion-implied 0.5, 1.0) = 0.5
    assert fn(_state(cushion=Decimal("0"), days_since_funding=5)) == 0.5
    # After the stop: min(1.0, 0.0) = 0.0
    assert fn(_state(cushion=Decimal("2000"), days_since_funding=20)) == 0.0



def test_constant_scale_always_returns_k():
    fn = constant_scale(0.1)
    assert fn(_state(days_since_funding=0)) == 0.1
    assert fn(_state(days_since_funding=500, cushion=Decimal("1900"), locked=True)) == 0.1
