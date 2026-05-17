"""Sensitivity sweep + PBO/DSR end-to-end on synthetic strategies."""

from __future__ import annotations

import numpy as np
import pytest

from topstep50k.analysis.sensitivity import sensitivity_sweep


def test_sweep_finds_dominant_param():
    """One param value has a real positive edge; others are noise. The
    sweep's winner should be the edge value AND the DSR should be high."""
    rng = np.random.default_rng(0)
    T = 320

    def runner(params: dict):
        edge = params["edge"]
        return rng.normal(loc=edge, scale=0.01, size=T).tolist()

    grid = {"edge": [-0.001, -0.0005, 0.0, 0.0005, 0.003]}
    res = sensitivity_sweep(grid, runner=runner, pbo_slices=16)
    assert res.winner.params == {"edge": 0.003}
    assert res.dsr_for_winner > 0.9
    # With a single dominant edge, PBO should be small (winner consistent OOS).
    assert res.pbo < 0.4


def test_sweep_high_pbo_on_pure_noise():
    """All cells are pure noise; PBO should be near 0.5 (winner is random)."""
    rng = np.random.default_rng(5)
    T = 320

    def runner(params: dict):
        # Same scale, but draw fresh noise per call -- nobody has skill
        return rng.normal(0.0, 0.01, size=T).tolist()

    grid = {"a": list(range(8)), "b": list(range(3))}  # 24 cells
    res = sensitivity_sweep(grid, runner=runner, pbo_slices=16, rng=np.random.default_rng(1))
    # Pure noise -> PBO is high-variance around 0.5; just assert that the
    # "in-sample winner" doesn't *systematically* generalise. A clean
    # discriminator is "PBO >> 0.15": noise gates produce nothing close
    # to a "real edge" signature.
    assert res.pbo > 0.15, f"noise produced suspiciously low PBO={res.pbo}"


def test_sweep_rejects_empty_grid():
    with pytest.raises(ValueError):
        sensitivity_sweep({}, runner=lambda p: [0.0])


def test_sweep_min_periods_filter():
    def runner(p):
        return [0.001] * 5
    with pytest.raises(ValueError):
        sensitivity_sweep({"a": [1, 2, 3]}, runner=runner, min_periods=20)


def test_sweep_grid_shape_metadata():
    rng = np.random.default_rng(0)
    def runner(p):
        return rng.normal(0.001, 0.01, size=64).tolist()
    res = sensitivity_sweep({"a": [1, 2], "b": [3, 4, 5]}, runner=runner,
                            pbo_slices=8)
    assert res.grid_shape == {"a": 2, "b": 3}
    assert len(res.cells) == 6
