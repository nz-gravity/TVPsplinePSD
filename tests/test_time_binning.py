"""Likelihood coarse-graining (``time_bin``) correctness checks."""

from __future__ import annotations

import numpy as np
import pytest

from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface
from tv_pspline_psd.inference import bin_power_time_axis


def test_bin_power_time_axis_ragged_block_shapes_and_counts() -> None:
    n_time, n_freq, time_bin, n_components = 20, 5, 8, 3
    power = np.ones((n_time, n_freq))
    time_grid = np.linspace(0.0, 1.0, n_time)

    power_blocks, time_grid_blocks, counts_blocks = bin_power_time_axis(
        power, time_grid, time_bin, n_components
    )

    n_blocks = int(np.ceil(n_time / time_bin))  # 3 blocks: 8, 8, 4 (ragged last)
    assert power_blocks.shape == (n_blocks, n_freq)
    assert time_grid_blocks.shape == (n_blocks,)
    assert counts_blocks.shape == (n_blocks, 1)
    assert counts_blocks.sum() == n_components * n_time
    # Each block's power sum matches block size (power == 1 everywhere).
    np.testing.assert_allclose(power_blocks[:, 0], [8.0, 8.0, 4.0])
    # Block time coordinates are the block mean of the input grid.
    assert time_grid_blocks[0] == pytest.approx(np.mean(time_grid[:8]))
    assert time_grid_blocks[-1] == pytest.approx(np.mean(time_grid[16:]))


def _toy_surface(nt: int, nf: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A smooth, slowly-varying log-PSD surface, narrow relative to the knots."""
    time_grid = np.linspace(0.0, 1.0, nt)
    freq_grid = np.linspace(1e-3, 5e-3, nf)
    f_unit = (freq_grid - freq_grid[0]) / (freq_grid[-1] - freq_grid[0])
    log_psd = (
        -2.0
        + 0.6 * np.sin(2.0 * np.pi * time_grid)[:, None]
        - 1.2 * f_unit[None, :]
    )
    return time_grid, freq_grid, log_psd


def test_time_binned_posterior_matches_unbinned() -> None:
    nt, nf, time_bin = 128, 16, 8
    time_grid, freq_grid, true_log_psd = _toy_surface(nt, nf)
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal((1, nt, nf)) * np.exp(0.5 * true_log_psd)[None]

    config = PSplineConfig(
        n_interior_knots_time=6,
        n_interior_knots_freq=6,
        adaptive_time_knots=False,
    )
    common = dict(
        coeffs=coeffs, time_grid=time_grid, freq_grid=freq_grid,
        config=config, n_warmup=100, n_samples=100, random_seed=0,
        progress_bar=False,
    )
    full = fit_log_pspline_surface(**common, time_bin=1)
    binned = fit_log_pspline_surface(**common, time_bin=time_bin)

    assert full["time_bin"] == 1
    assert binned["time_bin"] == time_bin
    assert binned["log_psd_mean"].shape == full["log_psd_mean"].shape == true_log_psd.shape

    # Coarse-graining is exact for a block-constant surface; the truth here is
    # smooth relative to time_bin/n_time, so the two posterior means should
    # nearly coincide.
    diff = np.abs(binned["log_psd_mean"] - full["log_psd_mean"])
    assert diff.max() < 0.5
    assert np.mean(diff) < 0.15


def test_time_bin_rejects_invalid_values() -> None:
    config = PSplineConfig(adaptive_time_knots=False)
    coeffs = np.ones((1, 4, 5))
    with pytest.raises(ValueError, match="time_bin"):
        fit_log_pspline_surface(
            coeffs, np.arange(4), np.arange(5), config=config, time_bin=0,
        )
