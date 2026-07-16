"""Likelihood coarse-graining (``time_bin``) correctness checks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface
from tv_pspline_psd.inference import (
    adaptive_frequency_bin_starts,
    bin_power_rectangular,
    bin_power_time_axis,
    gap_aware_time_bin_starts,
)
from tv_pspline_psd.model import power_whittle_log_likelihood


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


def test_bin_power_rectangular_handles_ragged_and_variable_frequency_bins() -> None:
    power = np.ones((5, 7))
    time_grid = np.arange(5, dtype=float)
    freq_grid = np.arange(7, dtype=float)

    power_blocks, time_blocks, freq_blocks, counts = bin_power_rectangular(
        power,
        time_grid,
        freq_grid,
        2,
        time_bin=2,
        freq_bin_starts=np.array([0, 3, 4]),
    )

    assert power_blocks.shape == counts.shape == (3, 3)
    np.testing.assert_allclose(time_blocks, [0.5, 2.5, 4.0])
    np.testing.assert_allclose(freq_blocks, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(
        counts,
        2 * np.array([[6, 2, 6], [6, 2, 6], [3, 1, 3]]),
    )
    np.testing.assert_allclose(power_blocks, counts / 2)


def test_block_constant_surface_has_exact_coarse_grained_likelihood() -> None:
    """Summed power/counts must preserve the likelihood without approximation."""
    rng = np.random.default_rng(21)
    n_components = 3
    power = rng.gamma(shape=1.5, scale=2.0, size=(5, 7))
    time_grid = np.arange(5, dtype=float)
    freq_grid = np.arange(7, dtype=float)
    freq_starts = np.array([0, 3, 4])

    coarse_power, _, _, coarse_counts = bin_power_rectangular(
        power,
        time_grid,
        freq_grid,
        n_components,
        time_bin=2,
        freq_bin_starts=freq_starts,
    )
    coarse_log_psd = np.array(
        [[-1.2, 0.4, 1.1], [-0.7, 0.2, 0.8], [-0.3, 0.6, 1.4]]
    )
    time_index = np.array([0, 0, 1, 1, 2])
    freq_index = np.array([0, 0, 0, 1, 2, 2, 2])
    fine_log_psd = coarse_log_psd[time_index[:, None], freq_index[None, :]]

    exact = power_whittle_log_likelihood(
        jnp.asarray(power), jnp.asarray(n_components), jnp.asarray(fine_log_psd)
    )
    coarse = power_whittle_log_likelihood(
        jnp.asarray(coarse_power),
        jnp.asarray(coarse_counts),
        jnp.asarray(coarse_log_psd),
    )
    np.testing.assert_allclose(coarse, exact, rtol=1e-13, atol=1e-13)


def test_adaptive_frequency_bins_refine_a_sharp_feature() -> None:
    freq = np.linspace(0.0, 1.0, 101)
    pilot = -4.0 * np.exp(-0.5 * ((freq - 0.5) / 0.025) ** 2)[None, :]
    starts = adaptive_frequency_bin_starts(
        pilot, max_log_range=0.2, max_bin=16
    )
    widths = np.diff(np.r_[starts, freq.size])
    centers = np.array([freq[s : s + w].mean() for s, w in zip(starts, widths)])

    assert widths.max() == 16
    assert widths[abs(centers - 0.5) < 0.08].min() <= 2
    assert widths[abs(centers - 0.5) > 0.2].max() == 16


def test_gap_aware_time_bins_never_pool_across_a_missing_interval() -> None:
    time_grid = np.array([0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0])
    starts = gap_aware_time_bin_starts(time_grid, 2)

    np.testing.assert_array_equal(starts, [0, 2, 4, 6])
    power, pooled_time, _, counts = bin_power_rectangular(
        np.ones((time_grid.size, 3)),
        time_grid,
        np.arange(3, dtype=float),
        1,
        time_bin_starts=starts,
    )
    np.testing.assert_allclose(pooled_time, [0.5, 2.5, 8.5, 10.0])
    np.testing.assert_array_equal(counts[:, 0], [2, 2, 2, 1])
    np.testing.assert_array_equal(power[:, 0], [2.0, 2.0, 2.0, 1.0])


def test_gap_aware_time_partition_is_saved_completely_in_provenance() -> None:
    time_grid = np.array([0.0, 1.0, 2.0, 7.0, 8.0])
    starts = gap_aware_time_bin_starts(time_grid, 2)
    from tv_pspline_psd.provenance import binning_provenance

    recipe = binning_provenance(
        n_time=time_grid.size,
        n_freq=4,
        time_bin_starts=starts,
        selector_metadata={"time": {"method": "gap_aware_uniform", "nominal_width": 2}},
    )

    assert recipe["time"]["mode"] == "variable"
    assert recipe["time"]["starts"] == [0, 2, 3]
    assert recipe["time"]["widths"] == [2, 1, 2]
    assert recipe["selector"]["time"]["nominal_width"] == 2


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
        freq_knot_strategy="linear",
    )
    common = dict(
        coeffs=coeffs, time_grid=time_grid, freq_grid=freq_grid,
        config=config, n_warmup=100, n_samples=100, random_seed=0,
        progress_bar=False,
    )
    full = fit_log_pspline_surface(**common, time_bin=1)
    binned = fit_log_pspline_surface(
        **common,
        time_bin=time_bin,
        binning_metadata={"time": {"method": "fixed", "reason": "test"}},
    )

    assert full["time_bin"] == 1
    assert binned["time_bin"] == time_bin
    assert binned["log_psd_mean"].shape == full["log_psd_mean"].shape == true_log_psd.shape
    recipe = binned["provenance"]["binning"]
    assert recipe["input_shape"] == [nt, nf]
    assert recipe["output_shape"] == [16, nf]
    assert recipe["time"]["widths"] == [time_bin] * 16
    assert recipe["frequency"]["mode"] == "identity"
    assert recipe["selector"]["time"]["reason"] == "test"

    # Coarse-graining is exact for a block-constant surface; the truth here is
    # smooth relative to time_bin/n_time, so the two posterior means should
    # nearly coincide.
    diff = np.abs(binned["log_psd_mean"] - full["log_psd_mean"])
    assert diff.max() < 0.5
    assert np.mean(diff) < 0.15


def test_frequency_binned_posterior_matches_unbinned_on_smooth_surface() -> None:
    nt, nf = 24, 128
    time_grid, freq_grid, true_log_psd = _toy_surface(nt, nf)
    rng = np.random.default_rng(3)
    coeffs = rng.standard_normal((1, nt, nf)) * np.exp(0.5 * true_log_psd)[None]
    config = PSplineConfig(
        n_interior_knots_time=5,
        n_interior_knots_freq=6,
        freq_knot_strategy="linear",
    )
    common = dict(
        coeffs=coeffs,
        time_grid=time_grid,
        freq_grid=freq_grid,
        config=config,
        n_warmup=100,
        n_samples=100,
        random_seed=4,
        progress_bar=False,
    )

    full = fit_log_pspline_surface(**common)
    binned = fit_log_pspline_surface(**common, freq_bin=8, time_bin=2)

    assert binned["likelihood_grid_shape"] == (12, 16)
    diff = np.abs(binned["log_psd_mean"] - full["log_psd_mean"])
    assert diff.max() < 0.35
    assert np.mean(diff) < 0.12


def test_time_bin_rejects_invalid_values() -> None:
    config = PSplineConfig(freq_knot_strategy="linear")
    coeffs = np.ones((1, 4, 5))
    with pytest.raises(ValueError, match="time_bin"):
        fit_log_pspline_surface(
            coeffs, np.arange(4), np.arange(5), config=config, time_bin=0,
        )
    with pytest.raises(ValueError, match="freq_bin"):
        fit_log_pspline_surface(
            coeffs, np.arange(4), np.arange(5), config=config, freq_bin=0,
        )
    with pytest.raises(ValueError, match="freq_bin must be 1"):
        fit_log_pspline_surface(
            coeffs,
            np.arange(4),
            np.arange(5),
            config=config,
            freq_bin=2,
            freq_bin_starts=np.array([0, 2]),
        )
