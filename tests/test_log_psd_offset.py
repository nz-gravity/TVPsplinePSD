import numpy as np
import pytest

from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface
from tv_pspline_psd.inference import _reference_scaled_power, bin_power_rectangular


def _config() -> PSplineConfig:
    return PSplineConfig(
        n_interior_knots_time=2,
        n_interior_knots_freq=3,
        centered=True,
        trim_time_bins=0,
        trim_low_freq_channels=0,
        trim_high_freq_channels=0,
        freq_knot_strategy="linear",
    )


def test_log_psd_offset_validation():
    coefficients = np.ones((1, 12, 16))
    time = np.linspace(0.0, 1.0, 12)
    frequency = np.linspace(0.01, 0.1, 16)
    with pytest.raises(ValueError, match="log_psd_offset"):
        fit_log_pspline_surface(
            coefficients,
            time,
            frequency,
            config=_config(),
            log_psd_offset=np.zeros((11, 16)),
            n_warmup=1,
            n_samples=1,
            progress_bar=False,
        )


def test_offset_model_matches_rescaled_residual_model():
    rng = np.random.default_rng(81)
    n_time, n_frequency = 14, 18
    time = np.linspace(0.0, 1.0, n_time)
    frequency = np.linspace(0.01, 0.1, n_frequency)
    reference = np.exp(0.5 * time[:, None] + 1.2 * frequency[None, :])
    residual_coefficients = rng.normal(size=(1, n_time, n_frequency))
    physical_coefficients = residual_coefficients * np.sqrt(reference)[None, :, :]
    common = dict(
        time_grid=time,
        freq_grid=frequency,
        config=_config(),
        n_warmup=25,
        n_samples=25,
        random_seed=12,
        progress_bar=False,
    )
    residual = fit_log_pspline_surface(residual_coefficients, **common)
    offset = fit_log_pspline_surface(
        physical_coefficients,
        log_psd_offset=np.log(reference),
        **common,
    )

    assert np.allclose(
        offset["log_psd_mean"] - np.log(reference),
        residual["log_psd_mean"],
        atol=1e-8,
        rtol=1e-8,
    )
    assert offset["provenance"]["log_psd_offset"]["applied"] is True
    assert residual["provenance"]["log_psd_offset"]["applied"] is False


def test_reference_scaled_power_divides_before_pooling():
    power = np.array([[2.0, 8.0, 18.0, 32.0]])
    reference = np.array([[1.0e-8, 4.0, 9.0, 16.0]])

    scaled = _reference_scaled_power(power, np.log(reference))

    assert np.allclose(scaled, power / reference)
    assert not np.isclose(
        scaled.sum(),
        power.sum() / np.exp(np.mean(np.log(reference))),
    )

    masked_log_reference = np.log(reference)
    masked_log_reference[0, 0] = -1.0e3
    masked = _reference_scaled_power(
        power,
        masked_log_reference,
        np.array([[False, True, True, True]]),
    )
    assert masked[0, 0] == 0.0
    assert np.all(np.isfinite(masked))


def test_coarse_reference_residual_likelihood_matches_full_cell_sum():
    rng = np.random.default_rng(18)
    power = rng.lognormal(size=(6, 9))
    reference = rng.lognormal(mean=0.0, sigma=3.0, size=power.shape)
    mask = np.ones_like(power, dtype=bool)
    mask[1::3, 2::4] = False
    time = np.arange(power.shape[0], dtype=float)
    frequency = np.arange(power.shape[1], dtype=float)
    time_starts = np.array([0, 2, 5])
    frequency_starts = np.array([0, 3, 7])
    residual_blocks = rng.normal(scale=0.2, size=(3, 3))

    scaled = _reference_scaled_power(power, np.log(reference), mask)
    block_power, _, _, block_counts = bin_power_rectangular(
        scaled,
        time,
        frequency,
        1,
        time_bin_starts=time_starts,
        freq_bin_starts=frequency_starts,
        likelihood_mask=mask,
    )
    coarse = -0.5 * np.sum(
        block_counts * residual_blocks
        + block_power * np.exp(-residual_blocks)
    )

    time_index = np.searchsorted(time_starts, np.arange(power.shape[0]), side="right") - 1
    frequency_index = (
        np.searchsorted(frequency_starts, np.arange(power.shape[1]), side="right") - 1
    )
    residual_cells = residual_blocks[time_index[:, None], frequency_index[None, :]]
    full = -0.5 * np.sum(
        np.where(
            mask,
            residual_cells + power / reference * np.exp(-residual_cells),
            0.0,
        )
    )

    assert coarse == pytest.approx(full, rel=1e-13, abs=1e-13)


def test_coarse_offset_model_matches_reference_whitened_residual_model():
    rng = np.random.default_rng(820)
    n_time, n_frequency = 12, 18
    time = np.linspace(0.0, 1.0, n_time)
    frequency = np.linspace(0.01, 0.1, n_frequency)
    # Alternate by orders of magnitude inside every coarse frequency block so
    # averaging log(R) would produce a visibly different quadratic statistic.
    reference_frequency = np.resize(np.array([1.0e-4, 1.0, 25.0]), n_frequency)
    reference = np.exp(0.4 * time[:, None]) * reference_frequency[None, :]
    residual_coefficients = rng.normal(size=(1, n_time, n_frequency))
    physical_coefficients = residual_coefficients * np.sqrt(reference)[None, :, :]
    likelihood_mask = np.ones((n_time, n_frequency), dtype=bool)
    likelihood_mask[1::4, 1::3] = False
    common = dict(
        time_grid=time,
        freq_grid=frequency,
        config=_config(),
        time_bin_starts=np.arange(0, n_time, 2),
        freq_bin_starts=np.arange(0, n_frequency, 3),
        likelihood_mask=likelihood_mask,
        n_warmup=15,
        n_samples=15,
        random_seed=93,
        progress_bar=False,
    )

    residual = fit_log_pspline_surface(residual_coefficients, **common)
    offset = fit_log_pspline_surface(
        physical_coefficients,
        log_psd_offset=np.log(reference),
        **common,
    )

    assert np.allclose(
        offset["log_psd_mean"] - np.log(reference),
        residual["log_psd_mean"],
        atol=1e-8,
        rtol=1e-8,
    )
    provenance = offset["provenance"]["log_psd_offset"]
    assert provenance["coarse_likelihood_handling"] == (
        "cellwise_power_divided_by_reference_before_block_sum"
    )
    assert provenance["data_only_reference_log_determinant_omitted"] is True
