import numpy as np
import pytest

from tv_pspline_psd import PSplineConfig, run_stationary_psd_mcmc


def _config() -> PSplineConfig:
    return PSplineConfig(
        n_interior_knots_freq=3,
        centered=True,
        trim_time_bins=0,
        trim_low_freq_channels=0,
        trim_high_freq_channels=0,
        freq_knot_strategy="linear",
    )


def test_matched_stationary_rejects_bad_mask_and_bins():
    coeffs = np.ones((8, 12))
    frequency = np.linspace(0.01, 0.1, 12)
    with pytest.raises(ValueError, match="likelihood_mask"):
        run_stationary_psd_mcmc(
            coeffs,
            frequency,
            config=_config(),
            likelihood_mask=np.ones((8, 11), dtype=bool),
            n_warmup=1,
            n_samples=1,
        )
    with pytest.raises(ValueError, match="freq_bin_starts"):
        run_stationary_psd_mcmc(
            coeffs,
            frequency,
            config=_config(),
            freq_bin_starts=np.array([1, 4]),
            n_warmup=1,
            n_samples=1,
        )
    with pytest.raises(ValueError, match="log_psd_offset"):
        run_stationary_psd_mcmc(
            coeffs,
            frequency,
            config=_config(),
            log_psd_offset=np.zeros((8, 11)),
            n_warmup=1,
            n_samples=1,
        )


def test_matched_stationary_mask_bins_and_explicit_knots_smoke():
    rng = np.random.default_rng(17)
    n_time, n_frequency = 20, 18
    frequency = np.linspace(0.01, 0.1, n_frequency)
    truth = 0.7 + 4.0 * frequency
    coefficients = rng.normal(size=(n_time, n_frequency)) * np.sqrt(truth)[None, :]
    mask = np.ones_like(coefficients, dtype=bool)
    mask[::4] = False
    mask[:, 9:12] = False
    starts = np.array([0, 3, 6, 9, 12, 15])
    knots = np.array([0.025, 0.05, 0.075])

    result = run_stationary_psd_mcmc(
        coefficients,
        frequency,
        config=_config(),
        interior_knots_freq=knots,
        likelihood_mask=mask,
        freq_bin_starts=starts,
        n_warmup=15,
        n_samples=15,
        num_chains=2,
        random_seed=4,
        target_accept_prob=0.9,
    )

    assert result["psd_geometric_mean"].shape == (n_frequency,)
    assert result["psd_geometric_mean_surface"].shape == coefficients.shape
    assert np.all(np.isfinite(result["psd_geometric_mean"]))
    assert np.all(result["psd_geometric_mean"] > 0.0)
    assert result["likelihood_frequency_bins"] == 5  # the fully masked bin is dropped
    assert np.array_equal(result["likelihood_mask"], mask)
    assert np.array_equal(result["freq_bin_starts"], starts)


def test_reference_stationary_is_exactly_rescaled_stationary_likelihood():
    rng = np.random.default_rng(23)
    n_time, n_frequency = 16, 14
    frequency = np.linspace(0.01, 0.1, n_frequency)
    coefficients = rng.normal(size=(n_time, n_frequency))
    time_shape = 0.25 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_time))[:, None]
    frequency_shape = 0.4 * np.log(frequency / frequency.mean())[None, :]
    log_reference = time_shape + frequency_shape
    transformed = coefficients * np.exp(-0.5 * log_reference)
    common = dict(
        config=_config(),
        interior_knots_freq=np.array([0.025, 0.05, 0.075]),
        freq_bin_starts=np.array([0, 3, 6, 9, 12]),
        n_warmup=10,
        n_samples=10,
        random_seed=12,
        target_accept_prob=0.9,
    )

    reference_fit = run_stationary_psd_mcmc(
        coefficients,
        frequency,
        log_psd_offset=log_reference,
        **common,
    )
    transformed_fit = run_stationary_psd_mcmc(
        transformed,
        frequency,
        **common,
    )

    assert reference_fit["reference_applied"] is True
    assert np.allclose(
        reference_fit["residual_log_psd_mean"],
        transformed_fit["log_psd_mean"],
        rtol=1e-6,
        atol=1e-6,
    )
    expected_surface = np.exp(
        log_reference + transformed_fit["log_psd_mean"][None, :]
    )
    assert np.allclose(
        reference_fit["psd_geometric_mean_surface"], expected_surface
    )
