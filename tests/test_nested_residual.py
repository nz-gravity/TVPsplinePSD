import numpy as np
import pytest

from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface


def _config() -> PSplineConfig:
    return PSplineConfig(
        n_interior_knots_time=3,
        n_interior_knots_freq=3,
        trim_time_bins=0,
        trim_low_freq_channels=0,
        trim_high_freq_channels=0,
        freq_knot_strategy="linear",
        centered=True,
    )


def test_nested_residual_validates_structure_and_prior_scale():
    coeffs = np.ones((1, 8, 10))
    time = np.linspace(0.0, 1.0, 8)
    frequency = np.linspace(0.01, 0.1, 10)
    with pytest.raises(ValueError, match="residual_structure"):
        fit_log_pspline_surface(
            coeffs, time, frequency, config=_config(), residual_structure="bad"
        )
    with pytest.raises(ValueError, match="interaction_scale_prior"):
        fit_log_pspline_surface(
            coeffs,
            time,
            frequency,
            config=_config(),
            residual_structure="stationary_plus_interaction",
            interaction_scale_prior=0.0,
        )
    with pytest.raises(ValueError, match="interaction_time_knots"):
        fit_log_pspline_surface(
            coeffs,
            time,
            frequency,
            config=_config(),
            residual_structure="stationary_plus_interaction",
            interaction_time_knots=-1,
        )


def test_nested_interaction_basis_has_zero_time_mean_and_smoke_fits():
    rng = np.random.default_rng(42)
    n_time, n_frequency = 18, 16
    time = np.linspace(0.0, 1.0, n_time)
    frequency = np.linspace(0.01, 0.1, n_frequency)
    stationary = 0.8 + 3.0 * frequency
    modulation = np.exp(0.25 * np.sin(2.0 * np.pi * time))
    truth = modulation[:, None] * stationary[None, :]
    coefficients = rng.normal(size=(1, n_time, n_frequency)) * np.sqrt(truth)

    result = fit_log_pspline_surface(
        coefficients,
        time,
        frequency,
        config=_config(),
        residual_structure="stationary_plus_interaction",
        interaction_scale_prior=0.5,
        interaction_time_knots=1,
        n_warmup=15,
        n_samples=15,
        num_chains=2,
        random_seed=8,
        target_accept_prob=0.9,
        progress_bar=False,
    )

    assert result["residual_structure"] == "stationary_plus_interaction"
    assert np.allclose(result["B_time_interaction"].mean(axis=0), 0.0, atol=1e-14)
    assert result["B_time_interaction"].shape[1] < result["B_time"].shape[1] - 1
    assert np.all(np.isfinite(result["psd_geometric_mean"]))
    assert np.all(result["psd_geometric_mean"] > 0.0)
    assert np.all(np.asarray(result["interaction_scale_samples"]) > 0.0)
