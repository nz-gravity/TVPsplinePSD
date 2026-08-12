import numpy as np
import pytest

from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface


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
