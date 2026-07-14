"""Fast regression tests for scale-free behavior and front-end validation."""

from __future__ import annotations

import numpy as np
import pytest
from wdm_transform import TimeSeries

from tv_pspline_psd import (
    PSplineConfig,
    fit_log_pspline_surface,
    mse_log_psd,
    run_stft_mcmc,
    wdm_analysis_coefficients,
)
from tv_pspline_psd.datasets import monte_carlo_reference
from tv_pspline_psd.datasets._wdm import trimmed_keep_indices
from tv_pspline_psd.model import initialize_with_penalized_least_squares, power_floor
from tv_pspline_psd.splines import (
    create_bspline_basis,
    create_bspline_roughness_penalty,
)


def test_mse_log_psd_is_correct_at_tdi_scale_and_scale_invariant() -> None:
    reference = np.full((3, 4), 1e-40)
    estimate = 2.0 * reference
    expected = np.log(2.0) ** 2
    assert mse_log_psd(reference, estimate) == pytest.approx(expected)
    assert mse_log_psd(reference * 1e17, estimate * 1e17) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_mse_log_psd_rejects_nonpositive_inputs(bad: float) -> None:
    reference = np.ones((2, 2))
    estimate = np.ones((2, 2))
    estimate[0, 0] = bad
    with pytest.raises(ValueError, match="strictly positive"):
        mse_log_psd(reference, estimate)


def test_power_floor_scales_linearly() -> None:
    power = np.geomspace(1e-9, 1e3, 40).reshape(5, 8)
    scale = 1e-40
    assert power_floor(power * scale) == pytest.approx(power_floor(power) * scale)


def test_penalized_least_squares_init_is_scale_equivariant() -> None:
    time = np.linspace(0.0, 1.0, 8)
    freq = np.linspace(0.1, 1.0, 7)
    config = PSplineConfig(
        n_interior_knots_time=2,
        n_interior_knots_freq=2,
        freq_knot_strategy="linear",
    )
    basis_time, knots_time = create_bspline_basis(time, 2, degree=3)
    basis_freq, knots_freq = create_bspline_basis(freq, 2, degree=3)
    penalty_time = create_bspline_roughness_penalty(
        knots_time, degree=3, derivative_order=2
    )
    penalty_freq = create_bspline_roughness_penalty(
        knots_freq, degree=3, derivative_order=2
    )
    power = np.exp(np.sin(time)[:, None] + np.cos(freq)[None, :])

    base = initialize_with_penalized_least_squares(
        power, basis_time, basis_freq, penalty_time, penalty_freq, config
    )["log_psd"]
    scaled = initialize_with_penalized_least_squares(
        1e-40 * power,
        basis_time,
        basis_freq,
        penalty_time,
        penalty_freq,
        config,
    )["log_psd"]
    np.testing.assert_allclose(
        scaled - base,
        np.log(1e-40),
        rtol=0.0,
        atol=2e-4,
    )


@pytest.mark.parametrize(
    ("n_total", "nt", "message"),
    [
        (72, 10, "divisible"),
        (72, 9, "both nt"),
        (72, 24, "both nt"),
    ],
)
def test_wdm_sizing_validation(n_total: int, nt: int, message: str) -> None:
    config = PSplineConfig(freq_knot_strategy="linear")
    with pytest.raises(ValueError, match=message):
        wdm_analysis_coefficients(np.ones(n_total), 0.1, nt, config)


def test_fit_surface_validates_shapes_finiteness_and_grids() -> None:
    config = PSplineConfig(freq_knot_strategy="linear")
    coeffs = np.ones((1, 4, 5))
    with pytest.raises(ValueError, match="shape must match"):
        fit_log_pspline_surface(coeffs, np.arange(3), np.arange(5), config=config)

    nonfinite = coeffs.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_log_pspline_surface(nonfinite, np.arange(4), np.arange(5), config=config)

    with pytest.raises(ValueError, match="strictly increasing"):
        fit_log_pspline_surface(coeffs, np.array([0, 1, 1, 2]), np.arange(5), config=config)


def test_stft_rejects_untrimmed_dc_and_nyquist() -> None:
    config = PSplineConfig(
        trim_low_freq_channels=0,
        trim_high_freq_channels=0,
        freq_knot_strategy="linear",
    )
    with pytest.raises(ValueError, match="DC/Nyquist"):
        run_stft_mcmc(
            np.arange(128, dtype=float),
            dt=0.1,
            nperseg=16,
            config=config,
            n_warmup=1,
            n_samples=1,
            progress_bar=False,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_phi": 0.0},
        {"trim_time_bins": -1},
        {"diff_order_time": 4, "degree_time": 3},
        {"freq_knot_strategy": "invalid"},
    ],
)
def test_config_rejects_invalid_hyperparameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PSplineConfig(**kwargs)


def test_streamed_calibration_matches_stored_draw_mean() -> None:
    n_total, nt, dt, n_draws, seed = 64, 8, 0.1, 4, 17
    config = PSplineConfig(
        trim_time_bins=1,
        trim_low_freq_channels=1,
        trim_high_freq_channels=1,
        freq_knot_strategy="linear",
    )

    def simulate(rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(n_total)

    streamed = monte_carlo_reference(
        simulate,
        n_draws=n_draws,
        n_total=n_total,
        dt=dt,
        nt=nt,
        config=config,
        seed=seed,
    )

    keep_time, keep_freq = trimmed_keep_indices(n_total, dt, nt, config)
    rng = np.random.default_rng(seed)
    stored = []
    for _ in range(n_draws):
        coeffs = np.asarray(TimeSeries(simulate(rng), dt=dt).to_wdm(nt=nt).coeffs)
        if coeffs.ndim == 3:
            coeffs = coeffs[0]
        stored.append(coeffs[np.ix_(keep_time, keep_freq)] ** 2)
    np.testing.assert_allclose(streamed, np.mean(stored, axis=0))
