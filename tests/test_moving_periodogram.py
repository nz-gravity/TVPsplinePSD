from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tv_pspline_psd import (
    PSplineConfig,
    bin_tang_ordinates,
    run_tang_dynamic_whittle_mcmc,
)
from tv_pspline_psd.model import power_whittle_log_likelihood
from tv_pspline_psd.moving_periodogram import (
    _grouped_log_surface,
    _grouped_normal_equations,
    tang_moving_periodogram,
)


def _scalar_tang_reference(data: np.ndarray, *, m: int, thin: int) -> dict[str, np.ndarray]:
    """Pre-vectorization construction, retained as a regression oracle."""
    x = np.asarray(data, dtype=float)
    T = x.size
    n_blocks = (T - 2 * m) // (thin * m)
    nu = np.arange(2 * m + 1)
    j = np.arange(1, m + 1)
    lam = 2.0 * j / (2 * m + 1)
    omega = np.pi * lam
    phase = np.exp(-1j * np.pi * np.outer(nu, lam))

    u_out, omega_out, mi_out = [], [], []
    for block in range(n_blocks):
        for jj in range(1, m + 1):
            t = m + thin * block * m + jj
            window = x[t - m - 1 : t + m]
            coeff = np.sum(window * phase[:, jj - 1])
            u_out.append(t / T)
            omega_out.append(omega[jj - 1])
            mi_out.append(np.abs(coeff) ** 2 / (2.0 * np.pi * (2 * m + 1)))
    return {
        "u": np.asarray(u_out),
        "omega": np.asarray(omega_out),
        "mi": np.asarray(mi_out),
    }


def test_vectorized_periodogram_matches_scalar_reference() -> None:
    rng = np.random.default_rng(42)
    for n, m, thin in ((127, 4, 2), (256, 7, 3), (1024, 16, 2)):
        data = rng.standard_normal(n)
        expected = _scalar_tang_reference(data, m=m, thin=thin)
        actual = tang_moving_periodogram(data, m=m, thin=thin)
        np.testing.assert_array_equal(actual["u"], expected["u"])
        np.testing.assert_array_equal(actual["omega"], expected["omega"])
        np.testing.assert_allclose(actual["mi"], expected["mi"], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(
            np.abs(actual["coeff"]) ** 2,
            actual["mi"],
            rtol=2e-14,
            atol=2e-14,
        )


def test_tang_power_count_likelihood_matches_exponential_form() -> None:
    mi = jnp.asarray([0.3, 1.1, 2.7, 0.6])
    log_psd = jnp.asarray([-0.2, 0.4, 1.2, -0.7])
    expected = jnp.sum(-(log_psd + mi * jnp.exp(-log_psd)))
    actual = power_whittle_log_likelihood(2.0 * mi, 2.0, log_psd)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_identity_tang_binning_returns_unpooled_sufficient_statistics() -> None:
    ordinates = tang_moving_periodogram(
        np.random.default_rng(4).standard_normal(191), m=6, thin=2
    )
    observations = bin_tang_ordinates(ordinates)
    np.testing.assert_array_equal(observations["u"], ordinates["u"])
    np.testing.assert_array_equal(observations["omega"], ordinates["omega"])
    np.testing.assert_allclose(observations["summed_power"], 2 * ordinates["mi"])
    np.testing.assert_array_equal(
        observations["counts"], np.full(ordinates["mi"].shape, 2.0)
    )


def test_tang_binning_sums_power_counts_and_keeps_ragged_bins() -> None:
    # Three time blocks by five frequency rungs; both final bins are ragged.
    u = np.asarray(
        [[0.10, 0.11, 0.12, 0.13, 0.14],
         [0.30, 0.31, 0.32, 0.33, 0.34],
         [0.50, 0.51, 0.52, 0.53, 0.54]]
    )
    omega_rungs = np.arange(1.0, 6.0)
    mi = np.arange(1.0, 16.0).reshape(3, 5)
    observations = bin_tang_ordinates(
        {
            "u": u.reshape(-1),
            "omega": np.tile(omega_rungs, 3),
            "mi": mi.reshape(-1),
        },
        time_bin=2,
        freq_bin=2,
    )

    assert observations["summed_power"].shape == (6,)
    np.testing.assert_array_equal(
        observations["counts"].reshape(2, 3),
        np.asarray([[8.0, 8.0, 4.0], [4.0, 4.0, 2.0]]),
    )
    expected_mi_sums = np.asarray(
        [[1 + 2 + 6 + 7, 3 + 4 + 8 + 9, 5 + 10],
         [11 + 12, 13 + 14, 15]]
    )
    np.testing.assert_array_equal(
        observations["summed_power"].reshape(2, 3), 2 * expected_mi_sums
    )
    np.testing.assert_allclose(
        observations["omega"].reshape(2, 3),
        np.asarray([[1.5, 3.5, 5.0], [1.5, 3.5, 5.0]]),
    )


def test_tang_binning_accepts_variable_frequency_bin_starts() -> None:
    u = np.tile(np.arange(5.0), (3, 1)) + np.arange(3.0)[:, None] * 10
    omega_rungs = np.arange(1.0, 6.0)
    mi = np.ones((3, 5))
    observations = bin_tang_ordinates(
        {
            "u": u.reshape(-1),
            "omega": np.tile(omega_rungs, 3),
            "mi": mi.reshape(-1),
        },
        freq_bin_starts=np.asarray([0, 1, 4]),
    )
    np.testing.assert_array_equal(
        observations["counts"].reshape(3, 3),
        np.asarray([[2.0, 6.0, 2.0]] * 3),
    )
    np.testing.assert_allclose(
        observations["omega"].reshape(3, 3),
        np.asarray([[1.0, 3.0, 5.0]] * 3),
    )


def test_grouped_surface_matches_pointwise_evaluation() -> None:
    rng = np.random.default_rng(7)
    n_blocks, n_freq, k_time, k_freq = 9, 5, 7, 6
    basis_time = rng.normal(size=(n_blocks * n_freq, k_time))
    basis_freq_unique = rng.normal(size=(n_freq, k_freq))
    eig_coeffs = rng.normal(size=(k_time, k_freq))
    basis_freq_full = np.tile(basis_freq_unique, (n_blocks, 1))

    expected = np.sum((basis_time @ eig_coeffs) * basis_freq_full, axis=1)
    actual = np.asarray(_grouped_log_surface(
        jnp.asarray(basis_time),
        jnp.asarray(basis_freq_unique),
        jnp.asarray(eig_coeffs),
    ))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_grouped_normal_equations_match_explicit_design() -> None:
    rng = np.random.default_rng(12)
    n_blocks, n_freq, k_time, k_freq = 11, 5, 6, 4
    target = rng.normal(size=n_blocks * n_freq)
    basis_time = rng.normal(size=(n_blocks * n_freq, k_time))
    basis_freq_unique = rng.normal(size=(n_freq, k_freq))
    basis_freq_full = np.tile(basis_freq_unique, (n_blocks, 1))
    design = (
        basis_freq_full[:, :, None] * basis_time[:, None, :]
    ).reshape(target.size, k_freq * k_time)

    gram, rhs = _grouped_normal_equations(
        target, basis_time, basis_freq_unique
    )
    np.testing.assert_allclose(gram, design.T @ design, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(rhs, design.T @ target, rtol=2e-12, atol=2e-12)


def test_dynamic_whittle_smoke() -> None:
    data = np.random.default_rng(3).standard_normal(256)
    result = run_tang_dynamic_whittle_mcmc(
        data,
        dt=0.1,
        m=8,
        thin=2,
        config=PSplineConfig(
            n_interior_knots_time=4,
            n_interior_knots_freq=3,
            freq_knot_strategy="linear",
        ),
        n_time_grid=12,
        n_warmup=8,
        n_samples=8,
        num_chains=1,
        random_seed=5,
        time_bin=2,
        freq_bin=2,
    )
    assert result["psd_mean"].shape == (12, 8)
    assert np.isfinite(result["psd_mean"]).all()
    assert result["power_observations"]["counts"].sum() == 2 * len(
        result["ordinates"]["mi"]
    )
