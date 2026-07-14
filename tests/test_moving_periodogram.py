from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tv_pspline_psd import PSplineConfig, run_tang_dynamic_whittle_mcmc
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
    )
    assert result["psd_mean"].shape == (12, 8)
    assert np.isfinite(result["psd_mean"]).all()
