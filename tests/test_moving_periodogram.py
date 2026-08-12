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


# ---------------------------------------------------------------------------
# Paper-faithful oracle.
#
# The two functions below are transcribed directly from Tang, Kirch, Lee &
# Meyer, JASA (2026), doi:10.1080/01621459.2025.2594191, Definition 1 (eq. 2.2).
# They are deliberately naive -- plain loops, paper notation, no vectorisation --
# so they can be read side by side with the paper. Unlike
# a regression oracle (an older copy of our own code, which would share its
# conventions and so could never detect a departure from the paper), these
# encode only what the paper states.
# ---------------------------------------------------------------------------


def _tang_mod(t: int, m: int) -> int:
    """Tang Definition 1: mod(t) = 1 + ((t - 1) mod m), a function of t alone."""
    return 1 + ((t - 1) % m)


def _tang_d(x: np.ndarray, t: int, m: int) -> tuple[complex, float]:
    """Tang eq. (2.2): the moving Fourier coefficient d_t and its frequency.

    Written straight from the paper, with ``x`` 1-based-indexable via
    ``x[k - 1]``. Returns ``(d_t, lambda_mod(t))``, where

        d_t = sum_{nu=0}^{2m} X_{nu+t-m} exp(-i pi nu lambda_mod(t))
              / sqrt(2 pi (2m + 1)),     lambda_j = 2j / (2m + 1)

    and the periodogram ordinate of eq. (2.2) is MI_t = |d_t|^2.
    """
    j = _tang_mod(t, m)
    lam = 2.0 * j / (2 * m + 1)
    total = 0.0 + 0.0j
    for nu in range(0, 2 * m + 1):
        total += x[(nu + t - m) - 1] * np.exp(-1j * np.pi * nu * lam)
    return total / np.sqrt(2.0 * np.pi * (2 * m + 1)), lam


def _tang_MI(x: np.ndarray, t: int, m: int) -> tuple[float, float]:
    """Tang eq. (2.2): the moving periodogram ordinate MI_t = |d_t|^2."""
    d, lam = _tang_d(x, t, m)
    return abs(d) ** 2, lam


def test_each_ordinate_equals_tang_definition_1_at_its_own_time_point() -> None:
    """Every returned ordinate is exactly Tang eq. (2.2) evaluated at its own t.

    This deliberately does not assume *which* time points are retained (that is
    the documented boundary deviation); it reads each ordinate's reported time
    back out of ``u``, and checks the value and the frequency against the paper
    formula at that time point. So a change in the retained-ordinate convention
    stays visible in the tests below, while any error in eq. (2.2) itself --
    normalisation, phase sign, window alignment, or the mod(t) frequency
    assignment -- fails here.
    """
    rng = np.random.default_rng(11)
    for T, m, thin in ((20, 5, 1), (20, 5, 2), (512, 8, 3), (1000, 16, 2)):
        x = rng.standard_normal(T)
        out = tang_moving_periodogram(x, m=m, thin=thin)

        print(f"T={T}, m={m}, thin={thin}, n_ordinates={len(out['u'])}")
        print(out)

        t_all = np.rint(out["u"] * T).astype(int)  # recover the 1-based centre
        assert np.all(t_all - m >= 1), "window runs off the start of the series"
        assert np.all(t_all + m <= T), "window runs off the end of the series"

        for t, mi_got, omega_got in zip(t_all, out["mi"], out["omega"]):
            mi_want, lam_want = _tang_MI(x, int(t), m)
            np.testing.assert_allclose(mi_got, mi_want, rtol=1e-12, atol=1e-12)
            # omega = pi * lambda is our angular-frequency convention.
            np.testing.assert_allclose(omega_got, np.pi * lam_want, rtol=1e-12)


def test_complex_coefficient_uses_tang_phase_sign_convention() -> None:
    """Pin the sign of the retained complex coefficient.

    ``mi`` alone cannot detect a flipped phase sign: for real input, negating
    the exponent conjugates d_t and ``|d_t|^2`` is unchanged, so the likelihood
    is genuinely invariant. But ``coeff`` is exposed for a future
    coefficient-level signal likelihood, where d_t and conj(d_t) are not
    interchangeable -- so the convention is fixed here against eq. (2.2).
    """
    rng = np.random.default_rng(14)
    for T, m, thin in ((300, 5, 2), (512, 8, 3)):
        x = rng.standard_normal(T)
        out = tang_moving_periodogram(x, m=m, thin=thin)
        t_all = np.rint(out["u"] * T).astype(int)
        expected = np.asarray([_tang_d(x, int(t), m)[0] for t in t_all])
        np.testing.assert_allclose(out["coeff"], expected, rtol=1e-12, atol=1e-12)


def test_zigzag_frequency_matches_tang_mod_t_for_every_thin() -> None:
    """The zigzag comes from mod(t), not from thinning.

    Renate's review point: the cycling of the evaluated frequency is part of
    Definition 1 and is present even unthinned. Here the frequency assigned to
    each ordinate is checked against ``mod(t) = 1 + ((t-1) mod m)`` for
    ``thin = 1, 2, 3`` -- identical rule in all three cases.
    """
    rng = np.random.default_rng(12)
    for T, m in ((600, 7), (1000, 16)):
        x = rng.standard_normal(T)
        for thin in (1, 2, 3):
            out = tang_moving_periodogram(x, m=m, thin=thin)
            t_all = np.rint(out["u"] * T).astype(int)
            expected_j = np.asarray([_tang_mod(int(t), m) for t in t_all])
            expected_omega = np.pi * 2.0 * expected_j / (2 * m + 1)
            np.testing.assert_allclose(out["omega"], expected_omega, rtol=1e-12)


def test_documented_boundary_deviations_from_tang_definition_3() -> None:
    """Pin the two deviations from Definition 3 recorded in the module docstring.

    Deviation 1: no padding, so the first centre is t = m+1, not Tang's t = 1;
    in Tang's (l, j) indexing u_{j,l,i} = (m + i(l-1)m + j)/T.
    Deviation 2: whole blocks only, so up to i*m trailing samples are unused
    (Tang instead appends a ragged final block).
    """
    rng = np.random.default_rng(13)
    for T, m, thin in ((127, 4, 2), (256, 7, 3), (1000, 16, 2), (1024, 16, 2)):
        out = tang_moving_periodogram(rng.standard_normal(T), m=m, thin=thin)
        t_all = np.rint(out["u"] * T).astype(int)

        # Deviation 1: first retained centre is m + 1, never Tang's padded t = 1.
        assert t_all.min() == m + 1

        # Deviation 1: retained centres are exactly Tang's (l, j) grid plus m.
        n_blocks = (T - 2 * m) // (thin * m)
        expected_t = np.asarray(
            [m + thin * l * m + j for l in range(n_blocks) for j in range(1, m + 1)]
        )
        np.testing.assert_array_equal(t_all, expected_t)

        # Deviation 2: the dropped tail is the unfilled remainder plus the
        # (i-1)m stretch thinning would have skipped anyway.
        assert T - (t_all.max() + m) == (T - 2 * m) % (thin * m) + m * (thin - 1)
        assert T - (t_all.max() + m) < m * (2 * thin - 1)

    # The concrete figure quoted in the module docstring and in the paper.
    out = tang_moving_periodogram(rng.standard_normal(1000), m=16, thin=2)
    t_all = np.rint(out["u"] * 1000).astype(int)
    assert 1000 - (t_all.max() + 16) == 24


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
        binning_metadata={"time": {"method": "fixed"}},
    )
    assert result["psd_mean"].shape == (12, 8)
    assert np.isfinite(result["psd_mean"]).all()
    assert result["power_observations"]["counts"].sum() == 2 * len(
        result["ordinates"]["mi"]
    )
    recipe = result["provenance"]["binning"]
    assert recipe["output_shape"] == [
        int(np.ceil(recipe["input_shape"][0] / 2)),
        int(np.ceil(recipe["input_shape"][1] / 2)),
    ]
    assert recipe["time"]["bin_size"] == 2
    assert recipe["frequency"]["bin_size"] == 2
    assert recipe["selector"]["time"]["method"] == "fixed"


def test_dynamic_whittle_accepts_explicit_physical_knots() -> None:
    data = np.random.default_rng(31).standard_normal(256)
    config = PSplineConfig(
        n_interior_knots_time=2,
        n_interior_knots_freq=2,
        freq_knot_strategy="linear",
    )
    result = run_tang_dynamic_whittle_mcmc(
        data,
        dt=0.1,
        m=8,
        thin=2,
        config=config,
        interior_knots_time=np.array([0.4, 0.6]),
        interior_knots_freq=np.array([1.8, 3.2]),
        n_time_grid=8,
        n_warmup=4,
        n_samples=4,
        num_chains=1,
        random_seed=9,
    )
    np.testing.assert_allclose(
        result["knots_time_physical"][config.degree_time + 1:-(config.degree_time + 1)],
        [0.4, 0.6],
    )
    np.testing.assert_allclose(
        result["knots_freq_physical"][config.degree_freq + 1:-(config.degree_freq + 1)],
        [1.8, 3.2],
    )
    assert result["provenance"]["knot_allocation"] == {
        "time": "explicit",
        "frequency": "explicit",
    }
