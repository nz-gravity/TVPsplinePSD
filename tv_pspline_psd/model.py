"""Whitened tensor-product log-P-spline WDM Whittle model.

The latent surface is ``log S = B_t W B_f^T`` with an anisotropic roughness
prior on ``vec(W)``:

    Q(phi_t, phi_f) = phi_t (I_f kron P_t) + phi_f (P_f kron I_t).

Because ``(I_f kron P_t)`` and ``(P_f kron I_t)`` are simultaneously
diagonalized by the marginal penalty eigenvectors ``U_t, U_f``, the joint
precision is diagonal in the tensor eigenbasis with eigenvalues

    d[a, b] = phi_t * lam_t[a] + phi_f * lam_f[b].

The eigen-coefficients may be sampled in centered or non-centered form. Both
parameterizations use the same prior scale and expose the same reconstructed
coefficients to callers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .config import PSplineConfig


def power_floor(power: np.ndarray) -> float:
    """Scale-free floor for ``log(power + floor)`` targets.

    A small fraction of a low percentile of the *nonzero* power, so the floor
    tracks the data scale (absolute floors like 1e-8 swamp tiny-amplitude data,
    e.g. LISA fractional-frequency series with power ~1e-40).
    """
    positive = power[power > 0]
    if positive.size == 0:
        return 1.0
    return 0.05 * float(np.percentile(positive, 10.0))


def whiten_penalty_pair(
    penalty_time: np.ndarray,
    penalty_freq: np.ndarray,
    *,
    null_tol: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Eigendecompose the marginal penalties for the non-centered tensor prior.

    Returns the eigenvalues/eigenvectors of each marginal penalty and the joint
    null-space mask (where both marginal eigenvalues vanish), which the model
    treats with a fixed weak prior instead of the ``phi``-scaled penalty.
    """
    lam_t, U_t = np.linalg.eigh(penalty_time)
    lam_f, U_f = np.linalg.eigh(penalty_freq)
    lam_t = np.clip(lam_t, 0.0, None)
    lam_f = np.clip(lam_f, 0.0, None)
    null_t = lam_t <= null_tol * max(lam_t.max(), 1.0)
    null_f = lam_f <= null_tol * max(lam_f.max(), 1.0)
    joint_null = np.outer(null_t, null_f)
    return {
        "lam_time": lam_t,
        "lam_freq": lam_f,
        "U_time": U_t,
        "U_freq": U_f,
        "joint_null": joint_null,
    }


def _sample_log_gamma(
    name: str,
    alpha: float,
    beta: float,
    base_scale: float,
) -> jnp.ndarray:
    """Sample ``phi`` with a ``Gamma(alpha, beta)`` prior on the log scale.

    The site itself is ``log phi`` with a broad Normal reference measure; a
    ``factor`` corrects the density to the exact ``Gamma`` prior (with the
    log-Jacobian), giving an unconstrained, well-scaled sampling variable. This
    mirrors the approach used in ``log_psplines``.
    """
    base = dist.Normal(0.0, base_scale)
    log_phi = numpyro.sample(name, base)
    phi = jnp.exp(log_phi)
    gamma = dist.Gamma(alpha, beta)
    numpyro.factor(
        f"{name}_prior",
        gamma.log_prob(phi) + log_phi - base.log_prob(log_phi),
    )
    return phi


def eigen_prior_scale(
    phi_time: jnp.ndarray,
    phi_freq: jnp.ndarray,
    lam_time: jnp.ndarray,
    lam_freq: jnp.ndarray,
    joint_null: jnp.ndarray,
    config: PSplineConfig,
) -> jnp.ndarray:
    """Return the tensor eigen-coefficient prior scale.

    The fixed scale on the joint penalty null space makes the otherwise
    improper P-spline prior proper. One-dimensional callers can pass a
    singleton zero-valued time axis and remove it from the returned array.
    """
    precision = (
        phi_time * lam_time[:, None]
        + phi_freq * lam_freq[None, :]
    )
    return jnp.where(
        joint_null,
        1.0 / jnp.sqrt(config.null_precision),
        1.0 / jnp.sqrt(precision + config.ridge_eps),
    )


def sample_eigen_coefficients(
    name: str,
    scale: jnp.ndarray,
    shape: tuple[int, ...],
    config: PSplineConfig,
) -> jnp.ndarray:
    """Sample eigen-coefficients while honoring ``config.centered``.

    In centered form the sampling site is the coefficient itself and has prior
    ``Normal(0, scale)``. In non-centered form the site is standard Normal and
    is multiplied by ``scale`` before being returned.
    """
    n_weights = int(np.prod(shape))
    flat_scale = jnp.broadcast_to(scale, shape).reshape(-1)
    with numpyro.plate(f"{name}_plate", n_weights):
        if config.centered:
            site = numpyro.sample(name, dist.Normal(0.0, flat_scale))
            coeffs = site
        else:
            site = numpyro.sample(name, dist.Normal(0.0, 1.0))
            coeffs = site * flat_scale
    return coeffs.reshape(shape)


def pspline_surface_model(
    summed_power: jnp.ndarray,
    counts: jnp.ndarray,
    basis_eig_time: jnp.ndarray,
    basis_eig_freq: jnp.ndarray,
    lam_time: jnp.ndarray,
    lam_freq: jnp.ndarray,
    joint_null: jnp.ndarray,
    config: PSplineConfig,
    store_surface: bool = True,
) -> None:
    """Whitened tensor-product log-P-spline model with a Gaussian Whittle likelihood.

    The likelihood is ``c ~ N(0, S)`` on each real time-frequency coefficient, so
    a cell with ``R`` real components (``R = 1`` for WDM, ``R = 2`` for the real
    and imaginary parts of an STFT coefficient) contributes

        -0.5 * ( R * log S + (sum_r c_r^2) / S ).

    If ``S`` is constant over a block of ``m`` consecutive time bins (as in
    time-binned coarse-graining), summing the block's power gives the identical
    likelihood form with ``R -> m * R`` and the power summed over the block --
    the block power is ``S`` times a ``chi^2_{mR}`` variate. ``summed_power`` and
    ``counts`` already encode this: pass the raw per-cell ``sum_r c_r^2`` and
    ``R`` for the unbinned likelihood, or block-summed power and ``m * R`` for
    the coarse-grained one.

    Args:
        summed_power: Per-cell (optionally time-block-summed) squared power
            ``sum_r c_r^2``, shape ``(n_time, n_freq)`` (``n_time`` is the
            number of time blocks when coarse-graining).
        counts: Effective real-component count per cell, broadcastable to
            ``summed_power``'s shape (``R`` unbinned, ``m_T * R`` for a
            time block of ``m_T`` cells).
        basis_eig_time: ``B_t U_t`` of shape ``(n_time, K_t)``, evaluated on
            the same time grid as ``summed_power``/``counts``.
        basis_eig_freq: ``B_f U_f`` of shape ``(n_freq, K_f)``.
        lam_time: Eigenvalues of the time penalty, shape ``(K_t,)``.
        lam_freq: Eigenvalues of the frequency penalty, shape ``(K_f,)``.
        joint_null: Boolean mask ``(K_t, K_f)`` of the joint penalty null space.
        config: Estimator configuration.
    """
    n_basis_time = basis_eig_time.shape[1]
    n_basis_freq = basis_eig_freq.shape[1]

    phi_time = _sample_log_gamma(
        "phi_time", config.alpha_phi, config.beta_phi, config.phi_log_base_scale
    )
    phi_freq = _sample_log_gamma(
        "phi_freq", config.alpha_phi, config.beta_phi, config.phi_log_base_scale
    )

    scale = eigen_prior_scale(
        phi_time, phi_freq, lam_time, lam_freq, joint_null, config
    )
    eig_coeffs = sample_eigen_coefficients(
        "s", scale, (n_basis_time, n_basis_freq), config
    )

    log_psd = basis_eig_time @ eig_coeffs @ basis_eig_freq.T

    log_like = -0.5 * jnp.sum(
        counts * log_psd + summed_power * jnp.exp(-log_psd)
    )
    numpyro.factor("whittle", log_like)
    if store_surface:
        # Storing the full surface per sample is convenient but O(n_time*n_freq)
        # memory; large grids reconstruct it from the eigen-coefficients instead.
        numpyro.deterministic("log_psd", log_psd)


def initialize_with_penalized_least_squares(
    observed_power: np.ndarray,
    B_time: np.ndarray,
    B_freq: np.ndarray,
    penalty_time: np.ndarray,
    penalty_freq: np.ndarray,
    config: PSplineConfig,
) -> dict[str, np.ndarray | float]:
    """Penalized least-squares warm start in the *coefficient* basis.

    Returns the fitted coefficient matrix ``W`` and heuristic smoothing
    precisions; :func:`whitened_init_values` converts these to the model sites.
    """
    floor = power_floor(observed_power)
    target = np.log(observed_power + floor)
    kron_time = np.kron(np.eye(B_freq.shape[1]), penalty_time)
    kron_freq = np.kron(penalty_freq, np.eye(B_time.shape[1]))
    # The design is kron(B_freq, B_time), so its Gram is the Kronecker of the
    # per-axis Grams and the projection is B_t^T Y B_f: never form the
    # O(n_time*n_freq x n_basis) design matrix.
    n_basis = B_time.shape[1] * B_freq.shape[1]
    system = (
        np.kron(B_freq.T @ B_freq, B_time.T @ B_time)
        + config.init_penalty_time * kron_time
        + config.init_penalty_freq * kron_freq
        + config.ridge_eps * np.eye(n_basis)
    )
    rhs = (B_time.T @ target @ B_freq).reshape(-1, order="F")
    weights = np.linalg.solve(system, rhs)
    W_fit = weights.reshape((B_time.shape[1], B_freq.shape[1]), order="F")
    fitted = B_time @ W_fit @ B_freq.T

    penalty_time_energy = float(weights @ kron_time @ weights)
    penalty_freq_energy = float(weights @ kron_freq @ weights)
    phi_time_init = max(1e-2, fitted.size / (penalty_time_energy + 1e-6))
    phi_freq_init = max(1e-2, fitted.size / (penalty_freq_energy + 1e-6))

    return {
        "W": W_fit,
        "phi_time": phi_time_init,
        "phi_freq": phi_freq_init,
        "log_psd": fitted,
    }


def whitened_init_values(
    pls_init: dict[str, np.ndarray | float],
    whitened: dict[str, np.ndarray],
    config: PSplineConfig,
) -> dict[str, np.ndarray]:
    """Map a penalized-least-squares fit to the whitened model's sampling sites."""
    U_t = whitened["U_time"]
    U_f = whitened["U_freq"]
    lam_t = whitened["lam_time"]
    lam_f = whitened["lam_freq"]
    joint_null = whitened["joint_null"]

    phi_time = float(pls_init["phi_time"])
    phi_freq = float(pls_init["phi_freq"])
    eig_coeffs = U_t.T @ np.asarray(pls_init["W"]) @ U_f  # Z in the eigenbasis

    if config.centered:
        s = eig_coeffs
    else:
        d = phi_time * lam_t[:, None] + phi_freq * lam_f[None, :]
        inv_scale = np.where(
            joint_null,
            np.sqrt(config.null_precision),
            np.sqrt(d + config.ridge_eps),
        )
        s = eig_coeffs * inv_scale
    return {
        "s": s.reshape(-1),
        "phi_time": float(np.log(phi_time)),
        "phi_freq": float(np.log(phi_freq)),
    }
