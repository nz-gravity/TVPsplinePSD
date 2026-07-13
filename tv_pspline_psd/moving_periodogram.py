"""Tang-style zigzag moving periodogram + thinned dynamic Whittle.

Faithful to the construction of Tang et al. (the ``beyondWhittle`` dynamic
Whittle): a moving periodogram ordinate is formed from a centred ``2m+1``-point
window and evaluated at a single Fourier frequency ``lambda_{mod(t)}`` that
*cycles (zigzags)* with the time index,

    MI_t = |sum_{nu=0}^{2m} X_{nu+t-m} exp(-i pi nu lambda_{mod(t)})|^2
           / (2 pi (2m+1)),   lambda_j = 2j/(2m+1),  mod(t) = 1 + ((t-1) mod m).

The thinned variant keeps blocks of ``m`` ordinates (one per frequency) spaced by
``i*m`` to reduce correlation. The resulting *scattered* ``(u, omega)`` ordinates
are fit with the SAME whitened tensor-product P-spline prior used for WDM, under
the exponential (dynamic) Whittle likelihood ``MI ~ Exp(1/f)``. Only the
time-frequency representation differs from the WDM estimator.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value

from .config import PSplineConfig
from .model import (
    _sample_log_gamma,
    power_floor,
    whiten_penalty_pair,
    whitened_init_values,
)
from .splines import (
    create_bspline_basis,
    create_bspline_roughness_penalty,
    evaluate_bspline_basis,
)


def tang_moving_periodogram(
    data: np.ndarray, *, m: int, thin: int = 2
) -> dict[str, np.ndarray]:
    """Thinned zigzag moving-periodogram ordinates (Tang et al.).

    Args:
        data: Real time series of length ``T``.
        m: Order (window half-width; window length ``2m+1``, ``m`` frequencies).
        thin: Thinning factor ``i`` (2 or 3); blocks are spaced by ``i*m``.

    Returns:
        Dict with scattered arrays ``u`` (rescaled time in ``(0,1)``), ``omega``
        (angular frequency in ``(0, pi)``), and ``mi`` (the ordinates).
    """
    x = np.asarray(data, dtype=float)
    T = x.size
    n_blocks = (T - 2 * m) // (thin * m)
    if n_blocks < 1:
        raise ValueError("Series too short for these (m, thin).")

    nu = np.arange(2 * m + 1)
    j = np.arange(1, m + 1)
    lam = 2.0 * j / (2 * m + 1)        # Fourier frequencies in (0, 1)
    omega = np.pi * lam               # angular frequency in (0, pi)
    phase = np.exp(-1j * np.pi * np.outer(nu, lam))  # (2m+1, m)

    # ``windows[k]`` is the length-(2m+1) window whose 1-based centre is
    # ``t = k + m + 1``.  The retained window starts form an
    # ``(n_blocks, m)`` array, preserving the original block-major ordering.
    windows = np.lib.stride_tricks.sliding_window_view(x, 2 * m + 1)
    window_starts = (
        thin * m * np.arange(n_blocks)[:, None] + np.arange(m)[None, :]
    ).reshape(-1)
    freq_index = np.tile(np.arange(m), n_blocks)

    # Advanced indexing materializes the selected windows and phase rows, so
    # process bounded chunks rather than allocating an O(T*m) temporary.  The
    # target accounts for one float window and one complex phase row per point.
    bytes_per_point = (2 * m + 1) * (x.dtype.itemsize + phase.dtype.itemsize)
    chunk_points = max(1, (64 * 1024**2) // max(bytes_per_point, 1))
    coeff = np.empty(window_starts.size, dtype=np.complex128)
    phase_by_freq = phase.T
    for start in range(0, window_starts.size, chunk_points):
        stop = min(start + chunk_points, window_starts.size)
        selected_windows = windows[window_starts[start:stop]]
        selected_phase = phase_by_freq[freq_index[start:stop]]
        coeff[start:stop] = np.einsum(
            "pn,pn->p", selected_windows, selected_phase, optimize=True
        )

    t = window_starts + m + 1
    return {
        "u": t / T,
        "omega": np.tile(omega, n_blocks),
        "mi": np.abs(coeff) ** 2 / (2.0 * np.pi * (2 * m + 1)),
    }


def _grouped_normal_equations(target, B_time, B_freq_unique):
    """Return ``X.T @ X`` and ``X.T @ target`` without forming scattered ``X``.

    For frequency rung ``j``, every design row is
    ``kron(B_freq[j], B_time[p])``.  Accumulating the Kronecker products of the
    small per-rung time Grams avoids the ``P x (K_t*K_f)`` design matrix.
    """
    n_freq, n_basis_freq = B_freq_unique.shape
    n_basis_time = B_time.shape[1]
    if target.size % n_freq:
        raise ValueError("Tang ordinate count must be divisible by its frequencies.")
    time_grouped = B_time.reshape((-1, n_freq, n_basis_time))
    target_grouped = target.reshape((-1, n_freq))
    n_weights = n_basis_time * n_basis_freq
    gram = np.zeros((n_weights, n_weights))
    rhs = np.zeros(n_weights)
    for j in range(n_freq):
        bt = time_grouped[:, j, :]
        bf = B_freq_unique[j]
        gram += np.kron(np.outer(bf, bf), bt.T @ bt)
        rhs += np.kron(bf, bt.T @ target_grouped[:, j])
    return gram, rhs


def _scattered_pls_init(mi, B_time, B_freq_unique, P_time, P_freq, config):
    """Penalized least-squares warm start on scattered log-ordinates."""
    n_t, n_f = B_time.shape[1], B_freq_unique.shape[1]
    floor = power_floor(mi)
    target = np.log(mi + floor)
    gram, rhs = _grouped_normal_equations(target, B_time, B_freq_unique)
    kron_time = np.kron(np.eye(n_f), P_time)
    kron_freq = np.kron(P_freq, np.eye(n_t))
    system = (
        gram
        + config.init_penalty_time * kron_time
        + config.init_penalty_freq * kron_freq
        + config.ridge_eps * np.eye(n_f * n_t)
    )
    w = np.linalg.solve(system, rhs)
    W = w.reshape((n_t, n_f), order="F")
    phi_time = max(1e-2, w.size / (float(w @ kron_time @ w) + 1e-6))
    phi_freq = max(1e-2, w.size / (float(w @ kron_freq @ w) + 1e-6))
    return {"W": W, "phi_time": phi_time, "phi_freq": phi_freq}


def _grouped_log_surface(basis_eig_time, basis_eig_freq_unique, eig_coeffs):
    """Evaluate the tensor surface at block-major Tang ordinates.

    Tang repeats the same ``m`` Fourier frequencies in every retained block.
    Contracting ``W`` with those unique frequency rows first changes the main
    evaluation from ``O(P*K_t*K_f)`` to ``O(m*K_t*K_f + P*K_t)``.
    """
    n_freq = basis_eig_freq_unique.shape[0]
    n_basis_time = basis_eig_time.shape[1]
    basis_time_grouped = basis_eig_time.reshape((-1, n_freq, n_basis_time))
    coeffs_by_freq = (eig_coeffs @ basis_eig_freq_unique.T).T
    return jnp.sum(
        basis_time_grouped * coeffs_by_freq[None, :, :], axis=2
    ).reshape(-1)


def _dynamic_whittle_model(mi, basis_eig_time, basis_eig_freq_unique, lam_time, lam_freq,
                           joint_null, config):
    n_t, n_f = basis_eig_time.shape[1], basis_eig_freq_unique.shape[1]
    phi_time = _sample_log_gamma("phi_time", config.alpha_phi, config.beta_phi,
                                 config.phi_log_base_scale)
    phi_freq = _sample_log_gamma("phi_freq", config.alpha_phi, config.beta_phi,
                                 config.phi_log_base_scale)
    d = phi_time * lam_time[:, None] + phi_freq * lam_freq[None, :]
    scale = jnp.where(joint_null, 1.0 / jnp.sqrt(config.null_precision),
                      1.0 / jnp.sqrt(d + config.ridge_eps))
    with numpyro.plate("eig_plate", n_t * n_f):
        s_flat = numpyro.sample("s", dist.Normal(0.0, 1.0))
    eig_coeffs = s_flat.reshape((n_t, n_f)) * scale
    # Scattered evaluation at each ordinate's own (u_p, omega_p), grouped by
    # the m repeated frequency rungs.
    log_f = _grouped_log_surface(basis_eig_time, basis_eig_freq_unique, eig_coeffs)
    log_like = jnp.sum(-(log_f + mi * jnp.exp(-log_f)))
    numpyro.factor("dynamic_whittle", log_like)
    numpyro.deterministic("eig_coeffs", eig_coeffs)


def run_tang_dynamic_whittle_mcmc(
    data: np.ndarray,
    *,
    dt: float,
    m: int,
    thin: int = 2,
    config: PSplineConfig,
    n_time_grid: int = 60,
    n_warmup: int = 250,
    n_samples: int = 300,
    num_chains: int = 1,
    random_seed: int = 7,
) -> dict[str, object]:
    """Fit the thinned dynamic-Whittle model and evaluate the PSD on a grid."""
    ordinates = tang_moving_periodogram(data, m=m, thin=thin)
    u, omega, mi = ordinates["u"], ordinates["omega"], ordinates["mi"]
    freq_unit = omega / np.pi  # in (0, 1)

    B_time, knots_time = create_bspline_basis(
        u, config.n_interior_knots_time, degree=config.degree_time
    )
    unique_freq_unit = np.unique(freq_unit)
    B_freq_unique, knots_freq = create_bspline_basis(
        unique_freq_unit, config.n_interior_knots_freq, degree=config.degree_freq
    )
    B_freq = evaluate_bspline_basis(
        freq_unit, knots_freq, degree=config.degree_freq
    )
    P_time = create_bspline_roughness_penalty(
        knots_time, degree=config.degree_time, derivative_order=config.diff_order_time
    )
    P_freq = create_bspline_roughness_penalty(
        knots_freq, degree=config.degree_freq, derivative_order=config.diff_order_freq
    )
    whitened = whiten_penalty_pair(P_time, P_freq)
    basis_eig_time = B_time @ whitened["U_time"]
    basis_eig_freq = B_freq @ whitened["U_freq"]
    basis_eig_freq_unique = B_freq_unique @ whitened["U_freq"]

    pls_init = _scattered_pls_init(
        mi, B_time, B_freq_unique, P_time, P_freq, config
    )
    init_sites = whitened_init_values(pls_init, whitened, config)

    kernel = NUTS(
        _dynamic_whittle_model,
        init_strategy=init_to_value(values=init_sites),
        max_tree_depth=10, target_accept_prob=0.85,
    )
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=num_chains, chain_method="sequential", progress_bar=False)
    nuts_t0 = time.perf_counter()
    mcmc.run(
        random.PRNGKey(random_seed),
        jnp.asarray(mi), jnp.asarray(basis_eig_time), jnp.asarray(basis_eig_freq_unique),
        jnp.asarray(whitened["lam_time"]), jnp.asarray(whitened["lam_freq"]),
        jnp.asarray(whitened["joint_null"]), config,
        extra_fields=("diverging",),
    )
    nuts_runtime_s = time.perf_counter() - nuts_t0

    # Evaluate the posterior PSD on a regular (u, omega) grid for comparison.
    eig_samples = np.asarray(mcmc.get_samples()["eig_coeffs"])  # (n, K_t, K_f)
    dense_u = np.linspace(u.min(), u.max(), n_time_grid)
    omega_grid = np.unique(omega)                       # the m Fourier frequencies
    freq_grid_hz = omega_grid / (2.0 * np.pi * dt)
    BUt = evaluate_bspline_basis(dense_u, knots_time, degree=config.degree_time) @ whitened["U_time"]
    BUf = evaluate_bspline_basis(omega_grid / np.pi, knots_freq, degree=config.degree_freq) @ whitened["U_freq"]
    log_psd_grid = np.einsum(
        "ta,nab,fb->ntf", BUt, eig_samples, BUf, optimize=True
    )

    return {
        "mcmc": mcmc,
        "ordinates": ordinates,
        "time_grid": dense_u,
        "freq_grid": freq_grid_hz,
        "omega_grid": omega_grid,
        "psd_mean": np.exp(np.mean(log_psd_grid, axis=0)),
        "psd_lower": np.exp(np.percentile(log_psd_grid, 5.0, axis=0)),
        "psd_upper": np.exp(np.percentile(log_psd_grid, 95.0, axis=0)),
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
        "nuts_runtime_s": float(nuts_runtime_s),
    }
