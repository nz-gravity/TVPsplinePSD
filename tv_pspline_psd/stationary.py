"""Stationary (time-invariant) log-P-spline PSD baseline.

This is the comparison point for the non-stationary estimator: either a single
spectrum ``S(f)`` or a stationary multiplicative residual around a fixed
time-dependent reference ``S_ref(t,f)``. With no reference and
WDM coefficients ``w_nm ~ N(0, S_m)`` and ``S_m`` shared across all time bins, the
per-channel total power ``P_m = sum_n w_nm^2`` is sufficient, and the
log-likelihood is

    -0.5 * sum_m [ N_t * log S_m + P_m / S_m ].

With a reference, ``S_nm = S_ref,nm exp(g_m)`` and the same likelihood is
obtained by replacing ``w_nm^2`` with ``w_nm^2 / S_ref,nm``. This makes the
stationary and time-varying residual models differ only in whether their free
correction can vary in time.

The model is the frequency marginal of :mod:`tv_pspline_psd.model`: a whitened
1D P-spline on ``log S(f)`` with a Gamma-hyperprior smoothing precision. Fitting
this to genuinely non-stationary data forces a time-averaged spectrum, which is
exactly the bias the non-stationary estimator avoids.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np
import numpyro
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value

from .config import PSplineConfig
from .model import (
    _sample_log_gamma,
    eigen_prior_scale,
    power_floor,
    sample_eigen_coefficients,
)
from .splines import (
    create_bspline_basis,
    create_bspline_roughness_penalty,
    evaluate_bspline_basis,
)


def _stationary_model(total_power, counts, basis_eig_freq, lam_freq, null_freq, config):
    n_basis = basis_eig_freq.shape[1]
    phi = _sample_log_gamma("phi_freq", config.alpha_phi, config.beta_phi,
                            config.phi_log_base_scale)
    scale = eigen_prior_scale(
        jnp.asarray(0.0), phi, jnp.zeros(1), lam_freq,
        null_freq[None, :], config,
    )[0]
    eig_coeffs = sample_eigen_coefficients("s", scale, (n_basis,), config)
    log_psd = basis_eig_freq @ eig_coeffs
    log_like = -0.5 * jnp.sum(counts * log_psd + total_power * jnp.exp(-log_psd))
    numpyro.factor("stationary_whittle", log_like)
    numpyro.deterministic("log_psd", log_psd)


def run_stationary_psd_mcmc(
    coeffs: np.ndarray,
    freq_grid: np.ndarray,
    *,
    config: PSplineConfig,
    interior_knots_freq: np.ndarray | None = None,
    likelihood_mask: np.ndarray | None = None,
    freq_bin_starts: np.ndarray | None = None,
    log_psd_offset: np.ndarray | None = None,
    n_warmup: int = 300,
    n_samples: int = 400,
    num_chains: int = 1,
    random_seed: int = 7,
    max_tree_depth: int = 10,
    target_accept_prob: float = 0.85,
    progress_bar: bool = False,
) -> dict[str, object]:
    """Fit a stationary ``log S(f)`` to time-frequency coefficients.

    Args:
        coeffs: Real coefficients of shape ``(n_time, n_freq)`` (single component
            per cell, e.g. WDM).
        freq_grid: Channel frequencies (Hz), shape ``(n_freq,)``.
        interior_knots_freq: Optional explicit physical-frequency knots, shared
            with a time-varying comparison model.
        likelihood_mask: Optional retained-cell mask, shape ``(n_time, n_freq)``.
            Masked cells contribute neither power nor normalization counts.
        freq_bin_starts: Optional starts of common adaptive frequency bins.
            Powers and retained-cell counts are summed within each bin, exactly
            matching the coarse likelihood used by the time-varying model.
        log_psd_offset: Optional fixed reference log-PSD surface with shape
            ``(n_time, n_freq)``. When supplied, the inferred stationary spline
            is a frequency-only log-multiplicative residual ``g(f)`` in
            ``log S(t, f) = log S_ref(t, f) + g(f)``. The reference is not an
            assertion that the PSD is known: deviations in level and smooth
            spectral shape remain free through ``g``.

    Returns a dict with the 1D posterior spectrum and a ``psd_mean_surface`` that
    broadcasts it across the ``n_time`` rows for direct comparison with the
    non-stationary surface.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    freq_grid = np.asarray(freq_grid, dtype=float)
    if coeffs.ndim != 2:
        raise ValueError("coeffs must have shape (n_time, n_freq)")
    n_time, n_freq = coeffs.shape
    if freq_grid.shape != (n_freq,) or np.any(np.diff(freq_grid) <= 0.0):
        raise ValueError("freq_grid must be strictly increasing and match coeffs")
    if likelihood_mask is None:
        retained = np.ones_like(coeffs, dtype=bool)
    else:
        retained = np.asarray(likelihood_mask, dtype=bool)
        if retained.shape != coeffs.shape:
            raise ValueError("likelihood_mask must match coeffs")
    if log_psd_offset is None:
        reference_log_psd = np.zeros_like(coeffs)
        has_reference = False
    else:
        reference_log_psd = np.asarray(log_psd_offset, dtype=float)
        if reference_log_psd.shape != coeffs.shape:
            raise ValueError("log_psd_offset must match coeffs")
        if not np.all(np.isfinite(reference_log_psd)):
            raise ValueError("log_psd_offset must contain only finite values")
        has_reference = True

    # For S_nm = R_nm exp(g_m), terms involving only R_nm are constant in g.
    # The remaining likelihood has sufficient statistic
    # sum_n w_nm^2 / R_nm and the usual retained-cell count.
    reference_scaled_power = coeffs**2 * np.exp(-reference_log_psd)
    total_power_full = np.sum(
        np.where(retained, reference_scaled_power, 0.0), axis=0
    )
    counts_full = retained.sum(axis=0).astype(float)
    if not np.any(counts_full > 0.0):
        raise ValueError("likelihood_mask retains no cells")
    freq_unit = freq_grid / np.maximum(freq_grid[-1], 1e-12)

    explicit_knots_unit = None
    if interior_knots_freq is not None:
        explicit = np.asarray(interior_knots_freq, dtype=float)
        if explicit.shape != (config.n_interior_knots_freq,):
            raise ValueError("interior_knots_freq must match configured knot count")
        explicit_knots_unit = explicit / np.maximum(freq_grid[-1], 1e-12)

    B_freq, knots_freq = create_bspline_basis(
        freq_unit,
        config.n_interior_knots_freq,
        degree=config.degree_freq,
        interior_knots=explicit_knots_unit,
    )
    P_freq = create_bspline_roughness_penalty(
        knots_freq, degree=config.degree_freq, derivative_order=config.diff_order_freq)
    lam_f, U_f = np.linalg.eigh(P_freq)
    lam_f = np.clip(lam_f, 0.0, None)
    null_f = lam_f <= 1e-10 * max(lam_f.max(), 1.0)
    basis_eig_freq = B_freq @ U_f

    if freq_bin_starts is None:
        starts = np.arange(n_freq, dtype=int)
    else:
        starts = np.asarray(freq_bin_starts)
        if (
            starts.ndim != 1
            or starts.size == 0
            or starts[0] != 0
            or np.any(np.diff(starts) <= 0)
            or starts[-1] >= n_freq
            or not np.issubdtype(starts.dtype, np.integer)
        ):
            raise ValueError("freq_bin_starts must be increasing integer starts from zero")
        starts = starts.astype(int)
    total_power_binned = np.add.reduceat(total_power_full, starts)
    counts_binned = np.add.reduceat(counts_full, starts)
    stop = np.r_[starts[1:], n_freq]
    fit_frequency = np.asarray(
        [np.mean(freq_unit[left:right]) for left, right in zip(starts, stop, strict=True)]
    )
    active = counts_binned > 0.0
    total_power = total_power_binned[active]
    counts = counts_binned[active]
    basis_eig_freq_fit = evaluate_bspline_basis(
        fit_frequency[active], knots_freq, degree=config.degree_freq
    ) @ U_f

    # Warm start: penalized LS for the eigen-coordinate Z (log S = basis_eig @ Z),
    # then map to the whitened site s = Z * sqrt(phi*lam) to match the model.
    mean_power = total_power / counts
    floor = power_floor(mean_power)
    target = np.log(mean_power + floor)
    system = (basis_eig_freq_fit.T @ basis_eig_freq_fit
              + config.init_penalty_freq * np.diag(np.where(null_f, 0.0, lam_f))
              + config.ridge_eps * np.eye(basis_eig_freq_fit.shape[1]))
    z_init = np.linalg.solve(system, basis_eig_freq_fit.T @ target)
    phi_init = max(1e-2, z_init.size / (float(np.sum(lam_f * z_init**2)) + 1e-6))
    if config.centered:
        s_init = z_init
    else:
        inv_scale = np.where(
            null_f, np.sqrt(config.null_precision),
            np.sqrt(phi_init * lam_f + config.ridge_eps),
        )
        s_init = z_init * inv_scale
    init_sites = {"s": s_init, "phi_freq": float(np.log(phi_init))}

    kernel = NUTS(_stationary_model, init_strategy=init_to_value(values=init_sites),
                  max_tree_depth=max_tree_depth, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress_bar,
    )
    nuts_started = time.perf_counter()
    mcmc.run(random.PRNGKey(random_seed),
             jnp.asarray(total_power), jnp.asarray(counts), jnp.asarray(basis_eig_freq_fit),
             jnp.asarray(lam_f), jnp.asarray(null_f), config,
             extra_fields=("diverging", "accept_prob", "num_steps", "potential_energy"))
    nuts_runtime_s = time.perf_counter() - nuts_started

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    # The sampled deterministic site lives on the binned likelihood grid.
    # Reconstruct on the full physical grid for a matched surface comparison.
    eig_coefficients = samples["s"]
    if not config.centered:
        phi = np.exp(samples["phi_freq"])
        scale = np.where(
            null_f[None, :],
            1.0 / np.sqrt(config.null_precision),
            1.0 / np.sqrt(phi[:, None] * lam_f[None, :] + config.ridge_eps),
        )
        eig_coefficients = eig_coefficients * scale
    residual_log_psd = eig_coefficients @ basis_eig_freq.T
    residual_log_mean = residual_log_psd.mean(axis=0)
    residual_log_lower = np.percentile(residual_log_psd, 5.0, axis=0)
    residual_log_upper = np.percentile(residual_log_psd, 95.0, axis=0)
    residual_psd_geometric_mean = np.exp(residual_log_mean)
    total_log_mean_surface = reference_log_psd + residual_log_mean[None, :]
    total_log_lower_surface = reference_log_psd + residual_log_lower[None, :]
    total_log_upper_surface = reference_log_psd + residual_log_upper[None, :]
    psd_geometric_mean_surface = np.exp(total_log_mean_surface)
    psd_lower_surface = np.exp(total_log_lower_surface)
    psd_upper_surface = np.exp(total_log_upper_surface)
    return {
        "mcmc": mcmc,
        "samples": samples,
        "freq_grid": np.asarray(freq_grid),
        "basis_eig_freq": basis_eig_freq,
        "eig_coeff_samples": eig_coefficients,
        "likelihood_mask": retained,
        "freq_bin_starts": starts,
        "likelihood_frequency_bins": int(active.sum()),
        "reference_applied": has_reference,
        "log_psd_offset": reference_log_psd if has_reference else None,
        "residual_log_psd_mean": residual_log_mean,
        "residual_log_psd_samples": residual_log_psd,
        "residual_psd_geometric_mean": residual_psd_geometric_mean,
        # Backwards-compatible 1D aliases describe the stationary residual.
        # With no reference, the residual is the total stationary spectrum.
        "log_psd_mean": residual_log_mean,
        "psd_geometric_mean": residual_psd_geometric_mean,
        "psd_mean": residual_psd_geometric_mean,
        "psd_lower": np.exp(residual_log_lower),
        "psd_upper": np.exp(residual_log_upper),
        # Surface-valued outputs always describe the total PSD, including the
        # fixed reference when one is supplied.
        "psd_geometric_mean_surface": psd_geometric_mean_surface,
        "psd_mean_surface": psd_geometric_mean_surface,
        "psd_lower_surface": psd_lower_surface,
        "psd_upper_surface": psd_upper_surface,
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
        "nuts_runtime_s": float(nuts_runtime_s),
    }
