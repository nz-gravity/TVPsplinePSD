"""Stationary (time-invariant) log-P-spline PSD baseline.

This is the comparison point for the non-stationary estimator: a single spectrum
``S(f)``, constant in time, fitted with the stationary Whittle likelihood. With
WDM coefficients ``w_nm ~ N(0, S_m)`` and ``S_m`` shared across all time bins, the
per-channel total power ``P_m = sum_n w_nm^2`` is sufficient, and the
log-likelihood is

    -0.5 * sum_m [ N_t * log S_m + P_m / S_m ].

The model is the frequency marginal of :mod:`tv_pspline_psd.model`: a whitened
1D P-spline on ``log S(f)`` with a Gamma-hyperprior smoothing precision. Fitting
this to genuinely non-stationary data forces a time-averaged spectrum, which is
exactly the bias the non-stationary estimator avoids.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value

from .config import PSplineConfig
from .model import _sample_log_gamma
from .splines import create_bspline_basis, create_bspline_roughness_penalty


def _stationary_model(total_power, n_time, basis_eig_freq, lam_freq, null_freq, config):
    n_basis = basis_eig_freq.shape[1]
    phi = _sample_log_gamma("phi_freq", config.alpha_phi, config.beta_phi,
                            config.phi_log_base_scale)
    scale = jnp.where(null_freq, 1.0 / jnp.sqrt(config.null_precision),
                      1.0 / jnp.sqrt(phi * lam_freq + config.ridge_eps))
    with numpyro.plate("eig_coeffs", n_basis):
        s = numpyro.sample("s", dist.Normal(0.0, 1.0))
    log_psd = basis_eig_freq @ (s * scale)
    log_like = -0.5 * jnp.sum(n_time * log_psd + total_power * jnp.exp(-log_psd))
    numpyro.factor("stationary_whittle", log_like)
    numpyro.deterministic("log_psd", log_psd)


def run_stationary_psd_mcmc(
    coeffs: np.ndarray,
    freq_grid: np.ndarray,
    *,
    config: PSplineConfig,
    n_warmup: int = 300,
    n_samples: int = 400,
    random_seed: int = 7,
    max_tree_depth: int = 10,
    target_accept_prob: float = 0.85,
) -> dict[str, object]:
    """Fit a stationary ``log S(f)`` to time-frequency coefficients.

    Args:
        coeffs: Real coefficients of shape ``(n_time, n_freq)`` (single component
            per cell, e.g. WDM).
        freq_grid: Channel frequencies (Hz), shape ``(n_freq,)``.

    Returns a dict with the 1D posterior spectrum and a ``psd_mean_surface`` that
    broadcasts it across the ``n_time`` rows for direct comparison with the
    non-stationary surface.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    n_time, n_freq = coeffs.shape
    total_power = np.sum(coeffs**2, axis=0)
    freq_unit = freq_grid / np.maximum(freq_grid[-1], 1e-12)

    B_freq, knots_freq = create_bspline_basis(
        freq_unit, config.n_interior_knots_freq, degree=config.degree_freq)
    P_freq = create_bspline_roughness_penalty(
        knots_freq, degree=config.degree_freq, derivative_order=config.diff_order_freq)
    lam_f, U_f = np.linalg.eigh(P_freq)
    lam_f = np.clip(lam_f, 0.0, None)
    null_f = lam_f <= 1e-10 * max(lam_f.max(), 1.0)
    basis_eig_freq = B_freq @ U_f

    # Warm start: penalized LS for the eigen-coordinate Z (log S = basis_eig @ Z),
    # then map to the whitened site s = Z * sqrt(phi*lam) to match the model.
    mean_power = total_power / n_time
    floor = max(1e-12, 0.05 * np.percentile(mean_power, 10.0))
    target = np.log(mean_power + floor)
    system = (basis_eig_freq.T @ basis_eig_freq
              + config.init_penalty_freq * np.diag(np.where(null_f, 0.0, lam_f))
              + config.ridge_eps * np.eye(basis_eig_freq.shape[1]))
    z_init = np.linalg.solve(system, basis_eig_freq.T @ target)
    phi_init = max(1e-2, z_init.size / (float(np.sum(lam_f * z_init**2)) + 1e-6))
    inv_scale = np.where(null_f, np.sqrt(config.null_precision),
                         np.sqrt(phi_init * lam_f + config.ridge_eps))
    init_sites = {"s": z_init * inv_scale, "phi_freq": float(np.log(phi_init))}

    kernel = NUTS(_stationary_model, init_strategy=init_to_value(values=init_sites),
                  max_tree_depth=max_tree_depth, target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, progress_bar=False)
    mcmc.run(random.PRNGKey(random_seed),
             jnp.asarray(total_power), float(n_time), jnp.asarray(basis_eig_freq),
             jnp.asarray(lam_f), jnp.asarray(null_f), config,
             extra_fields=("diverging",))

    log_psd = np.asarray(mcmc.get_samples()["log_psd"])  # (n_samples, n_freq)
    log_mean = log_psd.mean(axis=0)
    psd_mean = np.exp(log_mean)
    return {
        "freq_grid": np.asarray(freq_grid),
        "psd_mean": psd_mean,
        "psd_lower": np.exp(np.percentile(log_psd, 5.0, axis=0)),
        "psd_upper": np.exp(np.percentile(log_psd, 95.0, axis=0)),
        "psd_mean_surface": np.broadcast_to(psd_mean, (n_time, n_freq)).copy(),
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
    }
