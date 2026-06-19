"""Joint signal + non-stationary-noise inference in the time-frequency domain.

Extends the noise estimator with a coherent signal mean: the coefficients are
modelled as ``c ~ N(h(beta), S)`` with ``h = sum_k beta_k g_k`` a linear
combination of (precomputed) template coefficients ``g_k``. The whitened P-spline
noise prior is unchanged, so the noise PSD and the signal amplitudes are inferred
jointly -- the wavelet-domain analogue of a global fit. This is exactly what the
Gaussian *coefficient* likelihood enables; a power periodogram could not.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value

from .config import PSplineConfig
from .inference import reconstruct_eig_coeff_samples, surface_summaries
from .model import (
    _sample_log_gamma,
    initialize_with_penalized_least_squares,
    pspline_surface_model,
    whiten_penalty_pair,
    whitened_init_values,
)
from .splines import (
    create_bspline_basis,
    create_bspline_roughness_penalty,
)


def _joint_model(coeffs, templates, basis_eig_time, basis_eig_freq,
                 lam_time, lam_freq, joint_null, amp_scale, config,
                 store_surface=True):
    n_basis_time = basis_eig_time.shape[1]
    n_basis_freq = basis_eig_freq.shape[1]

    phi_time = _sample_log_gamma("phi_time", config.alpha_phi, config.beta_phi,
                                 config.phi_log_base_scale)
    phi_freq = _sample_log_gamma("phi_freq", config.alpha_phi, config.beta_phi,
                                 config.phi_log_base_scale)
    d = phi_time * lam_time[:, None] + phi_freq * lam_freq[None, :]
    scale = jnp.where(joint_null, 1.0 / jnp.sqrt(config.null_precision),
                      1.0 / jnp.sqrt(d + config.ridge_eps))
    n_weights = n_basis_time * n_basis_freq
    with numpyro.plate("eig_coeffs", n_weights):
        s_flat = numpyro.sample("s", dist.Normal(0.0, 1.0))
    log_psd = basis_eig_time @ (s_flat.reshape((n_basis_time, n_basis_freq)) * scale) @ basis_eig_freq.T

    with numpyro.plate("templates", templates.shape[0]):
        beta = numpyro.sample("beta", dist.Normal(0.0, amp_scale))
    signal = jnp.tensordot(beta, templates, axes=1)  # (n_time, n_freq)

    # Single real coefficient per cell (WDM): w ~ N(signal, S).
    resid = coeffs - signal
    log_like = -0.5 * jnp.sum(log_psd + resid**2 * jnp.exp(-log_psd))
    numpyro.factor("whittle", log_like)
    if store_surface:
        numpyro.deterministic("log_psd", log_psd)


def run_joint_signal_noise_mcmc(
    coeffs: np.ndarray,
    templates: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    *,
    config: PSplineConfig,
    amp_scale: float | None = None,
    n_warmup: int = 400,
    n_samples: int = 400,
    num_chains: int = 1,
    random_seed: int = 7,
    max_tree_depth: int = 10,
    target_accept_prob: float = 0.9,
    store_log_psd_samples: bool = True,
) -> dict[str, object]:
    """Jointly infer the noise PSD surface and the signal amplitudes.

    Args:
        coeffs: WDM coefficients of shape ``(n_time, n_freq)``.
        templates: Template coefficients of shape ``(K, n_time, n_freq)``; the
            signal mean is ``sum_k beta_k * templates[k]``.
        time_grid: Rescaled time grid in ``[0, 1]``.
        freq_grid: Frequencies (Hz).
        config: Estimator configuration.
        amp_scale: Prior scale for the amplitudes ``beta`` (default: data RMS).
        store_log_psd_samples: see :func:`fit_log_pspline_surface`.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    templates = np.asarray(templates, dtype=float)
    power = coeffs**2
    freq_unit = freq_grid / np.maximum(freq_grid[-1], 1e-12)
    if amp_scale is None:
        amp_scale = float(np.sqrt(np.mean(coeffs**2)))

    B_time, knots_time = create_bspline_basis(
        time_grid, config.n_interior_knots_time, degree=config.degree_time
    )
    B_freq, knots_freq = create_bspline_basis(
        freq_unit, config.n_interior_knots_freq, degree=config.degree_freq
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

    pls_init = initialize_with_penalized_least_squares(
        power, B_time, B_freq, P_time, P_freq, config
    )
    init_sites = whitened_init_values(pls_init, whitened, config)
    init_sites["beta"] = np.zeros(templates.shape[0])

    kernel = NUTS(
        _joint_model,
        init_strategy=init_to_value(values=init_sites),
        max_tree_depth=max_tree_depth, target_accept_prob=target_accept_prob,
    )
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=num_chains, chain_method="sequential", progress_bar=False)
    mcmc.run(
        random.PRNGKey(random_seed),
        jnp.asarray(coeffs), jnp.asarray(templates),
        jnp.asarray(basis_eig_time), jnp.asarray(basis_eig_freq),
        jnp.asarray(whitened["lam_time"]), jnp.asarray(whitened["lam_freq"]),
        jnp.asarray(whitened["joint_null"]), float(amp_scale), config,
        store_log_psd_samples,
        extra_fields=("diverging",),
    )
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    eig_samples = reconstruct_eig_coeff_samples(samples, whitened, config)
    log_mean, log_lower, log_upper = surface_summaries(
        eig_samples, basis_eig_time, basis_eig_freq,
        precomputed=samples.get("log_psd"),
    )
    return {
        "mcmc": mcmc,
        "samples": samples,
        "time_grid": np.asarray(time_grid),
        "freq_grid": np.asarray(freq_grid),
        "beta_mean": samples["beta"].mean(axis=0),
        "beta_std": samples["beta"].std(axis=0),
        "psd_mean": np.exp(log_mean),
        "psd_lower": np.exp(log_lower),
        "psd_upper": np.exp(log_upper),
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
    }


def _signal_amplitude_model(coeffs, templates, log_psd, amp_scale):
    """Signal block: linear template amplitudes with the noise surface fixed.

    Conditional on the current noise surface ``log_psd`` (held constant), the
    coefficients are ``c ~ N(sum_k beta_k g_k, S)`` with ``S = exp(log_psd)``. The
    noise-dependent normalisation ``log S`` is constant in this block and dropped.
    """
    with numpyro.plate("templates", templates.shape[0]):
        beta = numpyro.sample("beta", dist.Normal(0.0, amp_scale))
    signal = jnp.tensordot(beta, templates, axes=1)
    resid = coeffs - signal
    numpyro.factor("signal_like", -0.5 * jnp.sum(resid**2 * jnp.exp(-log_psd)))


def run_gibbs_signal_noise_mcmc(
    coeffs: np.ndarray,
    templates: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    *,
    config: PSplineConfig,
    amp_scale: float | None = None,
    n_sweeps: int = 80,
    n_burn_sweeps: int = 30,
    block_warmup: int = 40,
    block_samples: int = 8,
    random_seed: int = 7,
    max_tree_depth: int = 10,
    target_accept_prob: float = 0.9,
) -> dict[str, object]:
    """Blocked Gibbs fit alternating a NUTS update for each of two blocks.

    Each Gibbs sweep performs (i) a NUTS update of the whitened P-spline *noise*
    block conditional on the current signal estimate (fitting the residual
    coefficients ``c - h(beta)``), then (ii) a NUTS update of the *signal*
    amplitude block conditional on the current noise surface. Blocking lets the
    smooth, high-dimensional noise geometry and the low-dimensional amplitude
    geometry adapt their NUTS step size and mass matrix independently; the Gibbs
    sweep couples them through the conditioning. This is the
    Metropolis-within-Gibbs analogue of :func:`run_joint_signal_noise_mcmc`,
    which instead samples both blocks with a single joint trajectory.

    Each block re-adapts at every sweep (adaptive Gibbs), initialised from the
    previous sweep, so ``block_warmup`` only needs to refine an already-good
    state. Samples from the first ``n_burn_sweeps`` are discarded.

    TODO: the signal block currently samples the linear template amplitudes
    ``beta`` at fixed binary frequency. Swap ``_signal_amplitude_model`` for a
    nonlinear GB model to also sample ``theta = (f0, fdot, ...)``; the Gibbs
    structure is unchanged.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    templates = np.asarray(templates, dtype=float)
    if amp_scale is None:
        amp_scale = float(np.sqrt(np.mean(coeffs**2)))

    freq_unit = freq_grid / np.maximum(freq_grid[-1], 1e-12)
    B_time, knots_time = create_bspline_basis(
        time_grid, config.n_interior_knots_time, degree=config.degree_time)
    B_freq, knots_freq = create_bspline_basis(
        freq_unit, config.n_interior_knots_freq, degree=config.degree_freq)
    P_time = create_bspline_roughness_penalty(
        knots_time, degree=config.degree_time, derivative_order=config.diff_order_time)
    P_freq = create_bspline_roughness_penalty(
        knots_freq, degree=config.degree_freq, derivative_order=config.diff_order_freq)
    whitened = whiten_penalty_pair(P_time, P_freq)
    basis_eig_time = jnp.asarray(B_time @ whitened["U_time"])
    basis_eig_freq = jnp.asarray(B_freq @ whitened["U_freq"])
    lam_time = jnp.asarray(whitened["lam_time"])
    lam_freq = jnp.asarray(whitened["lam_freq"])
    joint_null = jnp.asarray(whitened["joint_null"])

    # Warm starts: penalized-LS surface from the raw power, linear amplitudes from
    # least squares of the templates against the data.
    pls_init = initialize_with_penalized_least_squares(
        coeffs**2, B_time, B_freq, P_time, P_freq, config)
    noise_init = whitened_init_values(pls_init, whitened, config)
    template_mat = templates.reshape(templates.shape[0], -1).T
    beta_init, *_ = np.linalg.lstsq(template_mat, coeffs.reshape(-1), rcond=None)
    beta_state = np.asarray(beta_init, dtype=float)

    noise_kernel = NUTS(
        pspline_surface_model, max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob)
    signal_kernel = NUTS(
        _signal_amplitude_model, max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob)
    noise_mcmc = MCMC(noise_kernel, num_warmup=block_warmup, num_samples=block_samples,
                      progress_bar=False)
    signal_mcmc = MCMC(signal_kernel, num_warmup=block_warmup, num_samples=block_samples,
                       progress_bar=False)

    coeffs_j = jnp.asarray(coeffs)
    templates_j = jnp.asarray(templates)
    key = random.PRNGKey(random_seed)

    log_psd_state = jnp.asarray(pls_init["log_psd"])
    beta_samples: list[np.ndarray] = []
    eig_samples: list[np.ndarray] = []
    divergences = 0

    for sweep in range(n_sweeps):
        # --- Noise block: fit the residual coefficients c - h(beta) ---
        signal = jnp.tensordot(jnp.asarray(beta_state), templates_j, axes=1)
        resid = (coeffs_j - signal)[None, :, :]  # (R=1, n_time, n_freq)
        key, k_noise = random.split(key)
        noise_mcmc.run(
            k_noise, resid, basis_eig_time, basis_eig_freq,
            lam_time, lam_freq, joint_null, config, True,
            init_params=noise_init, extra_fields=("diverging",))
        nsamp = {k: np.asarray(v) for k, v in noise_mcmc.get_samples().items()}
        noise_init = {"s": nsamp["s"][-1], "phi_time": nsamp["phi_time"][-1],
                      "phi_freq": nsamp["phi_freq"][-1]}
        log_psd_state = jnp.asarray(nsamp["log_psd"][-1])
        divergences += int(np.asarray(noise_mcmc.get_extra_fields()["diverging"]).sum())

        # --- Signal block: amplitudes with the noise surface fixed ---
        key, k_sig = random.split(key)
        signal_mcmc.run(
            k_sig, coeffs_j, templates_j, log_psd_state, float(amp_scale),
            init_params={"beta": jnp.asarray(beta_state)},
            extra_fields=("diverging",))
        ssamp = {k: np.asarray(v) for k, v in signal_mcmc.get_samples().items()}
        beta_state = ssamp["beta"][-1]
        divergences += int(np.asarray(signal_mcmc.get_extra_fields()["diverging"]).sum())

        if sweep >= n_burn_sweeps:
            beta_samples.append(ssamp["beta"])
            eig_samples.append(
                reconstruct_eig_coeff_samples(nsamp, whitened, config))

    beta_draws = np.concatenate(beta_samples, axis=0)
    eig_draws = np.concatenate(eig_samples, axis=0)
    log_mean, log_lower, log_upper = surface_summaries(
        eig_draws, np.asarray(basis_eig_time), np.asarray(basis_eig_freq))
    return {
        "time_grid": np.asarray(time_grid),
        "freq_grid": np.asarray(freq_grid),
        "beta_mean": beta_draws.mean(axis=0),
        "beta_std": beta_draws.std(axis=0),
        "beta_samples": beta_draws,
        "psd_mean": np.exp(log_mean),
        "psd_lower": np.exp(log_lower),
        "psd_upper": np.exp(log_upper),
        "n_sweeps": n_sweeps,
        "n_post_sweeps": n_sweeps - n_burn_sweeps,
        "divergences": divergences,
    }


def _multichannel_joint_model(coeffs, templates, basis_eig_time, basis_eig_freq,
                              lam_time, lam_freq, joint_null, amp_scale, config):
    """Per-channel noise surfaces with one shared signal amplitude vector.

    ``coeffs``    -- (C, n_time, n_freq) real coefficients per channel.
    ``templates`` -- (C, K, n_time, n_freq) per-channel template coefficients.
    The amplitudes ``beta`` (length K) are *shared* across channels (one source).
    """
    n_channels = coeffs.shape[0]
    n_t, n_f = basis_eig_time.shape[1], basis_eig_freq.shape[1]

    with numpyro.plate("templates", templates.shape[1]):
        beta = numpyro.sample("beta", dist.Normal(0.0, amp_scale))

    total = 0.0
    for c in range(n_channels):
        phi_time = _sample_log_gamma(f"phi_time_{c}", config.alpha_phi,
                                     config.beta_phi, config.phi_log_base_scale)
        phi_freq = _sample_log_gamma(f"phi_freq_{c}", config.alpha_phi,
                                     config.beta_phi, config.phi_log_base_scale)
        d = phi_time * lam_time[:, None] + phi_freq * lam_freq[None, :]
        scale = jnp.where(joint_null, 1.0 / jnp.sqrt(config.null_precision),
                          1.0 / jnp.sqrt(d + config.ridge_eps))
        with numpyro.plate(f"eig_{c}", n_t * n_f):
            s_flat = numpyro.sample(f"s_{c}", dist.Normal(0.0, 1.0))
        log_psd = basis_eig_time @ (s_flat.reshape((n_t, n_f)) * scale) @ basis_eig_freq.T
        signal = jnp.tensordot(beta, templates[c], axes=1)
        resid = coeffs[c] - signal
        total = total - 0.5 * jnp.sum(log_psd + resid**2 * jnp.exp(-log_psd))
        numpyro.deterministic(f"log_psd_{c}", log_psd)
    numpyro.factor("multichannel_joint", total)


def run_multichannel_joint_mcmc(
    coeffs: np.ndarray,
    templates: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    *,
    config: PSplineConfig,
    amp_scale: float | None = None,
    n_warmup: int = 400,
    n_samples: int = 400,
    num_chains: int = 1,
    random_seed: int = 7,
    max_tree_depth: int = 10,
    target_accept_prob: float = 0.9,
) -> dict[str, object]:
    """Joint A/E/T-style fit: per-channel noise PSD + one shared signal amplitude.

    Args:
        coeffs: ``(C, n_time, n_freq)`` real coefficients per channel.
        templates: ``(C, K, n_time, n_freq)`` per-channel template coefficients.
        time_grid, freq_grid: shared analysis grid.
        config: estimator configuration.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    templates = np.asarray(templates, dtype=float)
    n_channels = coeffs.shape[0]
    freq_unit = freq_grid / np.maximum(freq_grid[-1], 1e-12)
    if amp_scale is None:
        amp_scale = float(np.sqrt(np.mean(coeffs**2)))

    B_time, knots_time = create_bspline_basis(
        time_grid, config.n_interior_knots_time, degree=config.degree_time)
    B_freq, knots_freq = create_bspline_basis(
        freq_unit, config.n_interior_knots_freq, degree=config.degree_freq)
    P_time = create_bspline_roughness_penalty(
        knots_time, degree=config.degree_time, derivative_order=config.diff_order_time)
    P_freq = create_bspline_roughness_penalty(
        knots_freq, degree=config.degree_freq, derivative_order=config.diff_order_freq)
    whitened = whiten_penalty_pair(P_time, P_freq)
    basis_eig_time = B_time @ whitened["U_time"]
    basis_eig_freq = B_freq @ whitened["U_freq"]

    init_sites = {"beta": np.zeros(templates.shape[1])}
    for c in range(n_channels):
        pls = initialize_with_penalized_least_squares(
            coeffs[c] ** 2, B_time, B_freq, P_time, P_freq, config)
        ch = whitened_init_values(pls, whitened, config)
        init_sites[f"s_{c}"] = ch["s"]
        init_sites[f"phi_time_{c}"] = ch["phi_time"]
        init_sites[f"phi_freq_{c}"] = ch["phi_freq"]

    kernel = NUTS(_multichannel_joint_model,
                  init_strategy=init_to_value(values=init_sites),
                  max_tree_depth=max_tree_depth, target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=num_chains, chain_method="sequential", progress_bar=False)
    mcmc.run(
        random.PRNGKey(random_seed),
        jnp.asarray(coeffs), jnp.asarray(templates),
        jnp.asarray(basis_eig_time), jnp.asarray(basis_eig_freq),
        jnp.asarray(whitened["lam_time"]), jnp.asarray(whitened["lam_freq"]),
        jnp.asarray(whitened["joint_null"]), float(amp_scale), config,
        extra_fields=("diverging",))

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    psd_mean, psd_lower, psd_upper = [], [], []
    for c in range(n_channels):
        lp = samples[f"log_psd_{c}"]
        psd_mean.append(np.exp(np.mean(lp, axis=0)))
        psd_lower.append(np.exp(np.percentile(lp, 5.0, axis=0)))
        psd_upper.append(np.exp(np.percentile(lp, 95.0, axis=0)))
    return {
        "mcmc": mcmc,
        "samples": samples,
        "time_grid": np.asarray(time_grid),
        "freq_grid": np.asarray(freq_grid),
        "beta_mean": samples["beta"].mean(axis=0),
        "beta_std": samples["beta"].std(axis=0),
        "psd_mean": np.stack(psd_mean),
        "psd_lower": np.stack(psd_lower),
        "psd_upper": np.stack(psd_upper),
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
    }
