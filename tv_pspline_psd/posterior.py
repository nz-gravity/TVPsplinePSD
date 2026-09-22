"""Memory-bounded posterior reconstruction for tensor and nested PSD surfaces."""

from __future__ import annotations

import numpy as np

from .config import PSplineConfig


def reconstruct_eig_coeff_samples(
    samples: dict[str, np.ndarray],
    whitened: dict[str, np.ndarray],
    config: PSplineConfig,
) -> np.ndarray:
    """Per-sample eigen-coefficients ``Z`` of shape ``(n_samples, K_t, K_f)``.

    These are tiny (``K_t K_f`` numbers per sample) and fully determine the
    surface, so summaries can be reconstructed without storing it per sample.
    """
    lam_t = whitened["lam_time"]
    lam_f = whitened["lam_freq"]
    joint_null = whitened["joint_null"]
    n_t, n_f = lam_t.size, lam_f.size

    s = samples["s"].reshape(-1, n_t, n_f)
    if config.centered:
        return s
    phi_time = np.exp(samples["phi_time"])[:, None, None]  # the site stores log phi
    phi_freq = np.exp(samples["phi_freq"])[:, None, None]
    d = phi_time * lam_t[None, :, None] + phi_freq * lam_f[None, None, :]
    scale = np.where(
        joint_null[None],
        1.0 / np.sqrt(config.null_precision),
        1.0 / np.sqrt(d + config.ridge_eps),
    )
    return s * scale


def _summary_frequency_chunk(n_draws: int, n_time: int, requested: int) -> int:
    """Bound a float64 draw surface to 128 MiB before percentile workspace."""
    return max(1, min(requested, (128 * 1024**2) // max(1, n_draws * n_time * 8)))


def surface_summaries(
    eig_samples: np.ndarray,
    basis_eig_time: np.ndarray,
    basis_eig_freq: np.ndarray,
    *,
    precomputed: np.ndarray | None = None,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
    freq_chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean and central interval of ``log S`` on the analysis grid.

    The mean is reconstructed from the mean eigen-coefficients (exact, since the
    surface is linear in them); the interval is reconstructed in frequency chunks
    to bound peak memory. If ``precomputed`` (the stored per-sample surface) is
    given, it is used directly.
    """
    log_mean = basis_eig_time @ eig_samples.mean(axis=0) @ basis_eig_freq.T
    if precomputed is not None:
        return (
            log_mean,
            np.percentile(precomputed, lower_pct, axis=0),
            np.percentile(precomputed, upper_pct, axis=0),
        )

    n_t = basis_eig_time.shape[0]
    n_f = basis_eig_freq.shape[0]
    freq_chunk = _summary_frequency_chunk(len(eig_samples), n_t, freq_chunk)
    lower = np.empty((n_t, n_f))
    upper = np.empty((n_t, n_f))
    for j0 in range(0, n_f, freq_chunk):
        bf = basis_eig_freq[j0 : j0 + freq_chunk]
        # optimize=True factorises the 3-operand contraction into two BLAS
        # matmuls; without it numpy falls back to a naive element-wise kernel
        # that scales catastrophically on large (time x freq) grids.
        chunk = np.einsum(
            "ta,nab,jb->ntj", basis_eig_time, eig_samples, bf, optimize=True
        )
        lower[:, j0 : j0 + freq_chunk], upper[:, j0 : j0 + freq_chunk] = np.percentile(
            chunk, [lower_pct, upper_pct], axis=0
        )
    return log_mean, lower, upper


def nested_surface_summaries(
    g_samples: np.ndarray,
    h_samples: np.ndarray,
    sigma_samples: np.ndarray,
    basis_interaction_time: np.ndarray,
    basis_eig_freq: np.ndarray,
    *,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
    freq_chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize ``g(f) + sigma*h(t,f)`` without storing full draw surfaces."""
    g_samples = np.asarray(g_samples)
    h_samples = np.asarray(h_samples)
    sigma_samples = np.asarray(sigma_samples).reshape(-1)
    stationary_mean = g_samples.mean(axis=0) @ basis_eig_freq.T
    # h_samples are already drawn in the centered hierarchy with prior scale
    # sigma_interaction; do not multiply by sigma a second time.
    interaction_coefficients = h_samples
    interaction_mean = (
        basis_interaction_time
        @ interaction_coefficients.mean(axis=0)
        @ basis_eig_freq.T
    )
    log_mean = stationary_mean[None, :] + interaction_mean
    n_t = basis_interaction_time.shape[0]
    n_f = basis_eig_freq.shape[0]
    freq_chunk = _summary_frequency_chunk(len(g_samples), n_t, freq_chunk)
    lower = np.empty((n_t, n_f))
    upper = np.empty((n_t, n_f))
    for j0 in range(0, n_f, freq_chunk):
        bf = basis_eig_freq[j0 : j0 + freq_chunk]
        stationary_chunk = g_samples @ bf.T
        interaction_chunk = np.einsum(
            "ta,nab,jb->ntj",
            basis_interaction_time,
            interaction_coefficients,
            bf,
            optimize=True,
        )
        chunk = stationary_chunk[:, None, :] + interaction_chunk
        lower[:, j0 : j0 + freq_chunk], upper[:, j0 : j0 + freq_chunk] = np.percentile(
            chunk, [lower_pct, upper_pct], axis=0
        )
    return log_mean, lower, upper


def summarize_surface_samples(
    samples: dict[str, np.ndarray],
    whitened: dict[str, np.ndarray],
    config: PSplineConfig,
    basis_eig_time: np.ndarray,
    basis_eig_freq: np.ndarray,
    log_psd_offset: np.ndarray,
    *,
    nested: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean coefficients, log-PSD mean/limits and the final conditional draw."""
    if nested:
        n_interaction_time = whitened["lam_time"].size
        n_nested_freq = whitened["lam_freq"].size
        g_samples = samples["g"].reshape(-1, n_nested_freq)
        h_samples = samples["h"].reshape(-1, n_interaction_time, n_nested_freq)
        sigma_samples = samples["sigma_interaction"].reshape(-1)
        log_mean_residual, log_lower_residual, log_upper_residual = (
            nested_surface_summaries(
                g_samples,
                h_samples,
                sigma_samples,
                basis_eig_time,
                basis_eig_freq,
            )
        )
        interaction_eig_last = h_samples[-1]
        log_last_residual = (basis_eig_freq @ g_samples[-1])[None, :] + (
            basis_eig_time @ interaction_eig_last @ basis_eig_freq.T
        )
        eig_samples = h_samples
        W_mean = whitened["U_time"] @ h_samples.mean(axis=0) @ whitened["U_freq"].T
    else:
        eig_samples = reconstruct_eig_coeff_samples(samples, whitened, config)
        W_mean = whitened["U_time"] @ eig_samples.mean(axis=0) @ whitened["U_freq"].T
        log_mean_residual, log_lower_residual, log_upper_residual = surface_summaries(
            eig_samples,
            basis_eig_time,
            basis_eig_freq,
            precomputed=samples.get("log_psd"),
        )
        log_last_residual = basis_eig_time @ eig_samples[-1] @ basis_eig_freq.T
    log_mean = log_psd_offset + log_mean_residual
    log_lower = log_psd_offset + log_lower_residual
    log_upper = log_psd_offset + log_upper_residual
    # The final retained draw is the exact conditional draw a blocked
    # signal/noise sampler must pass to its signal block (never the mean).
    log_last = log_psd_offset + log_last_residual

    return W_mean, log_mean, log_lower, log_upper, log_last
