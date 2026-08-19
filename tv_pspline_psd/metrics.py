"""Error metrics for time-varying PSD estimates."""

from __future__ import annotations

import numpy as np


def relative_surface_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Relative Frobenius error ``||ref - est|| / ||ref||``."""
    reference = np.asarray(reference)
    estimate = np.asarray(estimate)
    return float(np.linalg.norm(reference - estimate) / np.linalg.norm(reference))


def mse_log_psd(
    true_psd: np.ndarray,
    estimate_psd: np.ndarray,
) -> float:
    """Mean squared error of the log-PSD over the time-frequency grid.

    Implements the metric used in the manuscript,

        MSE_logf = (1 / (T * (K + 1))) * sum_{t, j} (ln fhat - ln f0)^2,

    i.e. the average squared difference of ``log`` PSD across all grid points.
    Both inputs must be on the same grid.
    """
    return float(np.mean(_log_ratio(true_psd, estimate_psd) ** 2))


def _log_ratio(true_psd: np.ndarray, estimate_psd: np.ndarray) -> np.ndarray:
    """``log(estimate / truth)`` on a shared grid, with the shared validation."""
    true_psd = np.asarray(true_psd)
    estimate_psd = np.asarray(estimate_psd)
    if true_psd.shape != estimate_psd.shape:
        raise ValueError("true_psd and estimate_psd must share a grid/shape.")
    if not np.isfinite(true_psd).all() or not np.isfinite(estimate_psd).all():
        raise ValueError("PSD inputs must contain only finite values.")
    if np.any(true_psd <= 0) or np.any(estimate_psd <= 0):
        raise ValueError("PSD inputs must be strictly positive before taking logs.")
    return np.log(estimate_psd) - np.log(true_psd)


def bias_log_psd(
    true_psd: np.ndarray,
    estimate_psd: np.ndarray,
) -> float:
    """Signed mean log error of the PSD estimate over the grid.

    Implements ``Bias_logS`` of the manuscript,

        Bias_logS = (1 / Q) * sum_q (ln Shat_q - ln S0_q),

    in nats. Paired with :func:`rmse_log_psd` it separates a systematic offset
    from scatter: a small RMSE with a comparable bias is a shifted estimate,
    not a noisy one.
    """
    diff = _log_ratio(true_psd, estimate_psd)
    return float(np.mean(diff))


def rmse_log_psd(
    true_psd: np.ndarray,
    estimate_psd: np.ndarray,
) -> float:
    """Root-mean-square log error, ``RMSE_logS`` of the manuscript, in nats.

    This is the reported quantity; :func:`mse_log_psd` returns its square.
    """
    return float(np.sqrt(mse_log_psd(true_psd, estimate_psd)))


def interval_coverage(
    true_psd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of grid points where ``true_psd`` lies within ``[lower, upper]``."""
    true_psd = np.asarray(true_psd)
    return float(np.mean((true_psd >= np.asarray(lower)) & (true_psd <= np.asarray(upper))))
