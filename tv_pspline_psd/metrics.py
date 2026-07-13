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
    true_psd = np.asarray(true_psd)
    estimate_psd = np.asarray(estimate_psd)
    if true_psd.shape != estimate_psd.shape:
        raise ValueError("true_psd and estimate_psd must share a grid/shape.")
    if not np.isfinite(true_psd).all() or not np.isfinite(estimate_psd).all():
        raise ValueError("PSD inputs must contain only finite values.")
    if np.any(true_psd <= 0) or np.any(estimate_psd <= 0):
        raise ValueError("PSD inputs must be strictly positive before taking logs.")
    diff = np.log(estimate_psd) - np.log(true_psd)
    return float(np.mean(diff**2))


def interval_coverage(
    true_psd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of grid points where ``true_psd`` lies within ``[lower, upper]``."""
    true_psd = np.asarray(true_psd)
    return float(np.mean((true_psd >= np.asarray(lower)) & (true_psd <= np.asarray(upper))))
