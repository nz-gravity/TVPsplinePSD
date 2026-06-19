"""WDM tensor-product log-P-spline estimator for non-stationary noise PSDs.

The estimator fits a smooth ``log S(t, f)`` surface to squared WDM coefficients
with a WDM Whittle likelihood (``w_nm ~ N(0, S_nm)``) and a non-centered
(whitened) anisotropic P-spline prior. See :mod:`wdm_psd.model` for the prior
construction and :func:`wdm_psd.run_wdm_psd_mcmc` for the entry point.
"""

from __future__ import annotations

import jax as _jax

# The WDM Whittle likelihood evaluates exp(-log_psd); float64 is required for
# numerical stability across the PSD dynamic range.
_jax.config.update("jax_enable_x64", True)

from .config import PSplineConfig
from .diagnostics import summarize_mcmc_diagnostics
from .inference import (
    evaluate_dense_posterior_mean,
    fit_log_pspline_surface,
    run_wdm_psd_mcmc,
    wdm_analysis_coefficients,
)
from .joint import run_joint_signal_noise_mcmc, run_multichannel_joint_mcmc
from .moving_periodogram import run_tang_dynamic_whittle_mcmc, tang_moving_periodogram
from .metrics import interval_coverage, mse_log_psd, relative_surface_error
from .plotting import plot_channel_slice, plot_surface_comparison, save_figure
from .stft import moving_stft, run_stft_mcmc, stft_white_noise_calibration

__all__ = [
    "PSplineConfig",
    "fit_log_pspline_surface",
    "wdm_analysis_coefficients",
    "run_wdm_psd_mcmc",
    "run_joint_signal_noise_mcmc",
    "run_multichannel_joint_mcmc",
    "run_tang_dynamic_whittle_mcmc",
    "tang_moving_periodogram",
    "run_stft_mcmc",
    "moving_stft",
    "stft_white_noise_calibration",
    "evaluate_dense_posterior_mean",
    "summarize_mcmc_diagnostics",
    "relative_surface_error",
    "mse_log_psd",
    "interval_coverage",
    "plot_surface_comparison",
    "plot_channel_slice",
    "save_figure",
]
