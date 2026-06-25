"""WDM tensor-product log-P-spline estimator for non-stationary noise PSDs.

The estimator fits a smooth ``log S(t, f)`` surface to squared WDM coefficients
with a WDM Whittle likelihood (``w_nm ~ N(0, S_nm)``) and a non-centered
(whitened) anisotropic P-spline prior. See :mod:`tv_pspline_psd.model` for the prior
construction and :func:`tv_pspline_psd.run_wdm_psd_mcmc` for the entry point.
"""

from __future__ import annotations

import jax as _jax

# The WDM Whittle likelihood evaluates exp(-log_psd); float64 is required for
# numerical stability across the PSD dynamic range.
_jax.config.update("jax_enable_x64", True)

# Submodule imports must follow the x64 config above (E402 is expected here).
from .config import PSplineConfig  # noqa: E402
from .diagnostics import summarize_mcmc_diagnostics  # noqa: E402
from .inference import (  # noqa: E402
    evaluate_dense_posterior_mean,
    fit_log_pspline_surface,
    run_wdm_psd_mcmc,
    wdm_analysis_coefficients,
)
from .joint import (  # noqa: E402
    run_gibbs_signal_noise_mcmc,
    run_gibbs_stft_signal_noise_mcmc,
    run_joint_signal_noise_mcmc,
    run_joint_dL_wdm_mcmc,
    run_joint_sobh_wdm_mcmc,
    run_multichannel_joint_mcmc,
)
from .sobh_signal import (  # noqa: E402
    build_sobh_wdm_grid,
    make_sobh_wdm_signal_fn,
)
from .moving_periodogram import (  # noqa: E402
    run_tang_dynamic_whittle_mcmc,
    tang_moving_periodogram,
)
from .stationary import run_stationary_psd_mcmc  # noqa: E402
from .metrics import interval_coverage, mse_log_psd, relative_surface_error  # noqa: E402
from .plotting import (  # noqa: E402
    plot_channel_slice,
    plot_surface_comparison,
    save_figure,
    set_paper_style,
)
from .stft import moving_stft, run_stft_mcmc, stft_white_noise_calibration  # noqa: E402
from .io import (  # noqa: E402
    load_run,
    results_to_idata,
    save_run,
    surface_from_idata,
)

__all__ = [
    "PSplineConfig",
    "fit_log_pspline_surface",
    "wdm_analysis_coefficients",
    "run_wdm_psd_mcmc",
    "run_joint_signal_noise_mcmc",
    "run_joint_sobh_wdm_mcmc",
    "run_joint_dL_wdm_mcmc",
    "build_sobh_wdm_grid",
    "make_sobh_wdm_signal_fn",
    "run_gibbs_signal_noise_mcmc",
    "run_gibbs_stft_signal_noise_mcmc",
    "run_multichannel_joint_mcmc",
    "run_tang_dynamic_whittle_mcmc",
    "tang_moving_periodogram",
    "run_stationary_psd_mcmc",
    "run_stft_mcmc",
    "moving_stft",
    "stft_white_noise_calibration",
    "evaluate_dense_posterior_mean",
    "save_run",
    "load_run",
    "results_to_idata",
    "surface_from_idata",
    "summarize_mcmc_diagnostics",
    "relative_surface_error",
    "mse_log_psd",
    "interval_coverage",
    "plot_surface_comparison",
    "plot_channel_slice",
    "save_figure",
    "set_paper_style",
]
