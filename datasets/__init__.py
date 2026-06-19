"""Data-generating processes for the WDM PSD studies.

These modules are intentionally decoupled from the estimator: each simulator
returns a raw time series (and an analytic true PSD), with no dependence on the
spline machinery in :mod:`wdm_psd`.
"""

from __future__ import annotations

from ._wdm import (
    monte_carlo_reference,
    trimmed_keep_indices,
    wdm_white_noise_calibration,
)
from .lisa import (
    LISANoiseConfig,
    galactic_modulation,
    lisa_galactic_confusion_psd,
    lisa_instrument_psd,
    normalization_constant,
    simulate_tv_lisa_noise,
    true_psd_lisa,
)
from .galactic_binary import gb_quadratures, gb_signal
from .lisa_tdi import (
    gb_tdi_signal,
    lisa_tdi_confusion_psd,
    lisa_tdi_noise_psd,
    optimal_snr,
    simulate_tv_lisa_tdi,
    true_tv_lisa_tdi_psd,
)
from .ls2 import simulate_ls2, true_psd_ls2

__all__ = [
    "simulate_ls2",
    "true_psd_ls2",
    "gb_signal",
    "gb_quadratures",
    "gb_tdi_signal",
    "simulate_tv_lisa_tdi",
    "true_tv_lisa_tdi_psd",
    "lisa_tdi_noise_psd",
    "lisa_tdi_confusion_psd",
    "optimal_snr",
    "LISANoiseConfig",
    "simulate_tv_lisa_noise",
    "true_psd_lisa",
    "lisa_instrument_psd",
    "lisa_galactic_confusion_psd",
    "galactic_modulation",
    "normalization_constant",
    "monte_carlo_reference",
    "trimmed_keep_indices",
    "wdm_white_noise_calibration",
]
