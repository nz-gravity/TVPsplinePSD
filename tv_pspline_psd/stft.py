"""Short-time Fourier (moving-periodogram) front end for the P-spline estimator.

Splits the series into segments and forms windowed STFT coefficients. The real
and imaginary parts of each coefficient are the two real Gaussian observations
``Re, Im ~ N(0, S)`` fed to the shared estimator (:func:`fit_log_pspline_surface`)
-- the same likelihood used for WDM, which has one real coefficient per cell. The
summed power ``Re^2 + Im^2`` is the moving periodogram, so this is the
dynamic-Whittle method expressed on coefficients (and is signal-ready, unlike a
power-only periodogram).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from .config import PSplineConfig
from .inference import fit_log_pspline_surface


def moving_stft(
    data: np.ndarray,
    dt: float,
    *,
    nperseg: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windowed, non-overlapping STFT coefficients as real components.

    Returns:
        ``(time_grid, freq_grid, coeffs)`` where ``time_grid`` are the rescaled
        segment centres in ``[0, 1]``, ``freq_grid`` is in Hz, and ``coeffs`` has
        shape ``(2, n_segments, n_freq)`` holding the real and imaginary parts.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("STFT input data must be one-dimensional.")
    if dt <= 0:
        raise ValueError("dt must be strictly positive.")
    if not isinstance(nperseg, (int, np.integer)) or nperseg <= 1:
        raise ValueError("nperseg must be an integer greater than one.")
    n_seg = len(data) // nperseg
    if n_seg < 2:
        raise ValueError("Need at least two segments.")
    segments = data[: n_seg * nperseg].reshape(n_seg, nperseg)
    w = get_window(window, nperseg)
    spectrum = np.fft.rfft(segments * w[None, :], axis=1)
    coeffs = np.stack([spectrum.real, spectrum.imag], axis=0)
    freq_grid = np.fft.rfftfreq(nperseg, d=dt)
    time_grid = (np.arange(n_seg) + 0.5) / n_seg
    return time_grid, freq_grid, coeffs


def stft_white_noise_calibration(
    n_total: int, dt: float, nperseg: int, *, n_draws: int = 400, seed: int = 0,
    window: str = "hann",
) -> np.ndarray:
    """Per-channel fitted-PSD level of unit-variance white noise (per component).

    The estimator fits ``S = mean over components of c^2``; this returns that
    level for white noise, used to rescale estimates to the analytic-PSD scale.
    """
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(n_draws):
        _, _, coeffs = moving_stft(rng.standard_normal(n_total), dt,
                                   nperseg=nperseg, window=window)
        level = np.mean(coeffs**2, axis=0).mean(axis=0)  # mean over comps and segments
        acc = level if acc is None else acc + level
    return acc / n_draws


def run_stft_mcmc(
    data: np.ndarray,
    *,
    dt: float,
    nperseg: int,
    config: PSplineConfig,
    window: str = "hann",
    **fit_kwargs,
) -> dict[str, object]:
    """STFT front end: form moving-STFT coefficients, then fit the surface."""
    time_grid, freq_grid, coeffs = moving_stft(data, dt, nperseg=nperseg, window=window)
    keep_freq = np.arange(
        config.trim_low_freq_channels, coeffs.shape[2] - config.trim_high_freq_channels
    )
    if keep_freq.size == 0:
        raise ValueError("STFT trimming leaves an empty frequency grid.")
    coeffs = coeffs[:, :, keep_freq]
    freq_grid = freq_grid[keep_freq]
    zero_imag = [
        int(keep_freq[j])
        for j in range(coeffs.shape[2])
        if np.all(coeffs[1, :, j] == 0)
    ]
    if zero_imag:
        raise ValueError(
            "STFT DC/Nyquist channels have an identically zero imaginary component "
            f"and must be trimmed; untrimmed channel indices: {zero_imag}."
        )
    results = fit_log_pspline_surface(
        coeffs, time_grid, freq_grid, config=config, **fit_kwargs
    )
    results.update({"nperseg": nperseg, "keep_freq": keep_freq})
    results["provenance"].update({
        "dt": float(dt),
        "nperseg": int(nperseg),
        "trims": {
            "low_freq_channels": config.trim_low_freq_channels,
            "high_freq_channels": config.trim_high_freq_channels,
        },
        "source_data": {"shape": list(np.asarray(data).shape)},
    })
    return results
