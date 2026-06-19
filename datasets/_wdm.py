"""Shared helpers for building Monte Carlo WDM references from a simulator."""

from __future__ import annotations

from typing import Callable

import numpy as np
from wdm_transform import TimeSeries

from tv_pspline_psd import PSplineConfig


def trimmed_keep_indices(
    n_total: int, dt: float, nt: int, config: PSplineConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Time/frequency index ranges kept after WDM-edge trimming."""
    probe = TimeSeries(np.zeros(n_total), dt=dt).to_wdm(nt=nt)
    keep_time = np.arange(config.trim_time_bins, probe.nt - config.trim_time_bins)
    keep_freq = np.arange(
        config.trim_low_freq_channels,
        probe.nf + 1 - config.trim_high_freq_channels,
    )
    return keep_time, keep_freq


def wdm_white_noise_calibration(
    n_total: int,
    dt: float,
    nt: int,
    config: PSplineConfig,
    *,
    n_draws: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """Per-channel WDM power of unit-variance white noise.

    The WDM transform preserves a digital PSD only up to a (near-constant)
    normalization ``C_m`` per channel: ``E[w_nm^2] = C_m * S_dig(f_m)`` to first
    order. Multiplying an analytic digital-convention PSD by ``C_m`` expresses it
    in the same WDM-coefficient units the estimator infers. ``C_m`` is estimated
    here by passing unit white noise (``S_dig = 1``) through the transform.

    Returns:
        Array ``C_m`` over the trimmed frequency channels, shape ``(n_freq,)``.
    """
    levels = monte_carlo_reference(
        lambda rng: rng.standard_normal(n_total),
        n_draws=n_draws,
        n_total=n_total,
        dt=dt,
        nt=nt,
        config=config,
        seed=seed,
    )
    return levels.mean(axis=0)


def monte_carlo_reference(
    simulate: Callable[[np.random.Generator], np.ndarray],
    *,
    n_draws: int,
    n_total: int,
    dt: float,
    nt: int,
    config: PSplineConfig,
    seed: int,
) -> np.ndarray:
    """Empirical ``E[w^2]`` on the trimmed WDM grid from repeated simulation.

    Args:
        simulate: Callable mapping an RNG to one time-series realization.
        n_draws: Number of independent realizations to average.
        n_total: Samples per realization.
        dt: Sampling interval.
        nt: Number of WDM time bins.
        config: Estimator config (for matching the trim).
        seed: PRNG seed.
    """
    keep_time, keep_freq = trimmed_keep_indices(n_total, dt, nt, config)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        sample = simulate(rng)
        coeffs = np.asarray(TimeSeries(sample, dt=dt).to_wdm(nt=nt).coeffs)
        if coeffs.ndim == 3:
            coeffs = coeffs[0]
        draws.append(coeffs[np.ix_(keep_time, keep_freq)] ** 2)
    return np.mean(draws, axis=0)
