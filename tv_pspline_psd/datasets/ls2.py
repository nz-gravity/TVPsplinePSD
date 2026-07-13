"""LS2 locally stationary time-varying MA(1) process (Tang et al. 2026).

    X_{t,T} = w_t + 1.1 cos(1.5 - cos(4 pi t / T)) w_{t-1},

with i.i.d. standard-normal innovations. The analytic pointwise PSD is returned
in the Oppenheim-Schafer digital convention, where unit-variance white noise has
PSD 1 -- the same scale as the expected squared WDM coefficient ``E[w_nm^2]``.
"""

from __future__ import annotations

import numpy as np


def simulate_ls2(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Simulate one realization of the LS2 process of length ``n``."""
    w = rng.normal(0.0, 1.0, n + 2)
    data = np.zeros(n)
    for t in range(n):
        u = t / n
        b1 = 1.1 * np.cos(1.5 - np.cos(4.0 * np.pi * u))
        data[t] = w[t + 1] + b1 * w[t]
    return data


def true_psd_ls2(
    time_grid: np.ndarray,
    freq_grid_hz: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Analytic pointwise time-varying PSD of LS2 on a WDM grid.

    Args:
        time_grid: Rescaled time coordinates in ``[0, 1]``.
        freq_grid_hz: WDM channel centre frequencies in Hz.
        dt: Sampling interval (maps Hz to digital angular frequency).

    Returns:
        Array of shape ``(len(time_grid), len(freq_grid_hz))``.
    """
    omega = 2.0 * np.pi * dt * np.asarray(freq_grid_hz)
    psd = np.zeros((len(time_grid), len(omega)))
    for i, u in enumerate(time_grid):
        b1 = 1.1 * np.cos(1.5 - np.cos(4.0 * np.pi * u))
        psd[i, :] = 1.0 + b1**2 + 2.0 * b1 * np.cos(omega)
    return psd
