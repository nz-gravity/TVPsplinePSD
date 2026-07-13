"""A simple differentiable galactic-binary (GB) signal for the joint-fit demo.

In a single data channel a galactic binary is a nearly monochromatic, slowly
chirping line,

    h(t) = A cos(2 pi (f0 t + 1/2 fdot t^2) + phi0)
         = a c(t) + b s(t),

with quadratures ``c(t) = cos(psi(t))``, ``s(t) = sin(psi(t))`` and
``a = A cos phi0``, ``b = -A sin phi0``. Because ``h`` is linear in ``(a, b)``
and the WDM transform is linear, the signal's WDM coefficients are
``a g_c + b g_s`` for the (precomputed) template coefficients ``g_c, g_s`` -- a
well-conditioned signal model for joint inference with the noise PSD.

This is the single-channel morphology; the realistic LISA TDI response (e.g.
``jaxgb``) is a drop-in replacement for the injected waveform.
"""

from __future__ import annotations

import numpy as np


def gb_quadratures(n: int, dt: float, *, f0: float, fdot: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the cosine/sine quadrature time series ``c(t), s(t)``."""
    t = np.arange(n, dtype=float) * dt
    psi = 2.0 * np.pi * (f0 * t + 0.5 * fdot * t**2)
    return np.cos(psi), np.sin(psi)


def gb_signal(
    n: int, dt: float, *, f0: float, fdot: float = 0.0, amp: float, phi0: float = 0.0
) -> tuple[np.ndarray, tuple[float, float]]:
    """Galactic-binary time series and its true quadrature amplitudes ``(a, b)``."""
    c, s = gb_quadratures(n, dt, f0=f0, fdot=fdot)
    a = amp * np.cos(phi0)
    b = -amp * np.sin(phi0)
    return a * c + b * s, (a, b)
