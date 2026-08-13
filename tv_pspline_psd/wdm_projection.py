"""Project continuous PSD models onto WDM marginal-power cells.

For an interior WDM frequency channel centred at ``f_m``, the estimand is

``E[w_nm**2] = integral K(u) S(t_n, f_m + u * delta_f) du``

with ``K(u)`` proportional to the squared Meyer frequency window.  Evaluating
``S`` only at ``f_m`` is adequate for locally flat spectra, but it can fail at
an extremely deep transfer-function zero even though the WDM kernel is
compactly supported.  The helpers here expose a small deterministic quadrature
rule so response models and simulation truth can be projected onto exactly the
quantity fitted by the WDM likelihood.

The current analysis trims the DC and Nyquist edge channels, so this module
implements the common interior-channel kernel.  Edge-channel projection needs
its separate half-window normalization and is deliberately rejected.
"""

from __future__ import annotations

import numpy as np


def _meyer_phi_squared(u: np.ndarray, *, a: float) -> np.ndarray:
    """Squared cosine-tapered Meyer window on normalized frequency ``u``."""
    if not 0.0 < a < 0.5:
        raise ValueError("a must lie strictly between zero and one half")
    values = np.asarray(u, dtype=float)
    support = 1.0 - a
    taper_width = 1.0 - 2.0 * a
    absolute = np.abs(values)
    window = np.ones_like(absolute)
    taper = absolute > a
    window[taper] = np.cos(
        0.5 * np.pi * (absolute[taper] - a) / taper_width
    )
    window[absolute > support] = 0.0
    return window**2


def wdm_frequency_quadrature(
    n_nodes: int = 16,
    *,
    a: float = 1.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized offsets and weights for an interior WDM channel.

    Gauss--Legendre nodes cover the exact compact support
    ``|u| <= 1-a``.  The returned non-negative weights include the squared
    Meyer window and sum to one, so constants are preserved exactly up to
    floating-point roundoff.
    """
    if not isinstance(n_nodes, (int, np.integer)) or isinstance(
        n_nodes, (bool, np.bool_)
    ):
        raise TypeError("n_nodes must be an integer")
    if n_nodes < 4:
        raise ValueError("n_nodes must be at least four")
    if not 0.0 < a < 0.5:
        raise ValueError("a must lie strictly between zero and one half")

    unit_nodes, unit_weights = np.polynomial.legendre.leggauss(int(n_nodes))
    support = 1.0 - a
    offsets = support * unit_nodes
    weights = support * unit_weights * _meyer_phi_squared(offsets, a=a)
    weights /= weights.sum()
    return offsets, weights


def wdm_frequency_projection_grid(
    frequency_hz: np.ndarray,
    delta_f_hz: float,
    *,
    n_nodes: int = 16,
    a: float = 1.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return quadrature frequencies ``(n_frequency, n_nodes)`` and weights."""
    frequency = np.asarray(frequency_hz, dtype=float)
    if frequency.ndim != 1 or frequency.size == 0:
        raise ValueError("frequency_hz must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_hz must contain finite positive values")
    if np.any(np.diff(frequency) <= 0.0):
        raise ValueError("frequency_hz must be strictly increasing")
    if not np.isfinite(delta_f_hz) or delta_f_hz <= 0.0:
        raise ValueError("delta_f_hz must be finite and positive")

    offsets, weights = wdm_frequency_quadrature(n_nodes, a=a)
    grid = frequency[:, None] + delta_f_hz * offsets[None, :]
    if np.any(grid <= 0.0):
        raise ValueError(
            "WDM projection crosses zero frequency; trim the DC edge channel"
        )
    return grid, weights


def collapse_wdm_frequency_projection(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Collapse values sampled on the final quadrature axis.

    ``values`` may have any leading dimensions, but its final dimension must
    match ``weights``.  This split evaluation/collapse API lets callers process
    large time-frequency surfaces in bounded frequency chunks.
    """
    sampled = np.asarray(values, dtype=float)
    quadrature_weights = np.asarray(weights, dtype=float)
    if sampled.ndim < 1 or quadrature_weights.ndim != 1:
        raise ValueError("values and weights must have a quadrature dimension")
    if sampled.shape[-1] != quadrature_weights.size:
        raise ValueError("the final values dimension must match weights")
    if np.any(~np.isfinite(sampled)):
        raise ValueError("values must contain only finite entries")
    if np.any(~np.isfinite(quadrature_weights)) or np.any(quadrature_weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if not np.isclose(quadrature_weights.sum(), 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("weights must sum to one")
    return np.einsum("...q,q->...", sampled, quadrature_weights, optimize=True)


__all__ = [
    "collapse_wdm_frequency_projection",
    "wdm_frequency_projection_grid",
    "wdm_frequency_quadrature",
]
