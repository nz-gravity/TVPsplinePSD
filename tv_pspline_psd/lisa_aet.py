"""A/E/T helpers for the TDI XYZ archive.

``xyz_to_aet_series`` applies the orthonormal rotation to samples;
``xyz_covariance_to_aet_diagonal`` rotates a full XYZ covariance and returns
the A/E/T auto-PSDs.

There is deliberately no helper that rotates XYZ *auto-PSDs* alone. Doing so
assumes zero XYZ cross spectra, which no physical TDI response satisfies: the
cross terms are what cancel in T. Rotating the diagonal put the Galactic
foreground into T at ~1/3 of A instead of ~1e-4 of it. Rotate the covariance.
"""

from __future__ import annotations

import numpy as np


AET_CHANNELS = ("A", "E", "T")
XYZ_TO_AET = np.asarray(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    ],
    dtype=float,
)


def xyz_to_aet_series(xyz: np.ndarray) -> np.ndarray:
    """Rotate an array with leading XYZ channel axis into orthonormal A/E/T."""
    values = np.asarray(xyz, dtype=float)
    if values.ndim < 2 or values.shape[0] != 3:
        raise ValueError("xyz must have shape (3, ...)")
    return np.einsum("cx,x...->c...", XYZ_TO_AET, values, optimize=True)


def xyz_covariance_to_aet_diagonal(xyz_covariance: np.ndarray) -> np.ndarray:
    """Return diagonal A/E/T PSDs from a full trailing-axis XYZ covariance.

    ``xyz_covariance`` has shape ``(..., 3, 3)`` and may be complex Hermitian.
    Only ``diag(M C M.T)`` is returned; callers need not estimate or retain AET
    cross spectra to use the correct analytic AET auto-PSDs.
    """
    covariance = np.asarray(xyz_covariance)
    if covariance.ndim < 2 or covariance.shape[-2:] != (3, 3):
        raise ValueError("xyz_covariance must have shape (..., 3, 3)")
    if np.any(~np.isfinite(covariance)):
        raise ValueError("xyz_covariance must be finite")
    aet_covariance = np.einsum(
        "ai,...ij,bj->...ab",
        XYZ_TO_AET,
        covariance,
        XYZ_TO_AET,
        optimize=True,
    )
    diagonal = np.real(np.diagonal(aet_covariance, axis1=-2, axis2=-1))
    if np.any(diagonal < 0.0):
        raise ValueError("rotated AET covariance has a negative diagonal")
    return diagonal


__all__ = [
    "AET_CHANNELS",
    "XYZ_TO_AET",
    "xyz_covariance_to_aet_diagonal",
    "xyz_to_aet_series",
]
