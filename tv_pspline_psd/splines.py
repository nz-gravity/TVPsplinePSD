"""B-spline bases and roughness penalties for the WDM log-power surface."""

from __future__ import annotations

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d


def create_bspline_basis(
    x: np.ndarray,
    n_interior_knots: int,
    *,
    degree: int = 3,
    interior_knots: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a row-normalized B-spline basis on the supplied grid.

    Args:
        x: Evaluation grid (1D).
        n_interior_knots: Number of interior knots when ``interior_knots`` is None.
        degree: B-spline degree.
        interior_knots: Optional explicit interior knots (e.g. adaptive placement).

    Returns:
        A tuple ``(basis, knots)`` where ``basis`` has shape ``(len(x), n_basis)``
        and ``knots`` is the full (clamped) knot vector.
    """
    x = np.asarray(x, dtype=float)
    if interior_knots is None:
        interior = np.linspace(x.min(), x.max(), n_interior_knots + 2)[1:-1]
    else:
        interior = np.asarray(interior_knots, dtype=float)
        interior = interior[(interior > x.min()) & (interior < x.max())]
        if len(interior) != n_interior_knots:
            raise ValueError("interior_knots must match n_interior_knots.")
    knots = np.concatenate(
        [
            np.repeat(x.min(), degree + 1),
            interior,
            np.repeat(x.max(), degree + 1),
        ]
    )
    basis = _evaluate_basis(x, knots, degree)
    return basis, knots


def evaluate_bspline_basis(
    x: np.ndarray,
    knots: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    """Evaluate a B-spline basis defined by a full knot vector on a new grid."""
    return _evaluate_basis(np.asarray(x, dtype=float), np.asarray(knots, dtype=float), degree)


def _evaluate_basis(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    n_basis = len(knots) - degree - 1
    basis = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spline = interpolate.BSpline(knots, coeffs, degree, extrapolate=False)
        basis[:, i] = spline(x)
    basis = np.nan_to_num(basis)
    row_sums = basis.sum(axis=1, keepdims=True)
    basis /= np.maximum(row_sums, 1e-12)
    return basis


def create_adaptive_time_knots(
    x: np.ndarray,
    pilot_profile: np.ndarray,
    *,
    n_interior_knots: int,
    smoothing_sigma: float = 1.0,
    variation_floor: float = 0.25,
) -> np.ndarray:
    """Place more time knots where a pilot time profile changes fastest."""
    x = np.asarray(x, dtype=float)
    pilot_profile = np.asarray(pilot_profile, dtype=float)
    if x.ndim != 1 or pilot_profile.ndim != 1 or len(x) != len(pilot_profile):
        raise ValueError("x and pilot_profile must be 1D with matching length.")
    if n_interior_knots <= 0:
        return np.array([], dtype=float)

    smooth_profile = gaussian_filter1d(pilot_profile, sigma=smoothing_sigma, mode="nearest")
    local_variation = np.abs(np.gradient(smooth_profile, x))
    density = np.maximum(variation_floor + local_variation, 1e-10)
    cdf = np.cumsum(density)
    cdf = (cdf - cdf[0]) / np.maximum(cdf[-1] - cdf[0], 1e-12)
    targets = np.linspace(0.0, 1.0, n_interior_knots + 2)[1:-1]
    interior = np.interp(targets, cdf, x)

    interior = np.maximum.accumulate(interior)
    eps = np.finfo(float).eps * max(1.0, x.max() - x.min()) * 32.0
    for i in range(1, len(interior)):
        if interior[i] <= interior[i - 1]:
            interior[i] = interior[i - 1] + eps
    return np.clip(interior, x.min() + eps, x.max() - eps)


def create_bspline_roughness_penalty(
    knots: np.ndarray,
    *,
    degree: int,
    derivative_order: int = 2,
    quad_order: int = 8,
) -> np.ndarray:
    r"""Derivative-based B-spline roughness matrix.

    Entries are ``R_{ij} = \int B_i^{(q)}(x) B_j^{(q)}(x) dx`` with
    ``q = derivative_order``, evaluated by Gauss-Legendre quadrature on each
    non-degenerate knot span and normalized by its trace.
    """
    if derivative_order > degree:
        raise ValueError("derivative_order must be <= degree.")
    n_basis = len(knots) - degree - 1
    coeffs = np.eye(n_basis)
    deriv_splines = [
        interpolate.BSpline(knots, coeffs[i], degree, extrapolate=False).derivative(
            derivative_order
        )
        for i in range(n_basis)
    ]
    penalty = np.zeros((n_basis, n_basis))
    abscissa, weights = np.polynomial.legendre.leggauss(quad_order)
    for left, right in zip(knots[:-1], knots[1:]):
        if right <= left:
            continue
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        x_eval = midpoint + half_width * abscissa
        values = np.stack([spline(x_eval) for spline in deriv_splines], axis=0)
        values = np.nan_to_num(values)
        penalty += (values * weights[None, :]) @ values.T * half_width
    penalty = 0.5 * (penalty + penalty.T)
    return penalty / np.maximum(np.trace(penalty), 1e-12)


def create_difference_penalty_matrix(n_basis: int, *, diff_order: int = 2) -> np.ndarray:
    """Return the trace-normalized finite-difference penalty matrix ``D^T D``."""
    if n_basis <= diff_order:
        raise ValueError("Need more basis functions than the penalty order.")
    D = np.diff(np.eye(n_basis), n=diff_order, axis=0)
    penalty = D.T @ D
    return penalty / np.maximum(np.trace(penalty), 1e-12)
