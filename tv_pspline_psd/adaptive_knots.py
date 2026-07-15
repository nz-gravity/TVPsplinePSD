"""Experimental Whittle-MAP pilot fits and data-adaptive knot allocation.

This module is deliberately independent of the NumPyro inference path.  It
uses the sufficient statistics of independent real Gaussian coefficients:
summed squared coefficients (``power``) and their effective count (``counts``).
Consequently, rectangular cells can be coarsened without changing the
likelihood for a surface that is constant inside each coarse cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import minimize

from .splines import (
    create_bspline_basis,
    create_bspline_roughness_penalty,
    evaluate_bspline_basis,
)


@dataclass(frozen=True)
class WhittleStatistics:
    """Power/count sufficient statistics on a tensor-product grid."""

    power: np.ndarray
    counts: np.ndarray
    time_grid: np.ndarray
    freq_grid: np.ndarray


@dataclass(frozen=True)
class WhittleMapResult:
    """Result of a fixed-knot penalized Whittle maximum-likelihood fit."""

    log_psd: np.ndarray
    coefficients: np.ndarray
    knots_time: np.ndarray
    knots_freq: np.ndarray
    degree_time: int
    degree_freq: int
    time_grid: np.ndarray
    freq_grid: np.ndarray
    power: np.ndarray
    counts: np.ndarray
    penalized_objective: float
    negative_log_likelihood: float
    success: bool
    message: str
    n_iterations: int
    gradient_norm: float

    def predict(self, time_grid: np.ndarray, freq_grid: np.ndarray) -> np.ndarray:
        """Evaluate the fitted log-PSD surface on another tensor grid."""
        bt = evaluate_bspline_basis(
            np.asarray(time_grid, dtype=float), self.knots_time, degree=self.degree_time
        )
        bf = evaluate_bspline_basis(
            np.asarray(freq_grid, dtype=float), self.knots_freq, degree=self.degree_freq
        )
        return bt @ self.coefficients @ bf.T


@dataclass(frozen=True)
class AdaptiveKnotResult:
    """Pilot fit and the separable knot densities derived from it."""

    pilot: WhittleMapResult
    time_knots: np.ndarray
    freq_knots: np.ndarray
    time_density: np.ndarray
    freq_density: np.ndarray
    method: str


@dataclass(frozen=True)
class ChiSquareKnotResult:
    """FastSpec-inspired knots from linear fits to a running-median spectrum."""

    knots: np.ndarray
    running_median_log_power: np.ndarray
    threshold: float
    residual_scale: float
    window_bins: int


def _validated_statistics(
    power: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    counts: float | np.ndarray,
) -> WhittleStatistics:
    power = np.asarray(power, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    freq_grid = np.asarray(freq_grid, dtype=float)
    counts = np.broadcast_to(np.asarray(counts, dtype=float), power.shape).copy()
    if power.ndim != 2 or power.shape != (time_grid.size, freq_grid.size):
        raise ValueError("power must have shape (len(time_grid), len(freq_grid)).")
    if time_grid.ndim != 1 or freq_grid.ndim != 1:
        raise ValueError("time_grid and freq_grid must be one-dimensional.")
    if np.any(np.diff(time_grid) <= 0) or np.any(np.diff(freq_grid) <= 0):
        raise ValueError("time_grid and freq_grid must be strictly increasing.")
    if not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("power must be finite and non-negative.")
    if not np.isfinite(counts).all() or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("counts must be finite, non-negative, and have positive sum.")
    if np.sum(power) <= 0:
        raise ValueError("power must contain at least one positive value.")
    return WhittleStatistics(power, counts, time_grid, freq_grid)


def coarsen_whittle_statistics(
    power: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    *,
    counts: float | np.ndarray = 1.0,
    time_factor: int = 1,
    freq_factor: int = 1,
) -> WhittleStatistics:
    """Sum power and counts in rectangular blocks, retaining block centers."""
    stats = _validated_statistics(power, time_grid, freq_grid, counts)
    if not isinstance(time_factor, int) or time_factor < 1:
        raise ValueError("time_factor must be a positive integer.")
    if not isinstance(freq_factor, int) or freq_factor < 1:
        raise ValueError("freq_factor must be a positive integer.")
    ti = np.arange(0, stats.time_grid.size, time_factor)
    fi = np.arange(0, stats.freq_grid.size, freq_factor)

    def block_sum(values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(np.add.reduceat(values, ti, axis=0), fi, axis=1)

    coarse_time = np.asarray(
        [stats.time_grid[i : i + time_factor].mean() for i in ti]
    )
    coarse_freq = np.asarray(
        [stats.freq_grid[i : i + freq_factor].mean() for i in fi]
    )
    return WhittleStatistics(
        block_sum(stats.power), block_sum(stats.counts), coarse_time, coarse_freq
    )


def fit_whittle_map(
    power: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    *,
    counts: float | np.ndarray = 1.0,
    train_mask: np.ndarray | None = None,
    n_interior_knots_time: int = 8,
    n_interior_knots_freq: int = 12,
    interior_knots_time: np.ndarray | None = None,
    interior_knots_freq: np.ndarray | None = None,
    degree_time: int = 3,
    degree_freq: int = 3,
    penalty_time: float = 0.05,
    penalty_freq: float = 0.05,
    maxiter: int = 300,
) -> WhittleMapResult:
    r"""Fit a convex penalized Whittle likelihood with analytic gradient.

    The cell contribution is
    ``0.5 * (counts * log(S) + power / S)``.  ``train_mask`` can exclude
    held-out cells without changing the shape of the fitted surface.
    """
    stats = _validated_statistics(power, time_grid, freq_grid, counts)
    fit_power = stats.power.copy()
    fit_counts = stats.counts.copy()
    if train_mask is not None:
        mask = np.asarray(train_mask, dtype=bool)
        if mask.shape != stats.power.shape:
            raise ValueError("train_mask must have the same shape as power.")
        fit_power *= mask
        fit_counts *= mask
        if fit_counts.sum() <= 0 or fit_power.sum() <= 0:
            raise ValueError("train_mask must retain positive counts and power.")
    if penalty_time < 0 or penalty_freq < 0:
        raise ValueError("penalties must be non-negative.")

    if interior_knots_time is not None:
        interior_knots_time = np.asarray(interior_knots_time, dtype=float)
        n_interior_knots_time = interior_knots_time.size
    if interior_knots_freq is not None:
        interior_knots_freq = np.asarray(interior_knots_freq, dtype=float)
        n_interior_knots_freq = interior_knots_freq.size
    bt, knots_t = create_bspline_basis(
        stats.time_grid, n_interior_knots_time, degree=degree_time,
        interior_knots=interior_knots_time,
    )
    bf, knots_f = create_bspline_basis(
        stats.freq_grid, n_interior_knots_freq, degree=degree_freq,
        interior_knots=interior_knots_freq,
    )
    pt = create_bspline_roughness_penalty(
        knots_t, degree=degree_time, derivative_order=min(2, degree_time)
    )
    pf = create_bspline_roughness_penalty(
        knots_f, degree=degree_freq, derivative_order=min(2, degree_freq)
    )
    scale = fit_power.sum() / fit_counts.sum()
    scaled_power = fit_power / scale
    shape = (bt.shape[1], bf.shape[1])

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        w = flat.reshape(shape)
        surface = bt @ w @ bf.T
        inv_relative_psd = np.exp(np.minimum(-surface, 700.0))
        residual = 0.5 * (fit_counts - scaled_power * inv_relative_psd)
        likelihood = 0.5 * np.sum(
            fit_counts * surface + scaled_power * inv_relative_psd
        )
        roughness = 0.5 * (
            penalty_time * np.sum(w * (pt @ w))
            + penalty_freq * np.sum(w * (w @ pf))
        )
        gradient = (
            bt.T @ residual @ bf
            + penalty_time * (pt @ w)
            + penalty_freq * (w @ pf)
        )
        return float(likelihood + roughness), gradient.ravel()

    optimized = minimize(
        objective,
        np.zeros(np.prod(shape), dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-7},
    )
    w = optimized.x.reshape(shape)
    # Put the scale into the coefficient null space: B-spline rows sum to one.
    coefficients = w + np.log(scale)
    log_psd = bt @ coefficients @ bf.T
    nll = 0.5 * np.sum(fit_counts * log_psd + fit_power * np.exp(-log_psd))
    objective_with_scale = float(optimized.fun + 0.5 * fit_counts.sum() * np.log(scale))
    return WhittleMapResult(
        log_psd=log_psd,
        coefficients=coefficients,
        knots_time=knots_t,
        knots_freq=knots_f,
        degree_time=degree_time,
        degree_freq=degree_freq,
        time_grid=stats.time_grid,
        freq_grid=stats.freq_grid,
        power=fit_power,
        counts=fit_counts,
        penalized_objective=objective_with_scale,
        negative_log_likelihood=float(nll),
        success=bool(optimized.success),
        message=str(optimized.message),
        n_iterations=int(optimized.nit),
        gradient_norm=float(np.linalg.norm(optimized.jac)),
    )


def whittle_deviance(
    power: np.ndarray,
    counts: float | np.ndarray,
    log_psd: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return non-negative twice-log-likelihood loss from the saturated fit."""
    power = np.asarray(power, dtype=float)
    counts = np.broadcast_to(np.asarray(counts, dtype=float), power.shape)
    log_psd = np.asarray(log_psd, dtype=float)
    if log_psd.shape != power.shape or np.any(counts < 0):
        raise ValueError("power, counts, and log_psd must have compatible shapes.")
    positive = power[power > 0]
    floor = positive.min() * 0.5 if positive.size else np.finfo(float).tiny
    observed = np.maximum(power, floor) / np.maximum(counts, 1.0)
    ratio = observed * np.exp(-log_psd)
    deviance = counts * (ratio - 1.0 - np.log(ratio))
    deviance = np.maximum(deviance, 0.0)
    if mask is not None:
        deviance = np.where(np.asarray(mask, dtype=bool), deviance, 0.0)
    return deviance


def _normalized(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    mean = np.trapezoid(values, x) / (x[-1] - x[0])
    return values / max(mean, np.finfo(float).tiny)


def _project_min_spacing(
    knots: np.ndarray, lower: float, upper: float, spacing: float
) -> np.ndarray:
    n = len(knots)
    if n == 0:
        return knots
    if spacing < 0 or spacing * (n + 1) > upper - lower:
        raise ValueError("minimum spacing is negative or infeasible for the knot count.")
    lo = lower + spacing * np.arange(1, n + 1)
    hi = upper - spacing * np.arange(n, 0, -1)
    projected = np.clip(knots, lo, hi)
    for i in range(1, n):
        projected[i] = max(projected[i], projected[i - 1] + spacing)
    for i in range(n - 2, -1, -1):
        projected[i] = min(projected[i], projected[i + 1] - spacing)
    return np.clip(projected, lo, hi)


def allocate_knots_from_density(
    grid: np.ndarray,
    density: np.ndarray,
    n_interior_knots: int,
    *,
    min_spacing: float = 0.0,
) -> np.ndarray:
    """Place knots at integrated-density quantiles with a spacing constraint."""
    grid = np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    if grid.ndim != 1 or density.shape != grid.shape or np.any(density < 0):
        raise ValueError("grid and non-negative density must be matching 1D arrays.")
    if n_interior_knots < 0:
        raise ValueError("n_interior_knots must be non-negative.")
    if n_interior_knots == 0:
        return np.array([], dtype=float)
    increments = 0.5 * (density[:-1] + density[1:]) * np.diff(grid)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    if cdf[-1] <= 0:
        raise ValueError("density must have positive integral.")
    cdf /= cdf[-1]
    targets = np.arange(1, n_interior_knots + 1) / (n_interior_knots + 1)
    knots = np.interp(targets, cdf, grid)
    return _project_min_spacing(knots, grid[0], grid[-1], min_spacing)


def fit_running_median_chi2_knots(
    power: np.ndarray,
    freq_grid: np.ndarray,
    n_interior_knots: int,
    *,
    median_window_hz: float,
    min_window_bins: int = 4,
    min_spacing_fraction: float = 0.2,
) -> ChiSquareKnotResult:
    """Allocate a fixed knot budget with FastSpec-style linear-fit windows.

    The input may be a single power spectrum or a ``(time, frequency)`` power
    array.  For a surface, the pilot spectrum is the pointwise median over
    time.  A running median suppresses individual noisy cells, then a window
    grows from low to high frequency until the chi-square of a linear fit to
    the running-median log power crosses a threshold.  The threshold is found
    by bisection so the output uses the requested fixed knot budget.

    This deliberately adapts FastSpec's placement idea rather than its model
    dimension: the P-spline basis size remains fixed for NumPyro/NUTS.
    """
    freq = np.asarray(freq_grid, dtype=float)
    values = np.asarray(power, dtype=float)
    if freq.ndim != 1 or freq.size < 3 or np.any(np.diff(freq) <= 0):
        raise ValueError("freq_grid must be a strictly increasing 1D array.")
    if values.ndim == 1:
        if values.shape != freq.shape:
            raise ValueError("1D power must match freq_grid.")
        profile = values
    elif values.ndim == 2:
        if values.shape[1] != freq.size or values.shape[0] == 0:
            raise ValueError("2D power must have shape (time, len(freq_grid)).")
        profile = np.median(values, axis=0)
    else:
        raise ValueError("power must be one- or two-dimensional.")
    if not np.isfinite(values).all() or np.any(values < 0) or not np.any(values > 0):
        raise ValueError("power must be finite, non-negative, and contain a positive value.")
    if not isinstance(n_interior_knots, int) or n_interior_knots < 0:
        raise ValueError("n_interior_knots must be a non-negative integer.")
    if n_interior_knots == 0:
        return ChiSquareKnotResult(
            knots=np.array([], dtype=float),
            running_median_log_power=np.log(np.maximum(profile, np.min(values[values > 0]))),
            threshold=np.inf,
            residual_scale=1.0,
            window_bins=1,
        )
    if median_window_hz <= 0 or not np.isfinite(median_window_hz):
        raise ValueError("median_window_hz must be finite and positive.")
    if not isinstance(min_window_bins, int) or min_window_bins < 3:
        raise ValueError("min_window_bins must be an integer of at least 3.")
    if min_window_bins * (n_interior_knots + 1) >= freq.size:
        raise ValueError("frequency grid is too short for the knot and window counts.")

    df = float(np.median(np.diff(freq)))
    window_bins = max(3, int(round(median_window_hz / df)))
    window_bins += 1 - window_bins % 2
    positive = values[values > 0]
    floor = float(np.min(positive) * 0.5)
    running = median_filter(np.maximum(profile, floor), size=window_bins, mode="nearest")
    log_running = np.log(running)

    differences = np.diff(log_running)
    mad = np.median(np.abs(differences - np.median(differences)))
    residual_scale = float(max(1.4826 * mad / np.sqrt(2.0), np.finfo(float).eps))
    # Removing the arbitrary log-power offset improves the prefix-sum RSS
    # numerics and makes placement exactly invariant to physical power units.
    y = (log_running - log_running[0]) / residual_scale
    x = np.arange(freq.size, dtype=float)
    sx = np.concatenate(([0.0], np.cumsum(x)))
    sy = np.concatenate(([0.0], np.cumsum(y)))
    sxx = np.concatenate(([0.0], np.cumsum(x * x)))
    sxy = np.concatenate(([0.0], np.cumsum(x * y)))
    syy = np.concatenate(([0.0], np.cumsum(y * y)))

    def linear_rss(start: int, stop: int) -> float:
        count = stop - start + 1
        sum_x = sx[stop + 1] - sx[start]
        sum_y = sy[stop + 1] - sy[start]
        sum_xx = sxx[stop + 1] - sxx[start]
        sum_xy = sxy[stop + 1] - sxy[start]
        sum_yy = syy[stop + 1] - syy[start]
        denominator = count * sum_xx - sum_x * sum_x
        explained = (
            sum_xx * sum_y * sum_y
            - 2.0 * sum_x * sum_y * sum_xy
            + count * sum_xy * sum_xy
        ) / max(denominator, np.finfo(float).tiny)
        return float(max(sum_yy - explained, 0.0))

    def window_ends(threshold: float) -> np.ndarray:
        ends: list[int] = []
        start = 0
        last = freq.size - 1
        while start + min_window_bins < last:
            selected = last
            for stop in range(start + min_window_bins - 1, last + 1):
                if linear_rss(start, stop) > threshold:
                    selected = stop
                    break
            if selected >= last:
                break
            ends.append(selected)
            start = selected
        return np.asarray(ends, dtype=int)

    low = 0.0
    high = max(linear_rss(0, freq.size - 1), 1.0)
    while window_ends(high).size > n_interior_knots:
        high *= 2.0
    best_threshold = high
    best_ends = window_ends(high)
    for _ in range(80):
        middle = 0.5 * (low + high)
        ends = window_ends(middle)
        if abs(ends.size - n_interior_knots) < abs(best_ends.size - n_interior_knots):
            best_threshold, best_ends = middle, ends
        if ends.size > n_interior_knots:
            low = middle
        else:
            high = middle
            if ends.size == n_interior_knots:
                best_threshold, best_ends = middle, ends

    if best_ends.size != n_interior_knots:
        raise RuntimeError(
            "could not calibrate the chi-square threshold to the requested knot count; "
            f"nearest count was {best_ends.size}"
        )
    spacing = min_spacing_fraction * (freq[-1] - freq[0]) / (n_interior_knots + 1)
    knots = _project_min_spacing(freq[best_ends], freq[0], freq[-1], spacing)
    return ChiSquareKnotResult(
        knots=knots,
        running_median_log_power=log_running,
        threshold=float(best_threshold),
        residual_scale=residual_scale,
        window_bins=window_bins,
    )


def fit_adaptive_knots(
    power: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    *,
    counts: float | np.ndarray = 1.0,
    train_mask: np.ndarray | None = None,
    n_pilot_knots_time: int = 8,
    n_pilot_knots_freq: int = 16,
    n_knots_time: int = 8,
    n_knots_freq: int = 10,
    method: str = "curvature",
    coarsen_time: int = 1,
    coarsen_freq: int = 1,
    density_floor: float = 0.2,
    mixed_weight: float = 0.25,
    min_spacing_fraction: float = 0.2,
    penalty_time: float = 0.05,
    penalty_freq: float = 0.05,
) -> AdaptiveKnotResult:
    """Fit a cheap pilot and allocate time and frequency knots from its surface."""
    if method not in {"curvature", "deviance", "hybrid"}:
        raise ValueError("method must be 'curvature', 'deviance', or 'hybrid'.")
    stats = _validated_statistics(power, time_grid, freq_grid, counts)
    fit_power, fit_counts = stats.power, stats.counts
    if train_mask is not None:
        mask = np.asarray(train_mask, dtype=bool)
        if mask.shape != stats.power.shape:
            raise ValueError("train_mask must have the same shape as power.")
        fit_power = fit_power * mask
        fit_counts = fit_counts * mask
    coarse = coarsen_whittle_statistics(
        fit_power,
        stats.time_grid,
        stats.freq_grid,
        counts=fit_counts,
        time_factor=coarsen_time,
        freq_factor=coarsen_freq,
    )
    pilot = fit_whittle_map(
        coarse.power,
        coarse.time_grid,
        coarse.freq_grid,
        counts=coarse.counts,
        n_interior_knots_time=n_pilot_knots_time,
        n_interior_knots_freq=n_pilot_knots_freq,
        penalty_time=penalty_time,
        penalty_freq=penalty_freq,
    )
    eta = pilot.log_psd
    edge_t = 2 if coarse.time_grid.size >= 3 else 1
    edge_f = 2 if coarse.freq_grid.size >= 3 else 1
    dt2 = np.gradient(np.gradient(eta, coarse.time_grid, axis=0, edge_order=edge_t), coarse.time_grid, axis=0, edge_order=edge_t)
    df2 = np.gradient(np.gradient(eta, coarse.freq_grid, axis=1, edge_order=edge_f), coarse.freq_grid, axis=1, edge_order=edge_f)
    mixed = np.gradient(np.gradient(eta, coarse.time_grid, axis=0, edge_order=edge_t), coarse.freq_grid, axis=1, edge_order=edge_f)
    curvature_t = np.sqrt(np.mean(dt2**2, axis=1))
    curvature_f = np.sqrt(np.mean(df2**2, axis=0))
    mixed_t = np.sqrt(np.mean(mixed**2, axis=1))
    mixed_f = np.sqrt(np.mean(mixed**2, axis=0))
    dev = whittle_deviance(coarse.power, coarse.counts, eta)
    dev_t = np.mean(dev, axis=1)
    dev_f = np.mean(dev, axis=0)
    curv_density_t = _normalized(curvature_t, coarse.time_grid) + mixed_weight * _normalized(mixed_t, coarse.time_grid)
    curv_density_f = _normalized(curvature_f, coarse.freq_grid) + mixed_weight * _normalized(mixed_f, coarse.freq_grid)
    if method == "curvature":
        density_t, density_f = curv_density_t, curv_density_f
    elif method == "deviance":
        density_t = _normalized(dev_t, coarse.time_grid)
        density_f = _normalized(dev_f, coarse.freq_grid)
    else:
        density_t = curv_density_t + _normalized(dev_t, coarse.time_grid)
        density_f = curv_density_f + _normalized(dev_f, coarse.freq_grid)
    density_t = density_floor + density_t
    density_f = density_floor + density_f
    spacing_t = min_spacing_fraction * (coarse.time_grid[-1] - coarse.time_grid[0]) / (n_knots_time + 1)
    spacing_f = min_spacing_fraction * (coarse.freq_grid[-1] - coarse.freq_grid[0]) / (n_knots_freq + 1)
    return AdaptiveKnotResult(
        pilot=pilot,
        time_knots=allocate_knots_from_density(coarse.time_grid, density_t, n_knots_time, min_spacing=spacing_t),
        freq_knots=allocate_knots_from_density(coarse.freq_grid, density_f, n_knots_freq, min_spacing=spacing_f),
        time_density=density_t,
        freq_density=density_f,
        method=method,
    )
