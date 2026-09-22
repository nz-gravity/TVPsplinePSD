"""Input validation and spline preparation for surface inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adaptive_knots import fit_adaptive_knots
from .binning import _reference_scaled_power, bin_power_rectangular
from .config import PSplineConfig
from .model import whiten_penalty_pair
from .splines import (
    create_bspline_basis,
    create_bspline_roughness_penalty,
    evaluate_bspline_basis,
)


def validate_surface_inputs(
    coeffs: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    config: PSplineConfig,
    *,
    likelihood_beta: float,
    residual_structure: str,
    interaction_scale_prior: float,
    interaction_time_knots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize coefficient/grid arrays and reject invalid model settings."""
    coeffs = np.asarray(coeffs, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    freq_grid = np.asarray(freq_grid, dtype=float)
    if coeffs.ndim != 3:
        raise ValueError("coeffs must have shape (R, n_time, n_freq).")
    if coeffs.shape[0] == 0 or coeffs.shape[1] == 0 or coeffs.shape[2] == 0:
        raise ValueError(
            "coeffs and the analysis grid must be non-empty after trimming."
        )
    if time_grid.ndim != 1 or freq_grid.ndim != 1:
        raise ValueError("time_grid and freq_grid must be one-dimensional.")
    if coeffs.shape[1] != time_grid.size or coeffs.shape[2] != freq_grid.size:
        raise ValueError(
            "coeffs shape must match time_grid and freq_grid: expected "
            f"(*, {time_grid.size}, {freq_grid.size}), got {coeffs.shape}."
        )
    if not np.isfinite(coeffs).all():
        raise ValueError("coeffs must contain only finite values.")
    if not np.isfinite(time_grid).all() or not np.isfinite(freq_grid).all():
        raise ValueError("time_grid and freq_grid must contain only finite values.")
    if np.any(np.diff(time_grid) <= 0) or np.any(np.diff(freq_grid) <= 0):
        raise ValueError("time_grid and freq_grid must be strictly increasing.")
    if not 0.0 <= likelihood_beta <= 1.0:
        raise ValueError("likelihood_beta must lie in [0, 1].")
    if residual_structure not in {"tensor", "stationary_plus_interaction"}:
        raise ValueError(
            "residual_structure must be 'tensor' or 'stationary_plus_interaction'"
        )
    if not np.isfinite(interaction_scale_prior) or interaction_scale_prior <= 0.0:
        raise ValueError("interaction_scale_prior must be finite and positive")
    if (
        not isinstance(interaction_time_knots, (int, np.integer))
        or isinstance(interaction_time_knots, bool)
        or interaction_time_knots < 0
    ):
        raise ValueError("interaction_time_knots must be a non-negative integer")
    if residual_structure == "stationary_plus_interaction" and not config.centered:
        raise ValueError(
            "stationary_plus_interaction currently requires config.centered=True"
        )
    return coeffs, time_grid, freq_grid


def _validate_explicit_interior_knots(
    knots: np.ndarray | None,
    grid: np.ndarray,
    expected_count: int,
    *,
    axis: str,
) -> np.ndarray | None:
    """Validate explicit knots before basis construction or sampler startup."""
    if knots is None:
        return None
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 1:
        raise ValueError(f"interior_knots_{axis} must be one-dimensional.")
    if knots.size != expected_count:
        raise ValueError(
            f"interior_knots_{axis} must contain exactly {expected_count} values "
            f"to match config.n_interior_knots_{axis}."
        )
    if not np.isfinite(knots).all():
        raise ValueError(f"interior_knots_{axis} must contain only finite values.")
    if np.any(np.diff(knots) <= 0):
        raise ValueError(f"interior_knots_{axis} must be strictly increasing.")
    if np.any(knots <= grid[0]) or np.any(knots >= grid[-1]):
        unit = " Hz" if axis == "freq" else ""
        raise ValueError(
            f"interior_knots_{axis} must lie strictly inside the analysis-grid "
            f"range ({grid[0]:g}, {grid[-1]:g}){unit}."
        )
    return knots


def _prepare_spline_bases(
    power: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    config: PSplineConfig,
    *,
    n_components: int,
    likelihood_mask: np.ndarray | None = None,
    interior_knots_time: np.ndarray | None = None,
    interior_knots_freq: np.ndarray | None = None,
) -> dict[str, object]:
    """Build production bases and resolve the configured knot allocation."""
    explicit_time = _validate_explicit_interior_knots(
        interior_knots_time,
        time_grid,
        config.n_interior_knots_time,
        axis="time",
    )
    explicit_freq = _validate_explicit_interior_knots(
        interior_knots_freq,
        freq_grid,
        config.n_interior_knots_freq,
        axis="freq",
    )
    selected_time = explicit_time
    selected_freq = explicit_freq
    allocation = {
        "time": "explicit" if explicit_time is not None else "linear",
        "frequency": "explicit"
        if explicit_freq is not None
        else config.freq_knot_strategy,
    }
    if selected_freq is None:
        if config.freq_knot_strategy == "adaptive":
            pilot = fit_adaptive_knots(
                power,
                time_grid,
                freq_grid,
                counts=float(n_components),
                train_mask=likelihood_mask,
                n_pilot_knots_time=max(8, config.n_interior_knots_time),
                n_pilot_knots_freq=max(16, config.n_interior_knots_freq),
                n_knots_time=config.n_interior_knots_time,
                n_knots_freq=config.n_interior_knots_freq,
                method="curvature",
            )
            selected_freq = pilot.freq_knots
        elif config.freq_knot_strategy == "log":
            if freq_grid[0] <= 0:
                raise ValueError(
                    "freq_knot_strategy='log' requires a strictly positive frequency grid."
                )
            selected_freq = np.geomspace(
                freq_grid[0], freq_grid[-1], config.n_interior_knots_freq + 2
            )[1:-1]

    freq_scale = np.maximum(freq_grid[-1], 1e-12)
    freq_unit = freq_grid / freq_scale
    selected_freq_unit = None if selected_freq is None else selected_freq / freq_scale
    B_time, knots_time = create_bspline_basis(
        time_grid,
        config.n_interior_knots_time,
        degree=config.degree_time,
        interior_knots=selected_time,
    )
    B_freq, knots_freq_unit = create_bspline_basis(
        freq_unit,
        config.n_interior_knots_freq,
        degree=config.degree_freq,
        interior_knots=selected_freq_unit,
    )
    return {
        "B_time": B_time,
        "B_freq": B_freq,
        "knots_time": knots_time,
        "knots_freq_unit": knots_freq_unit,
        "knots_time_physical": knots_time.copy(),
        "knots_freq_physical": knots_freq_unit * freq_scale,
        "knot_allocation": allocation,
    }


@dataclass(frozen=True)
class NestedBasis:
    """Centered interaction basis and its native-grid reconstruction arrays."""

    knots_time: np.ndarray
    time_basis_mean: np.ndarray
    B_time: np.ndarray
    P_time: np.ndarray
    whitened: dict[str, np.ndarray]
    basis_time: np.ndarray
    basis_freq: np.ndarray
    null_freq: np.ndarray


@dataclass(frozen=True)
class SurfaceBasis:
    """Native spline bases, penalties and optional nested interaction."""

    spline: dict[str, object]
    P_time: np.ndarray
    P_freq: np.ndarray
    whitened: dict[str, np.ndarray]
    basis_time: np.ndarray
    basis_freq: np.ndarray
    interaction: NestedBasis | None


@dataclass(frozen=True)
class LikelihoodGrid:
    """Sufficient statistics and bases on the possibly pooled fit grid."""

    power: np.ndarray
    counts: np.ndarray | int
    B_time: np.ndarray
    B_freq: np.ndarray
    basis_time: np.ndarray
    basis_freq: np.ndarray
    log_offset: np.ndarray
    coarse_grained: bool
    B_time_interaction: np.ndarray | None
    basis_interaction_time: np.ndarray | None
    basis_nested_freq: np.ndarray | None


def prepare_surface_basis(
    power: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    config: PSplineConfig,
    *,
    n_components: int,
    likelihood_mask: np.ndarray | None,
    interior_knots_time: np.ndarray | None,
    interior_knots_freq: np.ndarray | None,
    nested: bool,
    interaction_time_knots: int,
) -> SurfaceBasis:
    """Construct native-grid bases and whiten penalties for the chosen model."""
    spline = _prepare_spline_bases(
        power,
        time_grid,
        freq_grid,
        config,
        n_components=n_components,
        likelihood_mask=likelihood_mask,
        interior_knots_time=interior_knots_time,
        interior_knots_freq=interior_knots_freq,
    )
    B_time = spline["B_time"]
    B_freq = spline["B_freq"]
    knots_time = spline["knots_time"]
    knots_freq = spline["knots_freq_unit"]
    P_time = create_bspline_roughness_penalty(
        knots_time, degree=config.degree_time, derivative_order=config.diff_order_time
    )
    P_freq = create_bspline_roughness_penalty(
        knots_freq, degree=config.degree_freq, derivative_order=config.diff_order_freq
    )
    whitened = whiten_penalty_pair(P_time, P_freq)
    basis_eig_time = B_time @ whitened["U_time"]
    basis_eig_freq = B_freq @ whitened["U_freq"]
    interaction = None
    if nested:
        B_time_interaction_full, knots_time_interaction = create_bspline_basis(
            time_grid,
            interaction_time_knots,
            degree=config.degree_time,
        )
        P_time_interaction_full = create_bspline_roughness_penalty(
            knots_time_interaction,
            degree=config.degree_time,
            derivative_order=config.diff_order_time,
        )
        # Partition unity makes the final centered B-spline column redundant.
        # Dropping it yields a full-rank basis spanning all zero-time-mean
        # spline functions, so h(t,f) cannot absorb the stationary g(f).
        time_basis_mean = B_time_interaction_full.mean(axis=0)
        B_time_interaction = (B_time_interaction_full - time_basis_mean[None, :])[
            :, :-1
        ]
        P_time_interaction = P_time_interaction_full[:-1, :-1]
        whitened_interaction = whiten_penalty_pair(P_time_interaction, P_freq)
        basis_interaction_time = B_time_interaction @ whitened_interaction["U_time"]
        basis_nested_freq = B_freq @ whitened_interaction["U_freq"]
        null_nested_freq = whitened_interaction["lam_freq"] <= 1e-10 * max(
            whitened_interaction["lam_freq"].max(), 1.0
        )

        interaction = NestedBasis(
            knots_time_interaction,
            time_basis_mean,
            B_time_interaction,
            P_time_interaction,
            whitened_interaction,
            basis_interaction_time,
            basis_nested_freq,
            null_nested_freq,
        )
    return SurfaceBasis(
        spline, P_time, P_freq, whitened, basis_eig_time, basis_eig_freq, interaction
    )


def prepare_likelihood_grid(
    power: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    basis: SurfaceBasis,
    config: PSplineConfig,
    *,
    n_components: int,
    likelihood_mask: np.ndarray | None,
    validated_log_offset: np.ndarray,
    offset_applied: bool,
    time_bin: int,
    freq_bin: int,
    time_bin_starts: np.ndarray | None,
    freq_bin_starts: np.ndarray | None,
) -> LikelihoodGrid:
    """Pool raw power/counts and evaluate bases without losing reference variation."""
    B_time, B_freq = basis.spline["B_time"], basis.spline["B_freq"]
    knots_time, knots_freq = basis.spline["knots_time"], basis.spline["knots_freq_unit"]
    whitened = basis.whitened
    basis_eig_time, basis_eig_freq = basis.basis_time, basis.basis_freq
    mask_applied = likelihood_mask is not None
    validated_likelihood_mask = likelihood_mask
    B_time_interaction_fit = basis_interaction_time_fit = basis_nested_freq_fit = None
    coarse_grained = (
        time_bin > 1
        or freq_bin > 1
        or time_bin_starts is not None
        or freq_bin_starts is not None
    )
    if coarse_grained:
        # If S_i = R_i exp(r_b), where the spline residual r_b is treated as
        # constant within a coarse block, the exact parameter-dependent block
        # likelihood is
        #   -1/2 [N_b r_b + exp(-r_b) sum_i(power_i / R_i)].
        # The data-only sum_i log(R_i) is omitted, consistently with the other
        # Whittle constants.  In particular, do not average log(R) and reuse its
        # geometric mean in the quadratic term: that is biased whenever the
        # reference varies inside a bin (notably near moving response nulls).
        power_for_fit = (
            _reference_scaled_power(
                power,
                validated_log_offset,
                validated_likelihood_mask if mask_applied else None,
            )
            if offset_applied
            else power
        )
        power_fit, time_grid_fit, freq_grid_fit, counts_fit = bin_power_rectangular(
            power_for_fit,
            time_grid,
            freq_grid,
            n_components,
            time_bin=time_bin,
            freq_bin=freq_bin,
            time_bin_starts=(time_bin_starts),
            freq_bin_starts=freq_bin_starts,
            likelihood_mask=(validated_likelihood_mask if mask_applied else None),
        )
        B_time_fit = evaluate_bspline_basis(
            time_grid_fit, knots_time, degree=config.degree_time
        )
        B_freq_fit = evaluate_bspline_basis(
            freq_grid_fit / np.maximum(freq_grid[-1], 1e-12),
            knots_freq,
            degree=config.degree_freq,
        )
        basis_eig_time_fit = B_time_fit @ whitened["U_time"]
        basis_eig_freq_fit = B_freq_fit @ whitened["U_freq"]
        # The coarse model samples the residual log PSD.  The original
        # full-resolution reference is added back only during reconstruction.
        log_offset_fit = np.zeros_like(power_fit)
    else:
        if mask_applied:
            power_fit = np.where(validated_likelihood_mask, power, 0.0)
            counts_fit = n_components * validated_likelihood_mask.astype(int)
        else:
            power_fit = power
            counts_fit = n_components
        B_time_fit = B_time
        B_freq_fit = B_freq
        time_grid_fit = time_grid
        basis_eig_time_fit = basis_eig_time
        basis_eig_freq_fit = basis_eig_freq
        log_offset_fit = validated_log_offset

    if basis.interaction is not None:
        knots_time_interaction = basis.interaction.knots_time
        time_basis_mean = basis.interaction.time_basis_mean
        whitened_interaction = basis.interaction.whitened
        B_time_interaction_fit_full = evaluate_bspline_basis(
            time_grid_fit,
            knots_time_interaction,
            degree=config.degree_time,
        )
        B_time_interaction_fit = (
            B_time_interaction_fit_full - time_basis_mean[None, :]
        )[:, :-1]
        basis_interaction_time_fit = (
            B_time_interaction_fit @ whitened_interaction["U_time"]
        )
        basis_nested_freq_fit = B_freq_fit @ whitened_interaction["U_freq"]

    return LikelihoodGrid(
        power_fit,
        counts_fit,
        B_time_fit,
        B_freq_fit,
        basis_eig_time_fit,
        basis_eig_freq_fit,
        log_offset_fit,
        coarse_grained,
        B_time_interaction_fit,
        basis_interaction_time_fit,
        basis_nested_freq_fit,
    )
