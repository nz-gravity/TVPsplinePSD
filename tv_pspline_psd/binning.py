"""Time-frequency likelihood partitions, masked power pooling and initialization."""

from __future__ import annotations

import numpy as np


def _regular_bin_starts(size: int, bin_size: int) -> np.ndarray:
    return np.arange(0, size, bin_size, dtype=int)


def _validate_bin_starts(
    starts: np.ndarray | None,
    size: int,
    bin_size: int,
    *,
    axis: str,
) -> np.ndarray:
    if (
        not isinstance(bin_size, (int, np.integer))
        or isinstance(bin_size, bool)
        or bin_size < 1
    ):
        raise ValueError(f"{axis}_bin must be a positive integer.")
    if starts is None:
        return _regular_bin_starts(size, int(bin_size))
    if bin_size != 1:
        raise ValueError(f"{axis}_bin must be 1 when {axis}_bin_starts is provided.")
    starts = np.asarray(starts)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError(
            f"{axis}_bin_starts must be a non-empty one-dimensional array."
        )
    if not np.issubdtype(starts.dtype, np.integer):
        raise ValueError(f"{axis}_bin_starts must contain integer indices.")
    starts = starts.astype(int, copy=False)
    if starts[0] != 0 or starts[-1] >= size or np.any(np.diff(starts) <= 0):
        raise ValueError(
            f"{axis}_bin_starts must begin at 0 and contain strictly increasing "
            f"indices smaller than the {axis} grid size ({size})."
        )
    return starts


def _validate_likelihood_mask(
    likelihood_mask: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a boolean per-cell likelihood mask on the analysis grid."""
    if likelihood_mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(likelihood_mask)
    if mask.shape != shape:
        raise ValueError(
            "likelihood_mask must match the (time, frequency) analysis grid: "
            f"expected {shape}, got {mask.shape}."
        )
    if mask.dtype != np.bool_:
        raise ValueError("likelihood_mask must contain boolean values.")
    if not np.any(mask):
        raise ValueError("likelihood_mask must retain at least one analysis cell.")
    return mask


def bin_power_rectangular(
    power: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    n_components: int,
    *,
    time_bin: int = 1,
    freq_bin: int = 1,
    time_bin_starts: np.ndarray | None = None,
    freq_bin_starts: np.ndarray | None = None,
    likelihood_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum powers over a separable rectangular time--frequency partition.

    ``*_bin_starts`` are optional zero-based starts for variable-width bins.
    Keeping the partition separable preserves the fast tensor evaluation
    ``B_t W B_f.T`` on the reduced likelihood grid.
    """
    power = np.asarray(power)
    time_grid = np.asarray(time_grid)
    freq_grid = np.asarray(freq_grid)
    if power.shape != (time_grid.size, freq_grid.size):
        raise ValueError("power shape must match time_grid and freq_grid.")
    if not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    mask = _validate_likelihood_mask(likelihood_mask, power.shape)

    time_starts = _validate_bin_starts(
        time_bin_starts, time_grid.size, time_bin, axis="time"
    )
    freq_starts = _validate_bin_starts(
        freq_bin_starts, freq_grid.size, freq_bin, axis="freq"
    )
    time_sizes = np.diff(np.r_[time_starts, time_grid.size])
    freq_sizes = np.diff(np.r_[freq_starts, freq_grid.size])

    power_blocks = np.add.reduceat(np.where(mask, power, 0.0), time_starts, axis=0)
    power_blocks = np.add.reduceat(power_blocks, freq_starts, axis=1)
    time_grid_blocks = np.add.reduceat(time_grid, time_starts) / time_sizes
    freq_grid_blocks = np.add.reduceat(freq_grid, freq_starts) / freq_sizes
    if likelihood_mask is None:
        counts_blocks = int(n_components) * time_sizes[:, None] * freq_sizes[None, :]
    else:
        counts_blocks = np.add.reduceat(mask.astype(int), time_starts, axis=0)
        counts_blocks = np.add.reduceat(counts_blocks, freq_starts, axis=1)
        counts_blocks *= int(n_components)
    return power_blocks, time_grid_blocks, freq_grid_blocks, counts_blocks


def _reference_scaled_power(
    power: np.ndarray,
    log_psd_offset: np.ndarray,
    likelihood_mask: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return the exact residual-likelihood power statistic ``power / R``.

    For ``S_i = R_i exp(r_b)`` with a residual ``r_b`` approximated as constant
    inside a coarse bin, the parameter-dependent quadratic term is

    ``exp(-r_b) * sum_i(power_i / R_i)``.

    The division must therefore happen at the original cell resolution before
    powers are summed.  Pooling ``log(R)`` and dividing the summed power by the
    resulting geometric-mean reference is not equivalent when ``R`` varies
    within the bin.
    """
    power = np.asarray(power, dtype=float)
    log_psd_offset = np.asarray(log_psd_offset, dtype=float)
    if power.shape != log_psd_offset.shape:
        raise ValueError("power and log_psd_offset must have matching shapes")
    mask = _validate_likelihood_mask(likelihood_mask, power.shape)
    # Mask before exponentiation so an excluded exact/near response zero cannot
    # overflow despite having zero weight in the likelihood.
    retained_power = np.where(mask, power, 0.0)
    retained_log_offset = np.where(mask, log_psd_offset, 0.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = retained_power * np.exp(-retained_log_offset)
    if not np.all(np.isfinite(scaled)):
        raise ValueError(
            "reference-scaled power must be finite; check the reference PSD scale"
        )
    return scaled


def _mean_power_for_masked_initialization(
    summed_power: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Fill masked cells for initialization without changing the target.

    Retained cells use their per-component mean power. Missing cells are filled
    by log-linear frequency interpolation within each likelihood row. A fully
    masked row receives the global retained-cell median. These values seed NUTS
    only; masked cells still have zero power and zero count in the likelihood.
    """
    summed_power = np.asarray(summed_power, dtype=float)
    counts = np.broadcast_to(np.asarray(counts, dtype=float), summed_power.shape)
    valid = (counts > 0.0) & (summed_power > 0.0)
    retained = summed_power[valid] / counts[valid]
    global_fill = float(np.median(retained)) if retained.size else 1.0
    output = np.empty_like(summed_power)
    frequency_index = np.arange(summed_power.shape[1], dtype=float)
    for row in range(summed_power.shape[0]):
        row_valid = valid[row]
        if np.any(row_valid):
            locations = frequency_index[row_valid]
            values = np.log(summed_power[row, row_valid] / counts[row, row_valid])
            output[row] = np.exp(np.interp(frequency_index, locations, values))
        else:
            output[row] = global_fill
    return output


def adaptive_frequency_bin_starts(
    pilot_log_psd: np.ndarray,
    *,
    max_log_range: float = 0.15,
    max_bin: int = 32,
) -> np.ndarray:
    """Greedily choose shared frequency bins from a pilot log-PSD surface.

    A proposed bin is extended while its log-PSD range is no larger than
    ``max_log_range`` at every pilot time and its width is below ``max_bin``.
    Sharp features therefore retain fine channels, while smooth regions use
    wider bins. The returned starts define a common nonuniform frequency grid,
    preserving tensor-product likelihood evaluation.
    """
    pilot = np.asarray(pilot_log_psd, dtype=float)
    if pilot.ndim != 2 or pilot.shape[0] == 0 or pilot.shape[1] == 0:
        raise ValueError("pilot_log_psd must be a non-empty (time, frequency) array.")
    if not np.isfinite(pilot).all():
        raise ValueError("pilot_log_psd must contain only finite values.")
    if not np.isfinite(max_log_range) or max_log_range <= 0:
        raise ValueError("max_log_range must be finite and positive.")
    if (
        not isinstance(max_bin, (int, np.integer))
        or isinstance(max_bin, bool)
        or max_bin < 1
    ):
        raise ValueError("max_bin must be a positive integer.")

    starts = [0]
    start = 0
    low = pilot[:, 0].copy()
    high = low.copy()
    for j in range(1, pilot.shape[1]):
        candidate_low = np.minimum(low, pilot[:, j])
        candidate_high = np.maximum(high, pilot[:, j])
        too_wide = j - start >= max_bin
        too_variable = float(np.max(candidate_high - candidate_low)) > max_log_range
        if too_wide or too_variable:
            starts.append(j)
            start = j
            low = pilot[:, j].copy()
            high = low.copy()
        else:
            low = candidate_low
            high = candidate_high
    return np.asarray(starts, dtype=int)


def gap_aware_time_bin_starts(
    time_grid: np.ndarray,
    time_bin: int,
    *,
    max_gap: float | None = None,
    gap_factor: float = 1.5,
) -> np.ndarray:
    """Return uniform-width time-bin starts without crossing missing intervals.

    Consecutive retained rows are not necessarily consecutive in physical time:
    after rows affected by a data gap are removed, blindly grouping array rows
    can join cells on opposite sides of that gap.  This helper splits the grid
    into contiguous runs first, then partitions each run independently.  Ragged
    bins are therefore allowed immediately before every gap and at the end.

    If ``max_gap`` is omitted, a break is any step larger than
    ``gap_factor * median(diff(time_grid))``.  Supplying ``max_gap`` is useful
    when the nominal cadence is known exactly.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    if time_grid.ndim != 1 or time_grid.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")
    if not np.isfinite(time_grid).all() or np.any(np.diff(time_grid) <= 0):
        raise ValueError("time_grid must be finite and strictly increasing.")
    if (
        not isinstance(time_bin, (int, np.integer))
        or isinstance(time_bin, bool)
        or time_bin < 1
    ):
        raise ValueError("time_bin must be a positive integer.")
    if time_grid.size == 1:
        return np.array([0], dtype=int)

    steps = np.diff(time_grid)
    if max_gap is None:
        if not np.isfinite(gap_factor) or gap_factor <= 1.0:
            raise ValueError("gap_factor must be finite and larger than 1.")
        max_gap = float(gap_factor * np.median(steps))
    elif not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("max_gap must be finite and positive.")

    breaks = np.flatnonzero(steps > max_gap) + 1
    run_starts = np.r_[0, breaks]
    run_stops = np.r_[breaks, time_grid.size]
    starts = [
        start
        for run_start, run_stop in zip(run_starts, run_stops)
        for start in range(int(run_start), int(run_stop), int(time_bin))
    ]
    return np.asarray(starts, dtype=int)


def bin_power_time_axis(
    power: np.ndarray,
    time_grid: np.ndarray,
    time_bin: int,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block-sum power/time along the time axis for likelihood coarse-graining.

    The last block is ragged when ``time_grid.size`` is not a multiple of
    ``time_bin``. Block time coordinates are the block mean of ``time_grid``.

    Args:
        power: Summed squared power per cell, shape ``(n_time, n_freq)``.
        time_grid: Time coordinates, shape ``(n_time,)``.
        time_bin: Number of consecutive time bins per block (``>= 1``).
        n_components: Real components per cell (``R``), used to scale counts.

    Returns:
        ``(power_blocks, time_grid_blocks, counts_blocks)`` with
        ``power_blocks``/``time_grid_blocks`` shape ``(n_blocks, ...)`` and
        ``counts_blocks`` shape ``(n_blocks, 1)`` (``= block_size * R``,
        summing to ``R * n_time`` over all blocks).
    """
    power_blocks, time_grid_blocks, _, counts_blocks = bin_power_rectangular(
        power,
        time_grid,
        np.arange(power.shape[1], dtype=float),
        n_components,
        time_bin=time_bin,
    )
    return power_blocks, time_grid_blocks, counts_blocks[:, :1]
