"""Time-frequency log-P-spline PSD inference.

The estimator is representation-agnostic: :func:`fit_log_pspline_surface` fits a
smooth ``log S(t, f)`` surface to an array of real time-frequency coefficients
``c ~ N(0, S)``. Front ends (WDM, STFT, ...) only differ in the transform that
turns a time series into ``(time_grid, freq_grid, coeffs)``. A WDM cell carries
one real coefficient (``R = 1``); an STFT cell carries two (real and imaginary).
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value
from wdm_transform import TimeSeries

from .adaptive_knots import fit_adaptive_knots
from .config import PSplineConfig
from .model import (
    initialize_with_penalized_least_squares,
    pspline_surface_model,
    whiten_penalty_pair,
    whitened_init_values,
)
from .provenance import provenance
from .splines import (
    create_bspline_basis,
    create_bspline_roughness_penalty,
    evaluate_bspline_basis,
)


def fit_log_pspline_surface(
    coeffs: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    *,
    config: PSplineConfig,
    interior_knots_time: np.ndarray | None = None,
    interior_knots_freq: np.ndarray | None = None,
    n_warmup: int = 250,
    n_samples: int = 300,
    num_chains: int = 1,
    random_seed: int = 7,
    max_tree_depth: int = 10,
    target_accept_prob: float = 0.85,
    progress_bar: bool = True,
) -> dict[str, object]:
    """Fit a smooth ``log S(t, f)`` surface to real time-frequency coefficients.

    The per-sample ``log S`` surface is never stored: only the tiny posterior
    sites (``s``, ``phi_time``, ``phi_freq``) are kept, and all surface summaries
    are reconstructed from the eigen-coefficients in frequency chunks. This keeps
    the result (and any saved artifact, see :mod:`tv_pspline_psd.io`) small while
    letting the full surface be regenerated on demand.

    Args:
        coeffs: Real coefficients of shape ``(R, n_time, n_freq)`` (``R`` real
            components per cell), already trimmed to the analysis grid.
        time_grid: Rescaled time coordinates in ``[0, 1]``, shape ``(n_time,)``.
        freq_grid: Frequencies (Hz) of each channel, shape ``(n_freq,)``.
        config: Estimator configuration. Time knots are linear; frequency
            placement follows ``config.freq_knot_strategy`` unless overridden.
        interior_knots_time: Optional explicit interior time knots, in the same
            coordinates as ``time_grid``. These override linear time knots.
        interior_knots_freq: Optional explicit interior frequency knots in Hz.
            The fit still uses an internally normalized frequency coordinate.
        progress_bar: Show the NUTS progress bar. Set False for quiet batch runs.

    Returns:
        A results dict with the posterior PSD surface and summaries, including
        the ``nuts_runtime_s`` wall-clock time.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    freq_grid = np.asarray(freq_grid, dtype=float)
    if coeffs.ndim != 3:
        raise ValueError("coeffs must have shape (R, n_time, n_freq).")
    if coeffs.shape[0] == 0 or coeffs.shape[1] == 0 or coeffs.shape[2] == 0:
        raise ValueError("coeffs and the analysis grid must be non-empty after trimming.")
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
    power = np.sum(coeffs**2, axis=0)  # summed squared components per cell
    spline = _prepare_spline_bases(
        power,
        time_grid,
        freq_grid,
        config,
        n_components=coeffs.shape[0],
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

    # The warm start fits log S to the per-component mean power, matching the
    # likelihood mode (S = mean of squared components).
    pls_init = initialize_with_penalized_least_squares(
        power / coeffs.shape[0], B_time, B_freq, P_time, P_freq, config
    )
    init_sites = whitened_init_values(pls_init, whitened, config)

    model_args = (
        jnp.asarray(coeffs),
        jnp.asarray(basis_eig_time),
        jnp.asarray(basis_eig_freq),
        jnp.asarray(whitened["lam_time"]),
        jnp.asarray(whitened["lam_freq"]),
        jnp.asarray(whitened["joint_null"]),
        config,
        False,  # never store the per-sample log_psd surface; reconstruct instead
    )
    kernel = NUTS(
        pspline_surface_model,
        init_strategy=init_to_value(values=init_sites),
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob,
    )
    mcmc = MCMC(
        kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=num_chains,
        chain_method="sequential", progress_bar=progress_bar,
    )
    nuts_t0 = time.perf_counter()
    mcmc.run(
        random.PRNGKey(random_seed), *model_args,
        extra_fields=("diverging", "accept_prob", "num_steps", "potential_energy"),
    )
    nuts_runtime_s = time.perf_counter() - nuts_t0

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    eig_samples = reconstruct_eig_coeff_samples(samples, whitened, config)
    W_mean = whitened["U_time"] @ eig_samples.mean(axis=0) @ whitened["U_freq"].T
    log_mean, log_lower, log_upper = surface_summaries(
        eig_samples, basis_eig_time, basis_eig_freq,
        precomputed=samples.get("log_psd"),
    )

    fit_provenance = provenance(
        seed=random_seed,
        config=config,
        source_data={"shape": list(coeffs.shape)},
    )
    fit_provenance["knot_allocation"] = spline["knot_allocation"]
    return {
        "mcmc": mcmc,
        "config": config,
        "coeffs": coeffs,
        "power": power,
        "time_grid": np.asarray(time_grid),
        "freq_grid": np.asarray(freq_grid),
        "knots_time": knots_time,
        "knots_freq": knots_freq,
        # ``knots_freq`` is retained in its historical normalized coordinate
        # for saved-run compatibility. The explicit physical vectors remove
        # ambiguity for callers selecting knots in Hz.
        "knots_time_physical": spline["knots_time_physical"],
        "knots_freq_physical": spline["knots_freq_physical"],
        "knots_freq_unit": knots_freq,
        "knot_allocation": spline["knot_allocation"],
        "B_time": B_time,
        "B_freq": B_freq,
        "whitened": whitened,
        "samples": samples,
        "W_mean": W_mean,
        "log_psd_mean": log_mean,
        "log_psd_lower": log_lower,
        "log_psd_upper": log_upper,
        # Geometric posterior mean exp(E[log S]); ``psd_mean`` is a deprecated
        # compatibility alias retained for one release.
        "psd_geometric_mean": np.exp(log_mean),
        "psd_mean": np.exp(log_mean),
        "psd_lower": np.exp(log_lower),
        "psd_upper": np.exp(log_upper),
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
        "nuts_runtime_s": float(nuts_runtime_s),
        "provenance": fit_provenance,
    }


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
        "frequency": "explicit" if explicit_freq is not None else config.freq_knot_strategy,
    }
    if selected_freq is None:
        if config.freq_knot_strategy == "adaptive":
            pilot = fit_adaptive_knots(
                power,
                time_grid,
                freq_grid,
                counts=float(n_components),
                n_pilot_knots_time=max(8, config.n_interior_knots_time),
                n_pilot_knots_freq=max(16, config.n_interior_knots_freq),
                n_knots_time=config.n_interior_knots_time,
                n_knots_freq=config.n_interior_knots_freq,
                method="curvature",
            )
            selected_freq = pilot.freq_knots
        elif config.freq_knot_strategy == "log":
            if freq_grid[0] <= 0:
                raise ValueError("freq_knot_strategy='log' requires a strictly positive frequency grid.")
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


def reconstruct_eig_coeff_samples(
    samples: dict[str, np.ndarray],
    whitened: dict[str, np.ndarray],
    config: PSplineConfig,
) -> np.ndarray:
    """Per-sample eigen-coefficients ``Z`` of shape ``(n_samples, K_t, K_f)``.

    These are tiny (``K_t K_f`` numbers per sample) and fully determine the
    surface, so summaries can be reconstructed without storing it per sample.
    """
    lam_t = whitened["lam_time"]
    lam_f = whitened["lam_freq"]
    joint_null = whitened["joint_null"]
    n_t, n_f = lam_t.size, lam_f.size

    s = samples["s"].reshape(-1, n_t, n_f)
    if config.centered:
        return s
    phi_time = np.exp(samples["phi_time"])[:, None, None]  # the site stores log phi
    phi_freq = np.exp(samples["phi_freq"])[:, None, None]
    d = phi_time * lam_t[None, :, None] + phi_freq * lam_f[None, None, :]
    scale = np.where(
        joint_null[None],
        1.0 / np.sqrt(config.null_precision),
        1.0 / np.sqrt(d + config.ridge_eps),
    )
    return s * scale


def surface_summaries(
    eig_samples: np.ndarray,
    basis_eig_time: np.ndarray,
    basis_eig_freq: np.ndarray,
    *,
    precomputed: np.ndarray | None = None,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
    freq_chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean and central interval of ``log S`` on the analysis grid.

    The mean is reconstructed from the mean eigen-coefficients (exact, since the
    surface is linear in them); the interval is reconstructed in frequency chunks
    to bound peak memory. If ``precomputed`` (the stored per-sample surface) is
    given, it is used directly.
    """
    log_mean = basis_eig_time @ eig_samples.mean(axis=0) @ basis_eig_freq.T
    if precomputed is not None:
        return (log_mean,
                np.percentile(precomputed, lower_pct, axis=0),
                np.percentile(precomputed, upper_pct, axis=0))

    n_t = basis_eig_time.shape[0]
    n_f = basis_eig_freq.shape[0]
    lower = np.empty((n_t, n_f))
    upper = np.empty((n_t, n_f))
    for j0 in range(0, n_f, freq_chunk):
        bf = basis_eig_freq[j0:j0 + freq_chunk]
        # optimize=True factorises the 3-operand contraction into two BLAS
        # matmuls; without it numpy falls back to a naive element-wise kernel
        # that scales catastrophically on large (time x freq) grids.
        chunk = np.einsum("ta,nab,jb->ntj", basis_eig_time, eig_samples, bf,
                          optimize=True)
        lower[:, j0:j0 + freq_chunk] = np.percentile(chunk, lower_pct, axis=0)
        upper[:, j0:j0 + freq_chunk] = np.percentile(chunk, upper_pct, axis=0)
    return log_mean, lower, upper


def _wdm_coeffs_2d(wdm) -> np.ndarray:
    """Return WDM coefficients as a 2D ``(nt, nf + 1)`` array (squeezing batch)."""
    coeffs = np.asarray(wdm.coeffs)
    if coeffs.ndim == 3:
        if coeffs.shape[0] != 1:
            raise ValueError("Expected a single WDM series (batch size 1).")
        coeffs = coeffs[0]
    return coeffs


def wdm_analysis_coefficients(
    data: np.ndarray, dt: float, nt: int, config: PSplineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WDM-transform a series and trim to the analysis grid.

    Returns ``(coeffs, time_grid, freq_grid)`` with ``coeffs`` of shape
    ``(n_time, n_freq)``. Using this for both the data and any signal templates
    guarantees they share the same trimmed grid.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("WDM input data must be one-dimensional.")
    if data.size == 0:
        raise ValueError("WDM input data must be non-empty.")
    if dt <= 0:
        raise ValueError("dt must be strictly positive.")
    if not isinstance(nt, (int, np.integer)) or isinstance(nt, (bool, np.bool_)) or nt <= 0:
        raise ValueError("nt must be a positive integer.")
    n_total = data.size
    if n_total % nt != 0:
        raise ValueError(f"WDM sizing requires N ({n_total}) to be divisible by nt ({nt}).")
    nf = n_total // nt
    if nt % 2 != 0 or nf % 2 != 0:
        raise ValueError(
            f"WDM sizing requires both nt ({nt}) and nf=N/nt ({nf}) to be even."
        )

    wdm = TimeSeries(data, dt=dt).to_wdm(nt=nt)
    coeffs = _wdm_coeffs_2d(wdm)
    keep_time = np.arange(config.trim_time_bins, wdm.nt - config.trim_time_bins)
    keep_freq = np.arange(
        config.trim_low_freq_channels, wdm.nf + 1 - config.trim_high_freq_channels
    )
    if keep_time.size == 0 or keep_freq.size == 0:
        raise ValueError("WDM trimming leaves an empty time or frequency grid.")
    time_grid = np.asarray(wdm.time_grid)[keep_time] / wdm.duration
    freq_grid = np.asarray(wdm.freq_grid)[keep_freq]
    return coeffs[np.ix_(keep_time, keep_freq)], time_grid, freq_grid


def run_wdm_psd_mcmc(
    data: np.ndarray,
    *,
    dt: float,
    nt: int,
    config: PSplineConfig,
    **fit_kwargs,
) -> dict[str, object]:
    """WDM front end: transform to WDM coefficients, then fit the surface."""
    coeffs_fit, time_grid, freq_grid = wdm_analysis_coefficients(data, dt, nt, config)
    results = fit_log_pspline_surface(
        coeffs_fit[None, :, :], time_grid, freq_grid, config=config, **fit_kwargs
    )
    results.update({"coeffs_fit": coeffs_fit})
    results["provenance"].update({
        "dt": float(dt),
        "nt": int(nt),
        "trims": {
            "time_bins": config.trim_time_bins,
            "low_freq_channels": config.trim_low_freq_channels,
            "high_freq_channels": config.trim_high_freq_channels,
        },
        "source_data": {"shape": list(np.asarray(data).shape)},
    })
    return results


def evaluate_dense_posterior_mean(
    results: dict[str, object],
    *,
    n_time_dense: int = 200,
    n_freq_dense: int = 200,
) -> dict[str, np.ndarray]:
    """Evaluate the posterior-mean spline surface on a dense plotting grid."""
    config: PSplineConfig = results["config"]  # type: ignore[assignment]
    time_grid = results["time_grid"]
    freq_grid = results["freq_grid"]

    dense_time = np.linspace(time_grid[0], time_grid[-1], n_time_dense)
    dense_freq = np.linspace(freq_grid[0], freq_grid[-1], n_freq_dense)
    dense_freq_unit = dense_freq / np.maximum(freq_grid[-1], 1e-12)

    B_time_dense = evaluate_bspline_basis(
        dense_time, results["knots_time"], degree=config.degree_time
    )
    B_freq_dense = evaluate_bspline_basis(
        dense_freq_unit, results["knots_freq"], degree=config.degree_freq
    )
    dense_log_psd = B_time_dense @ results["W_mean"] @ B_freq_dense.T
    return {
        "time_grid": dense_time,
        "freq_grid": dense_freq,
        "log_psd_mean": dense_log_psd,
        "psd_geometric_mean": np.exp(dense_log_psd),
        "psd_mean": np.exp(dense_log_psd),
    }
