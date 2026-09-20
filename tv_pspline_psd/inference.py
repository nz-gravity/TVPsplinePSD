"""Time-frequency log-P-spline PSD inference.

The estimator is representation-agnostic: :func:`fit_log_pspline_surface` fits a
smooth ``log S(t, f)`` surface to an array of real time-frequency coefficients
``c ~ N(0, S)``. Front ends (WDM, STFT, ...) only differ in the transform that
turns a time series into ``(time_grid, freq_grid, coeffs)``. A WDM cell carries
one real coefficient (``R = 1``); an STFT cell carries two (real and imaginary).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from wdm_transform import TimeSeries

from ._surface_sampling import prepare_surface_model, run_surface_nuts
from ._surface_setup import (  # noqa: F401 - retain historical inference imports
    _prepare_spline_bases,
    _validate_explicit_interior_knots,
    prepare_likelihood_grid,
    prepare_surface_basis,
    validate_surface_inputs,
)
from .binning import (  # noqa: F401 - retain historical inference imports
    _mean_power_for_masked_initialization,
    _reference_scaled_power,
    _regular_bin_starts,
    _validate_bin_starts,
    _validate_likelihood_mask,
    adaptive_frequency_bin_starts,
    bin_power_rectangular,
    bin_power_time_axis,
    gap_aware_time_bin_starts,
)
from .config import PSplineConfig
from .posterior import (  # noqa: F401 - retain historical inference imports
    _summary_frequency_chunk,
    nested_surface_summaries,
    reconstruct_eig_coeff_samples,
    summarize_surface_samples,
    surface_summaries,
)
from .provenance import binning_provenance, provenance
from .splines import (
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
    time_bin: int = 1,
    freq_bin: int = 1,
    time_bin_starts: np.ndarray | None = None,
    freq_bin_starts: np.ndarray | None = None,
    binning_metadata: Mapping[str, Any] | None = None,
    likelihood_mask: np.ndarray | None = None,
    log_psd_offset: np.ndarray | None = None,
    residual_structure: str = "tensor",
    interaction_scale_prior: float = 0.5,
    interaction_time_knots: int = 5,
    initial_state: Any | None = None,
    likelihood_beta: float = 1.0,
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
        time_bin: Number of consecutive time bins to coarse-grain the *likelihood*
            over (the last block may be ragged). Exact given block-constant ``S``:
            each block's power is a ``chi^2`` sum of the same form as the
            per-cell likelihood with the component count scaled by the block
            size (see :func:`tv_pspline_psd.model.pspline_surface_model`). The
            approximation error is controlled by the surface's within-block
            variation and should be checked against an unbinned fit. Surface
            summaries/results are still reported on the full (unbinned)
            ``time_grid``; only the likelihood evaluation grid shrinks by
            ``~time_bin``. Default 1 (no binning).
        freq_bin: Number of consecutive frequency channels per likelihood bin.
            Uses the same summed-power/count construction as ``time_bin``.
        time_bin_starts: Optional zero-based starts for a variable-width time
            partition.  This is intended for gap-aware pooling; use
            :func:`gap_aware_time_bin_starts` after removing affected WDM rows.
            When supplied, ``time_bin`` must remain 1.
        freq_bin_starts: Optional zero-based starts for variable-width frequency
            bins, typically from :func:`adaptive_frequency_bin_starts`. When
            supplied, ``freq_bin`` must remain 1.
        binning_metadata: Optional JSON-serializable description of how a
            variable partition was selected (for example the pilot smoother,
            tolerance, and maximum width). The realised starts and widths are
            always recorded automatically in provenance.
        likelihood_mask: Optional boolean array with shape ``(n_time, n_freq)``.
            ``False`` cells contribute neither squared power nor the ``log S``
            normalization to the likelihood. The spline is still evaluated on
            the complete grid, so callers must retain this mask and treat the
            corresponding surface values as prior-driven interpolation rather
            than recovered PSD estimates.
        log_psd_offset: Optional fixed log-PSD surface with shape
            ``(n_time, n_freq)`` in the same units as ``coeffs**2``. The spline
            then models a free log-multiplicative residual around this reference.
            This is useful for placing known transfer-function structure in the
            mean without fixing the unknown PSD level or smooth departures.
        residual_structure: ``"tensor"`` for the original unrestricted
            tensor residual, or ``"stationary_plus_interaction"`` for an
            explicit stationary correction ``g(f)`` plus a zero-time-mean,
            shrinkable interaction ``h(t,f)``.
        interaction_scale_prior: Half-Normal scale for the log-PSD interaction
            amplitude in the nested residual model.
        interaction_time_knots: Interior-knot count for the interaction's own
            deliberately coarse time basis. This is independent of
            ``config.n_interior_knots_time`` used by the unrestricted tensor
            model.
        initial_state: Optional ``last_state`` from a previous result on the
            same model dimensions. When supplied, warmup is skipped and the
            persistent NUTS chain (positions plus adapted step size and mass
            matrix) is advanced for ``n_samples`` further transitions. This is
            the noise-block contract for blocked signal/noise samplers: the
            data (residual) may change between calls, the spline setup may not.

    Returns:
        A results dict with the posterior PSD surface and summaries, including
        the ``nuts_runtime_s`` wall-clock time.
    """
    coeffs, time_grid, freq_grid = validate_surface_inputs(
        coeffs,
        time_grid,
        freq_grid,
        config,
        likelihood_beta=likelihood_beta,
        residual_structure=residual_structure,
        interaction_scale_prior=interaction_scale_prior,
        interaction_time_knots=interaction_time_knots,
    )
    validated_time_starts = _validate_bin_starts(
        time_bin_starts, time_grid.size, time_bin, axis="time"
    )
    validated_freq_starts = _validate_bin_starts(
        freq_bin_starts, freq_grid.size, freq_bin, axis="freq"
    )
    power = np.sum(coeffs**2, axis=0)  # summed squared components per cell
    mask_applied = likelihood_mask is not None
    validated_likelihood_mask = _validate_likelihood_mask(likelihood_mask, power.shape)
    if log_psd_offset is None:
        validated_log_offset = np.zeros_like(power)
        offset_applied = False
    else:
        validated_log_offset = np.asarray(log_psd_offset, dtype=float)
        if validated_log_offset.shape != power.shape:
            raise ValueError("log_psd_offset must match the (time, frequency) grid")
        if not np.isfinite(validated_log_offset).all():
            raise ValueError("log_psd_offset must contain only finite values")
        offset_applied = True
    nested = residual_structure == "stationary_plus_interaction"
    basis = prepare_surface_basis(
        power,
        time_grid,
        freq_grid,
        config,
        n_components=coeffs.shape[0],
        likelihood_mask=validated_likelihood_mask if mask_applied else None,
        interior_knots_time=interior_knots_time,
        interior_knots_freq=interior_knots_freq,
        nested=nested,
        interaction_time_knots=interaction_time_knots,
    )
    grid = prepare_likelihood_grid(
        power,
        time_grid,
        freq_grid,
        basis,
        config,
        n_components=coeffs.shape[0],
        likelihood_mask=validated_likelihood_mask if mask_applied else None,
        validated_log_offset=validated_log_offset,
        offset_applied=offset_applied,
        time_bin=time_bin,
        freq_bin=freq_bin,
        time_bin_starts=validated_time_starts if time_bin_starts is not None else None,
        freq_bin_starts=validated_freq_starts if freq_bin_starts is not None else None,
    )
    model, model_args, init_sites = prepare_surface_model(
        grid,
        basis,
        config,
        mask_applied=mask_applied,
        interaction_scale_prior=interaction_scale_prior,
        likelihood_beta=likelihood_beta,
    )
    mcmc, nuts_runtime_s = run_surface_nuts(
        model,
        model_args,
        init_sites,
        n_warmup=n_warmup,
        n_samples=n_samples,
        num_chains=num_chains,
        random_seed=random_seed,
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob,
        progress_bar=progress_bar,
        initial_state=initial_state,
    )

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    W_mean, log_mean, log_lower, log_upper, log_last = summarize_surface_samples(
        samples,
        basis.interaction.whitened if nested else basis.whitened,
        config,
        basis.interaction.basis_time if nested else basis.basis_time,
        basis.interaction.basis_freq if nested else basis.basis_freq,
        validated_log_offset,
        nested=nested,
    )

    fit_provenance = provenance(
        seed=random_seed,
        config=config,
        source_data={"shape": list(coeffs.shape)},
    )
    fit_provenance["knot_allocation"] = basis.spline["knot_allocation"]
    # Retain the original summary keys for readers of older artifacts while the
    # nested recipe below records the complete realised partition.
    fit_provenance.update(
        {
            "time_bin": int(time_bin),
            "freq_bin": int(freq_bin),
            "adaptive_frequency_bins": freq_bin_starts is not None,
            "likelihood_grid_shape": [int(v) for v in grid.power.shape],
        }
    )
    fit_provenance["binning"] = binning_provenance(
        n_time=time_grid.size,
        n_freq=freq_grid.size,
        time_bin=time_bin,
        freq_bin=freq_bin,
        time_bin_starts=(
            validated_time_starts if time_bin_starts is not None else None
        ),
        freq_bin_starts=(
            validated_freq_starts if freq_bin_starts is not None else None
        ),
        selector_metadata=binning_metadata,
    )
    retained_cells = int(np.count_nonzero(validated_likelihood_mask))
    total_cells = int(validated_likelihood_mask.size)
    fit_provenance["likelihood_mask"] = {
        "applied": bool(mask_applied),
        "retained_cells": retained_cells,
        "masked_cells": total_cells - retained_cells,
        "masked_fraction": float(1.0 - retained_cells / total_cells),
    }
    fit_provenance["log_psd_offset"] = {
        "applied": bool(offset_applied),
        "shape": list(validated_log_offset.shape),
        "coarse_likelihood_handling": (
            "cellwise_power_divided_by_reference_before_block_sum"
            if offset_applied and grid.coarse_grained
            else "cellwise_offset_on_likelihood_grid"
            if offset_applied
            else None
        ),
        "data_only_reference_log_determinant_omitted": bool(
            offset_applied and grid.coarse_grained
        ),
    }
    fit_provenance["residual_structure"] = {
        "name": residual_structure,
        "interaction_scale_prior": (float(interaction_scale_prior) if nested else None),
        "interaction_time_knots": int(interaction_time_knots) if nested else None,
        "interaction_time_basis_size": (
            int(basis.interaction.B_time.shape[1]) if nested else None
        ),
        "interaction_zero_time_mean": bool(nested),
    }
    return {
        "mcmc": mcmc,
        "config": config,
        "coeffs": coeffs,
        "power": power,
        "likelihood_mask": validated_likelihood_mask,
        "log_psd_offset": validated_log_offset,
        "time_grid": np.asarray(time_grid),
        "freq_grid": np.asarray(freq_grid),
        "knots_time": basis.spline["knots_time"],
        "knots_freq": basis.spline["knots_freq_unit"],
        # ``knots_freq`` is retained in its historical normalized coordinate
        # for saved-run compatibility. The explicit physical vectors remove
        # ambiguity for callers selecting knots in Hz.
        "knots_time_physical": basis.spline["knots_time_physical"],
        "knots_freq_physical": basis.spline["knots_freq_physical"],
        "knots_freq_unit": basis.spline["knots_freq_unit"],
        "knot_allocation": basis.spline["knot_allocation"],
        "B_time": basis.spline["B_time"],
        "B_freq": basis.spline["B_freq"],
        "B_time_interaction": basis.interaction.B_time if nested else None,
        "basis_interaction_time": basis.interaction.basis_time if nested else None,
        "basis_nested_freq": basis.interaction.basis_freq if nested else None,
        "whitened": basis.interaction.whitened if nested else basis.whitened,
        "samples": samples,
        "W_mean": W_mean,
        "residual_structure": residual_structure,
        "interaction_scale_samples": (
            samples.get("sigma_interaction") if nested else None
        ),
        "log_psd_mean": log_mean,
        "log_psd_lower": log_lower,
        "log_psd_upper": log_upper,
        # Geometric posterior mean exp(E[log S]); ``psd_mean`` is a deprecated
        # compatibility alias retained for one release.
        "psd_geometric_mean": np.exp(log_mean),
        "psd_last_draw": np.exp(log_last),
        "last_state": mcmc.last_state,
        "psd_mean": np.exp(log_mean),
        "psd_lower": np.exp(log_lower),
        "psd_upper": np.exp(log_upper),
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
        "nuts_runtime_s": float(nuts_runtime_s),
        "time_bin": int(time_bin),
        "freq_bin": int(freq_bin),
        "freq_bin_starts": (
            None if freq_bin_starts is None else validated_freq_starts.copy()
        ),
        "likelihood_grid_shape": tuple(int(v) for v in grid.power.shape),
        "provenance": fit_provenance,
    }


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
    if (
        not isinstance(nt, (int, np.integer))
        or isinstance(nt, (bool, np.bool_))
        or nt <= 0
    ):
        raise ValueError("nt must be a positive integer.")
    n_total = data.size
    if n_total % nt != 0:
        raise ValueError(
            f"WDM sizing requires N ({n_total}) to be divisible by nt ({nt})."
        )
    nf = n_total // nt
    if nt % 2 != 0 or nf % 2 != 0:
        raise ValueError(
            f"WDM sizing requires both nt ({nt}) and nf=N/nt ({nf}) to be even."
        )

    wdm = TimeSeries(data, dt=dt).to_wdm(nt=nt)
    return _trimmed_wdm_analysis_grid(wdm, config)


def _trimmed_wdm_analysis_grid(
    wdm, config: PSplineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim a WDM object to the analysis grid shared by every front end."""
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


def wdm_analysis_coefficients_from_fd(
    fd_data: np.ndarray, dt: float, nt: int, config: PSplineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WDM-transform a one-sided FD series directly, skipping the time domain.

    ``fd_data`` holds ``n // 2 + 1`` complex samples on the full
    ``rfftfreq(n, dt)`` grid in the continuous-FT convention
    ``h(f) = dt * rfft(x)`` (the BBHx/lisatools convention). The result is
    identical to ``wdm_analysis_coefficients(irfft(fd_data / dt), ...)``
    without the inverse/forward FFT round trip, and uses the same trimming, so
    FD templates and time-domain data land on one analysis grid.
    """
    from wdm_transform import FrequencySeries

    fd_data = np.asarray(fd_data)
    if fd_data.ndim != 1:
        raise ValueError("FD input data must be one-dimensional.")
    if fd_data.size < 2:
        raise ValueError("FD input data must cover a non-trivial rfft grid.")
    if not np.isfinite(fd_data).all():
        raise ValueError("FD input data must contain only finite values.")
    if dt <= 0:
        raise ValueError("dt must be strictly positive.")
    n_total = 2 * (fd_data.size - 1)
    half = np.asarray(fd_data) / dt
    # The DC and Nyquist bins of a real signal's spectrum are real.
    full = np.empty(n_total, dtype=complex)
    full[0] = half[0].real
    full[n_total // 2] = half[-1].real
    full[1 : n_total // 2] = half[1:-1]
    full[n_total // 2 + 1 :] = np.conj(half[1:-1][::-1])
    wdm = FrequencySeries(full, df=1.0 / (n_total * dt)).to_wdm(nt=nt)
    return _trimmed_wdm_analysis_grid(wdm, config)


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
    results["provenance"].update(
        {
            "dt": float(dt),
            "nt": int(nt),
            "trims": {
                "time_bins": config.trim_time_bins,
                "low_freq_channels": config.trim_low_freq_channels,
                "high_freq_channels": config.trim_high_freq_channels,
            },
            "source_data": {"shape": list(np.asarray(data).shape)},
        }
    )
    return results


def evaluate_dense_posterior_mean(
    results: dict[str, object],
    *,
    n_time_dense: int = 200,
    n_freq_dense: int = 200,
) -> dict[str, np.ndarray]:
    """Evaluate a tensor posterior mean without an offset on a dense grid.

    Offset and nested residual fits require their native-grid summaries.
    """
    if results.get("residual_structure", "tensor") != "tensor" or np.any(
        results.get("log_psd_offset", 0.0)
    ):
        raise ValueError(
            "Dense reconstruction requires a tensor model without log_psd_offset. "
            "Use the stored analysis grid for offset or stationary-plus-interaction fits."
        )
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
