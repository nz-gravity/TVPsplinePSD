"""Tang-style zigzag moving periodogram + thinned dynamic Whittle.

Faithful to the construction of Tang et al. (the ``beyondWhittle`` dynamic
Whittle): a moving periodogram ordinate is formed from a centred ``2m+1``-point
window and evaluated at a single Fourier frequency ``lambda_{mod(t)}`` that
*cycles (zigzags)* with the time index,

    MI_t = |sum_{nu=0}^{2m} X_{nu+t-m} exp(-i pi nu lambda_{mod(t)})|^2
           / (2 pi (2m+1)),   lambda_j = 2j/(2m+1),  mod(t) = 1 + ((t-1) mod m).

The thinned variant keeps blocks of ``m`` ordinates (one per frequency) spaced by
``i*m`` to reduce correlation. The resulting *scattered* ``(u, omega)`` ordinates
use the same power/count Whittle likelihood and whitened tensor-product P-spline
prior as WDM. A Tang ordinate is represented by ``summed_power=2*MI`` and
``counts=2``; optional binning sums those statistics after thinning. Only the
time-frequency representation and its surface evaluator differ.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
import numpyro
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value

from .config import PSplineConfig
from .model import (
    power_floor,
    power_whittle_log_likelihood,
    sample_tensor_eigen_coefficients,
    whiten_penalty_pair,
    whitened_init_values,
)
from .provenance import binning_provenance, provenance
from .splines import (
    create_bspline_basis,
    create_bspline_roughness_penalty,
    evaluate_bspline_basis,
)


def tang_moving_periodogram(
    data: np.ndarray, *, m: int, thin: int = 2
) -> dict[str, np.ndarray]:
    """Thinned zigzag moving-periodogram ordinates (Tang et al.).

    Args:
        data: Real time series of length ``T``.
        m: Order (window half-width; window length ``2m+1``, ``m`` frequencies).
        thin: Positive thinning factor ``i``; blocks are spaced by ``i*m``.
            Tang-style fits typically use 2 or 3.

    Returns:
        Dict with scattered arrays ``u`` (rescaled time in ``(0,1)``), ``omega``
        (angular frequency in ``(0, pi)``), ``coeff`` (normalised complex moving
        Fourier coefficients), and ``mi = abs(coeff)**2``.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError("Moving-periodogram input data must be one-dimensional.")
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
        raise ValueError("m must be a positive integer.")
    if not isinstance(thin, (int, np.integer)) or isinstance(thin, bool) or thin < 1:
        raise ValueError("thin must be a positive integer.")
    T = x.size
    n_blocks = (T - 2 * m) // (thin * m)
    if n_blocks < 1:
        raise ValueError("Series too short for these (m, thin).")

    nu = np.arange(2 * m + 1)
    j = np.arange(1, m + 1)
    lam = 2.0 * j / (2 * m + 1)        # Fourier frequencies in (0, 1)
    omega = np.pi * lam               # angular frequency in (0, pi)
    phase = np.exp(-1j * np.pi * np.outer(nu, lam))  # (2m+1, m)

    # ``windows[k]`` is the length-(2m+1) window whose 1-based centre is
    # ``t = k + m + 1``.  The retained window starts form an
    # ``(n_blocks, m)`` array, preserving the original block-major ordering.
    windows = np.lib.stride_tricks.sliding_window_view(x, 2 * m + 1)
    window_starts = (
        thin * m * np.arange(n_blocks)[:, None] + np.arange(m)[None, :]
    ).reshape(-1)
    freq_index = np.tile(np.arange(m), n_blocks)

    # Advanced indexing materializes the selected windows and phase rows, so
    # process bounded chunks rather than allocating an O(T*m) temporary.  The
    # target accounts for one float window and one complex phase row per point.
    bytes_per_point = (2 * m + 1) * (x.dtype.itemsize + phase.dtype.itemsize)
    chunk_points = max(1, (64 * 1024**2) // max(bytes_per_point, 1))
    coeff = np.empty(window_starts.size, dtype=np.complex128)
    phase_by_freq = phase.T
    for start in range(0, window_starts.size, chunk_points):
        stop = min(start + chunk_points, window_starts.size)
        selected_windows = windows[window_starts[start:stop]]
        selected_phase = phase_by_freq[freq_index[start:stop]]
        coeff[start:stop] = np.einsum(
            "pn,pn->p", selected_windows, selected_phase, optimize=True
        )

    t = window_starts + m + 1
    coeff /= np.sqrt(2.0 * np.pi * (2 * m + 1))
    return {
        "u": t / T,
        "omega": np.tile(omega, n_blocks),
        "coeff": coeff,
        "mi": np.abs(coeff) ** 2,
    }


def bin_tang_ordinates(
    ordinates: dict[str, np.ndarray],
    *,
    time_bin: int = 1,
    freq_bin: int = 1,
    freq_bin_starts: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Pool already-thinned Tang ordinates into power/count observations.

    The returned scattered arrays remain block-major with the same frequency
    rungs repeated for every time block. Under the dynamic-Whittle independence
    approximation, each raw ordinate has ``summed_power=2*MI`` and ``counts=2``.
    Separable bins sum both statistics and use the mean ordinate location.
    Ragged final bins are retained. ``freq_bin_starts`` optionally supplies
    zero-based starts for variable-width frequency bins and requires
    ``freq_bin=1``.

    This does *not* make the unthinned overlapping moving periodogram
    independent. ``thin`` remains the dependence-control step; this function is
    only a likelihood coarse-graining step applied afterwards.
    """
    if (
        not isinstance(time_bin, (int, np.integer))
        or isinstance(time_bin, bool)
        or time_bin < 1
    ):
        raise ValueError("time_bin must be a positive integer.")
    if (
        not isinstance(freq_bin, (int, np.integer))
        or isinstance(freq_bin, bool)
        or freq_bin < 1
    ):
        raise ValueError("freq_bin must be a positive integer.")

    u = np.asarray(ordinates["u"], dtype=float)
    omega = np.asarray(ordinates["omega"], dtype=float)
    mi = np.asarray(ordinates["mi"], dtype=float)
    if u.ndim != 1 or omega.ndim != 1 or mi.ndim != 1:
        raise ValueError("Tang ordinate arrays must be one-dimensional.")
    if not (u.size == omega.size == mi.size) or u.size == 0:
        raise ValueError("Tang ordinate arrays must have the same nonzero length.")
    if not np.isfinite(u).all() or not np.isfinite(omega).all():
        raise ValueError("Tang ordinate locations must be finite.")
    if not np.isfinite(mi).all() or np.any(mi < 0):
        raise ValueError("Tang moving-periodogram powers must be finite and non-negative.")

    freq_rungs = np.unique(omega)
    n_freq = freq_rungs.size
    if u.size % n_freq:
        raise ValueError("Tang ordinate count must be divisible by its frequencies.")
    n_blocks = u.size // n_freq
    omega_2d = omega.reshape(n_blocks, n_freq)
    if not np.allclose(omega_2d, freq_rungs[None, :], rtol=0.0, atol=1e-14):
        raise ValueError("Tang ordinates must use repeated block-major frequency rungs.")

    u_2d = u.reshape(n_blocks, n_freq)
    mi_2d = mi.reshape(n_blocks, n_freq)
    ti = np.arange(0, n_blocks, time_bin)
    if freq_bin_starts is None:
        fi = np.arange(0, n_freq, freq_bin)
    else:
        if freq_bin != 1:
            raise ValueError("freq_bin must be 1 when freq_bin_starts is provided.")
        fi = np.asarray(freq_bin_starts)
        if fi.ndim != 1 or fi.size == 0:
            raise ValueError("freq_bin_starts must be a non-empty one-dimensional array.")
        if not np.issubdtype(fi.dtype, np.integer):
            raise ValueError("freq_bin_starts must contain integer indices.")
        fi = fi.astype(int, copy=False)
        if fi[0] != 0 or fi[-1] >= n_freq or np.any(np.diff(fi) <= 0):
            raise ValueError(
                "freq_bin_starts must begin at 0 and contain strictly increasing "
                f"indices smaller than the frequency grid size ({n_freq})."
            )

    def block_sum(values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(np.add.reduceat(values, ti, axis=0), fi, axis=1)

    cells = block_sum(np.ones_like(mi_2d))
    summed_power = 2.0 * block_sum(mi_2d)
    counts = 2.0 * cells
    coarse_u = block_sum(u_2d) / cells
    freq_ends = np.r_[fi[1:], n_freq]
    coarse_freq = np.asarray(
        [freq_rungs[start:stop].mean() for start, stop in zip(fi, freq_ends)]
    )
    coarse_omega = np.broadcast_to(coarse_freq, coarse_u.shape).copy()
    return {
        "u": coarse_u.reshape(-1),
        "omega": coarse_omega.reshape(-1),
        "summed_power": summed_power.reshape(-1),
        "counts": counts.reshape(-1),
    }


def _grouped_normal_equations(target, B_time, B_freq_unique):
    """Return ``X.T @ X`` and ``X.T @ target`` without forming scattered ``X``.

    For frequency rung ``j``, every design row is
    ``kron(B_freq[j], B_time[p])``.  Accumulating the Kronecker products of the
    small per-rung time Grams avoids the ``P x (K_t*K_f)`` design matrix.
    """
    n_freq, n_basis_freq = B_freq_unique.shape
    n_basis_time = B_time.shape[1]
    if target.size % n_freq:
        raise ValueError("Tang ordinate count must be divisible by its frequencies.")
    time_grouped = B_time.reshape((-1, n_freq, n_basis_time))
    target_grouped = target.reshape((-1, n_freq))
    n_weights = n_basis_time * n_basis_freq
    gram = np.zeros((n_weights, n_weights))
    rhs = np.zeros(n_weights)
    for j in range(n_freq):
        bt = time_grouped[:, j, :]
        bf = B_freq_unique[j]
        gram += np.kron(np.outer(bf, bf), bt.T @ bt)
        rhs += np.kron(bf, bt.T @ target_grouped[:, j])
    return gram, rhs


def _scattered_pls_init(
    observed_power, B_time, B_freq_unique, P_time, P_freq, config
):
    """Penalized least-squares warm start on scattered log-ordinates."""
    n_t, n_f = B_time.shape[1], B_freq_unique.shape[1]
    floor = power_floor(observed_power)
    target = np.log(observed_power + floor)
    gram, rhs = _grouped_normal_equations(target, B_time, B_freq_unique)
    kron_time = np.kron(np.eye(n_f), P_time)
    kron_freq = np.kron(P_freq, np.eye(n_t))
    system = (
        gram
        + config.init_penalty_time * kron_time
        + config.init_penalty_freq * kron_freq
        + config.ridge_eps * np.eye(n_f * n_t)
    )
    w = np.linalg.solve(system, rhs)
    W = w.reshape((n_t, n_f), order="F")
    phi_time = max(1e-2, w.size / (float(w @ kron_time @ w) + 1e-6))
    phi_freq = max(1e-2, w.size / (float(w @ kron_freq @ w) + 1e-6))
    return {"W": W, "phi_time": phi_time, "phi_freq": phi_freq}


def _grouped_log_surface(basis_eig_time, basis_eig_freq_unique, eig_coeffs):
    """Evaluate the tensor surface at block-major Tang ordinates.

    Tang repeats the same ``m`` Fourier frequencies in every retained block.
    Contracting ``W`` with those unique frequency rows first changes the main
    evaluation from ``O(P*K_t*K_f)`` to ``O(m*K_t*K_f + P*K_t)``.
    """
    n_freq = basis_eig_freq_unique.shape[0]
    n_basis_time = basis_eig_time.shape[1]
    basis_time_grouped = basis_eig_time.reshape((-1, n_freq, n_basis_time))
    coeffs_by_freq = (eig_coeffs @ basis_eig_freq_unique.T).T
    return jnp.sum(
        basis_time_grouped * coeffs_by_freq[None, :, :], axis=2
    ).reshape(-1)


def _dynamic_whittle_model(
    summed_power,
    counts,
    basis_eig_time,
    basis_eig_freq_unique,
    lam_time,
    lam_freq,
    joint_null,
    config,
):
    eig_coeffs = sample_tensor_eigen_coefficients(
        basis_eig_time,
        basis_eig_freq_unique,
        lam_time,
        lam_freq,
        joint_null,
        config,
    )
    # Scattered evaluation at each ordinate's own (u_p, omega_p), grouped by
    # the m repeated frequency rungs.
    log_f = _grouped_log_surface(basis_eig_time, basis_eig_freq_unique, eig_coeffs)
    log_like = power_whittle_log_likelihood(summed_power, counts, log_f)
    numpyro.factor("dynamic_whittle", log_like)
    numpyro.deterministic("eig_coeffs", eig_coeffs)


def run_tang_dynamic_whittle_mcmc(
    data: np.ndarray,
    *,
    dt: float,
    m: int,
    thin: int = 2,
    config: PSplineConfig,
    interior_knots_time: np.ndarray | None = None,
    interior_knots_freq: np.ndarray | None = None,
    n_time_grid: int = 60,
    n_warmup: int = 250,
    n_samples: int = 300,
    num_chains: int = 1,
    random_seed: int = 7,
    time_bin: int = 1,
    freq_bin: int = 1,
    freq_bin_starts: np.ndarray | None = None,
    binning_metadata: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Fit the thinned dynamic-Whittle model and evaluate the PSD on a grid.

    ``thin`` reduces dependence between overlapping moving windows.
    ``interior_knots_time`` and ``interior_knots_freq`` optionally provide
    explicit interior knots in rescaled-time and Hz coordinates respectively.
    They are useful for controlled comparisons with another front end: both
    likelihoods can then use the same spline dimension and physical locations.
    ``time_bin`` and ``freq_bin`` subsequently pool the retained ordinates using
    the same summed-power/count likelihood used by the WDM/STFT front ends.
    ``freq_bin_starts`` supplies a variable-width frequency partition (and
    requires ``freq_bin=1``), including one obtained from the same adaptive
    pilot used for WDM. ``binning_metadata`` can record the JSON-serializable
    pilot settings; the realised partition is always retained in provenance.
    """
    ordinates = tang_moving_periodogram(data, m=m, thin=thin)
    observations = bin_tang_ordinates(
        ordinates,
        time_bin=time_bin,
        freq_bin=freq_bin,
        freq_bin_starts=freq_bin_starts,
    )
    u = observations["u"]
    omega = observations["omega"]
    summed_power = observations["summed_power"]
    counts = observations["counts"]
    freq_unit = omega / np.pi  # in (0, 1)
    freq_hz = omega / (2.0 * np.pi * dt)

    def _explicit_knots(values, grid, expected, name):
        if values is None:
            return None
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.size != expected:
            raise ValueError(
                f"{name} must be one-dimensional with exactly {expected} values."
            )
        if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} must contain finite, strictly increasing values.")
        if np.any(values <= grid.min()) or np.any(values >= grid.max()):
            raise ValueError(f"{name} must lie strictly inside the analysis-grid range.")
        return values

    explicit_time = _explicit_knots(
        interior_knots_time,
        ordinates["u"],
        config.n_interior_knots_time,
        "interior_knots_time",
    )
    explicit_freq_hz = _explicit_knots(
        interior_knots_freq,
        freq_hz,
        config.n_interior_knots_freq,
        "interior_knots_freq",
    )
    explicit_freq_unit = (
        None if explicit_freq_hz is None else 2.0 * dt * explicit_freq_hz
    )

    # Define the spline domain on the original ordinate range, then evaluate
    # the likelihood basis at the coarse cell centres. Otherwise binning would
    # inadvertently shrink the fitted domain at both boundary bins.
    _, knots_time = create_bspline_basis(
        ordinates["u"], config.n_interior_knots_time, degree=config.degree_time,
        interior_knots=explicit_time,
    )
    B_time = evaluate_bspline_basis(u, knots_time, degree=config.degree_time)
    unique_freq_unit = np.unique(freq_unit)
    _, knots_freq = create_bspline_basis(
        np.unique(ordinates["omega"]) / np.pi,
        config.n_interior_knots_freq,
        degree=config.degree_freq,
        interior_knots=explicit_freq_unit,
    )
    B_freq_unique = evaluate_bspline_basis(
        unique_freq_unit, knots_freq, degree=config.degree_freq
    )
    P_time = create_bspline_roughness_penalty(
        knots_time, degree=config.degree_time, derivative_order=config.diff_order_time
    )
    P_freq = create_bspline_roughness_penalty(
        knots_freq, degree=config.degree_freq, derivative_order=config.diff_order_freq
    )
    whitened = whiten_penalty_pair(P_time, P_freq)
    basis_eig_time = B_time @ whitened["U_time"]
    basis_eig_freq_unique = B_freq_unique @ whitened["U_freq"]

    pls_init = _scattered_pls_init(
        summed_power / counts, B_time, B_freq_unique, P_time, P_freq, config
    )
    init_sites = whitened_init_values(pls_init, whitened, config)

    kernel = NUTS(
        _dynamic_whittle_model,
        init_strategy=init_to_value(values=init_sites),
        max_tree_depth=10, target_accept_prob=0.85,
    )
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=num_chains, chain_method="sequential", progress_bar=False)
    nuts_t0 = time.perf_counter()
    mcmc.run(
        random.PRNGKey(random_seed),
        jnp.asarray(summed_power), jnp.asarray(counts),
        jnp.asarray(basis_eig_time), jnp.asarray(basis_eig_freq_unique),
        jnp.asarray(whitened["lam_time"]), jnp.asarray(whitened["lam_freq"]),
        jnp.asarray(whitened["joint_null"]), config,
        extra_fields=("diverging",),
    )
    nuts_runtime_s = time.perf_counter() - nuts_t0

    # Evaluate the posterior PSD on a regular (u, omega) grid for comparison.
    eig_samples = np.asarray(mcmc.get_samples()["eig_coeffs"])  # (n, K_t, K_f)
    dense_u = np.linspace(ordinates["u"].min(), ordinates["u"].max(), n_time_grid)
    # Report on the original m Fourier rungs even when the likelihood is binned.
    omega_grid = np.unique(ordinates["omega"])
    freq_grid_hz = omega_grid / (2.0 * np.pi * dt)
    BUt = evaluate_bspline_basis(dense_u, knots_time, degree=config.degree_time) @ whitened["U_time"]
    BUf = evaluate_bspline_basis(omega_grid / np.pi, knots_freq, degree=config.degree_freq) @ whitened["U_freq"]
    log_psd_grid = np.einsum(
        "ta,nab,fb->ntf", BUt, eig_samples, BUf, optimize=True
    )

    log_psd_mean = np.mean(log_psd_grid, axis=0)
    psd_geometric_mean = np.exp(log_psd_mean)
    n_freq_original = np.unique(ordinates["omega"]).size
    n_time_original = ordinates["mi"].size // n_freq_original
    fit_provenance = provenance(
        seed=random_seed,
        dt=dt,
        config=config,
        source_data={"shape": list(np.asarray(data).shape)},
    )
    fit_provenance["moving_periodogram"] = {"m": int(m), "thin": int(thin)}
    fit_provenance["knot_allocation"] = {
        "time": "explicit" if explicit_time is not None else "linear",
        "frequency": "explicit" if explicit_freq_hz is not None else "linear",
    }
    fit_provenance["binning"] = binning_provenance(
        n_time=n_time_original,
        n_freq=n_freq_original,
        time_bin=time_bin,
        freq_bin=freq_bin,
        freq_bin_starts=freq_bin_starts,
        selector_metadata=binning_metadata,
    )
    return {
        "mcmc": mcmc,
        "config": config,
        "ordinates": ordinates,
        "power_observations": observations,
        "time_bin": time_bin,
        "freq_bin": freq_bin,
        "freq_bin_starts": freq_bin_starts,
        "time_grid": dense_u,
        "freq_grid": freq_grid_hz,
        "omega_grid": omega_grid,
        "knots_time": knots_time,
        "knots_freq": knots_freq,
        "knots_time_physical": knots_time.copy(),
        "knots_freq_physical": knots_freq / (2.0 * dt),
        "whitened": whitened,
        "eig_coeff_samples": eig_samples,
        "log_psd_mean": log_psd_mean,
        "psd_geometric_mean": psd_geometric_mean,
        "psd_mean": psd_geometric_mean,  # Deprecated compatibility alias.
        "psd_lower": np.exp(np.percentile(log_psd_grid, 5.0, axis=0)),
        "psd_upper": np.exp(np.percentile(log_psd_grid, 95.0, axis=0)),
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
        "nuts_runtime_s": float(nuts_runtime_s),
        "provenance": fit_provenance,
    }
