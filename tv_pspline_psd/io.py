"""Persist a fit as a small ArviZ ``InferenceData`` (NetCDF) and reload it.

Compact posterior sites are stored for tensor or nested residual models -- the
``log S(t, f)`` surface is regenerated from them on demand. With the whitening
matrices and eigen-bases kept in ``constant_data``, the full posterior surface
(mean and credible interval) is reconstructed exactly, so a saved run supports
trace plots, divergence diagnostics, and surface replots from a file
that is megabytes rather than gigabytes.

Layout of the saved tree:

* ``posterior`` -- tensor or nested model sites (chain, draw, ...).
* ``sample_stats`` -- ``diverging``, ``acceptance_rate``, ``n_steps``, ``lp``.
* ``constant_data`` -- grids, knots, eigen-bases, whitening (everything needed to
  rebuild the surface), fixed offsets, masks, power and an optional true PSD.
* root ``attrs`` -- ``config`` (JSON), ``nuts_runtime_s``, ``mse_nuts``, and
  ``divergences``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

from .config import PSplineConfig
from .metrics import mse_log_psd
from .posterior import (
    nested_surface_summaries,
    reconstruct_eig_coeff_samples,
    surface_summaries,
)
from .provenance import provenance
from .splines import evaluate_bspline_basis

# Names threaded through reconstruct_eig_coeff_samples / surface_summaries.
_WHITENED_KEYS = ("U_time", "U_freq", "lam_time", "lam_freq", "joint_null")
_POSTERIOR_SITES = ("s", "phi_time", "phi_freq")
_NESTED_SITES = ("g", "h", "sigma_interaction")


def results_to_idata(
    results: dict[str, object],
    *,
    true_psd: np.ndarray | None = None,
) -> az.InferenceData:
    """Build a small ArviZ tree from a :func:`fit_log_pspline_surface` result.

    Args:
        results: The dict returned by ``fit_log_pspline_surface`` /
            ``run_wdm_psd_mcmc``.
        true_psd: Optional ground-truth PSD on the analysis grid; if given it is
            stored and used to record ``mse_nuts``.
    """
    mcmc = results["mcmc"]
    whitened = results["whitened"]
    config: PSplineConfig = results["config"]  # type: ignore[assignment]

    idata = az.from_numpyro(mcmc)
    # Joint models record the per-sample log_psd surface as a deterministic; drop
    # it from the saved tree -- it is regenerated from the tiny sites on demand.
    surface_vars = [
        v for v in idata["posterior"].dataset.data_vars if str(v).startswith("log_psd")
    ]
    if surface_vars:
        idata["posterior"] = idata["posterior"].dataset.drop_vars(surface_vars)

    structure = results.get("residual_structure", "tensor")
    const = {
        "time_grid": ("time", np.asarray(results["time_grid"])),
        "freq_grid": ("freq", np.asarray(results["freq_grid"])),
        "knots_time": ("knot_time", np.asarray(results["knots_time"])),
        "knots_freq": ("knot_freq", np.asarray(results["knots_freq"])),
        "power": (("time", "freq"), np.asarray(results["power"])),
    }
    if structure == "stationary_plus_interaction":
        const["basis_interaction_time"] = (
            ("time", "eig_time"),
            np.asarray(results["basis_interaction_time"]),
        )
        const["basis_nested_freq"] = (
            ("freq", "eig_freq"),
            np.asarray(results["basis_nested_freq"]),
        )
    elif structure == "tensor":
        const.update(
            {
                "basis_eig_time": (
                    ("time", "eig_time"),
                    np.asarray(results["B_time"]) @ np.asarray(whitened["U_time"]),
                ),
                "basis_eig_freq": (
                    ("freq", "eig_freq"),
                    np.asarray(results["B_freq"]) @ np.asarray(whitened["U_freq"]),
                ),
                "U_time": (("basis_time", "eig_time"), np.asarray(whitened["U_time"])),
                "U_freq": (("basis_freq", "eig_freq"), np.asarray(whitened["U_freq"])),
                "lam_time": ("eig_time", np.asarray(whitened["lam_time"])),
                "lam_freq": ("eig_freq", np.asarray(whitened["lam_freq"])),
                "joint_null": (
                    ("eig_time", "eig_freq"),
                    np.asarray(whitened["joint_null"]),
                ),
            }
        )
    else:
        raise ValueError(f"Unsupported residual_structure: {structure!r}")
    if results.get("log_psd_offset") is not None:
        const["log_psd_offset"] = (
            ("time", "freq"),
            np.asarray(results["log_psd_offset"]),
        )
    if "likelihood_mask" in results:
        const["likelihood_mask"] = (
            ("time", "freq"),
            np.asarray(results["likelihood_mask"], dtype=bool),
        )
    # New explicit-knot fits retain the historical normalized ``knots_freq``
    # for reconstruction compatibility and also persist the user-facing grid
    # coordinates. Older result dictionaries simply omit these optional vars.
    if "knots_time_physical" in results:
        const["knots_time_physical"] = (
            "knot_time",
            np.asarray(results["knots_time_physical"]),
        )
    if "knots_freq_physical" in results:
        const["knots_freq_physical"] = (
            "knot_freq",
            np.asarray(results["knots_freq_physical"]),
        )
    if true_psd is not None:
        const["true_psd"] = (("time", "freq"), np.asarray(true_psd))
    idata["constant_data"] = xr.Dataset(const)

    attrs: dict[str, object] = {
        "schema_version": 2,
        "residual_structure": structure,
        "config": json.dumps(asdict(config)),
        "provenance": json.dumps(results.get("provenance", provenance(config=config))),
        "nuts_runtime_s": _as_float(results.get("nuts_runtime_s")),
        "divergences": int(results.get("divergences", 0)),
    }
    if true_psd is not None:
        attrs["mse_nuts"] = mse_log_psd(true_psd, np.asarray(results["psd_mean"]))
    idata.attrs.update({k: v for k, v in attrs.items() if v is not None})
    return idata


def save_run(
    results: dict[str, object],
    path: str | Path,
    *,
    true_psd: np.ndarray | None = None,
) -> Path:
    """Save a fit to a single NetCDF file and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # DataTree-backed InferenceData requires NetCDF4.  Pin the backend so saving
    # does not depend on xarray's optional-engine discovery order.
    results_to_idata(results, true_psd=true_psd).to_netcdf(path, engine="h5netcdf")
    return path


def load_run(path: str | Path) -> az.InferenceData:
    """Load a saved fit."""
    return az.from_netcdf(str(path))


def _config_from_idata(idata: az.InferenceData) -> PSplineConfig:
    config_data = json.loads(idata.attrs["config"])
    # Saved artifacts from the retired time-only allocation API keep their
    # stored knot vectors, so reconstruction remains exact. Their frequency
    # basis was linear by construction.
    retired = {
        "adaptive_time_knots",
        "adaptive_time_knot_smoothing",
        "adaptive_time_knot_floor",
    }
    if retired.intersection(config_data):
        config_data.pop("freq_knot_strategy", None)
        config_data["freq_knot_strategy"] = "linear"
        for key in retired:
            config_data.pop(key, None)
    return PSplineConfig(**config_data)


def _posterior_samples(
    idata: az.InferenceData, sites: tuple[str, ...] = _POSTERIOR_SITES
) -> dict[str, np.ndarray]:
    """Posterior sites as ``(n_samples, ...)`` arrays (chains stacked)."""
    post = idata["posterior"].dataset
    stacked = post[list(sites)].stack(sample=("chain", "draw"))
    out = {}
    for name in sites:
        arr = np.asarray(stacked[name].transpose("sample", ...).values)
        out[name] = arr
    return out


def surface_from_idata(
    idata: az.InferenceData,
    *,
    n_time_dense: int | None = None,
    n_freq_dense: int | None = None,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> dict[str, np.ndarray]:
    """Regenerate the posterior ``log S`` / PSD surface from a saved fit.

    On the stored analysis grid this is exact for tensor and nested residual
    models, including fixed log-PSD offsets. Legacy files that omitted an
    applied offset cannot be reconstructed and raise a ValueError.
    Passing ``n_time_dense`` /
    ``n_freq_dense`` re-evaluates the B-spline bases on a denser plotting grid
    (posterior mean only, matching ``evaluate_dense_posterior_mean``). Dense
    evaluation currently supports tensor models without a nonzero offset only.

    Returns a dict with ``time_grid``, ``freq_grid``, ``log_psd_mean`` and
    ``psd_geometric_mean`` (always; with the deprecated ``psd_mean`` alias),
    plus ``log_psd_lower`` / ``log_psd_upper`` /
    ``psd_lower`` / ``psd_upper`` on the native analysis grid.
    """
    config = _config_from_idata(idata)
    const = idata["constant_data"].dataset
    structure = idata.attrs.get("residual_structure", "tensor")
    offset = np.asarray(const["log_psd_offset"]) if "log_psd_offset" in const else 0.0
    # Legacy files may record that an offset was used without storing its values.
    metadata = json.loads(idata.attrs.get("provenance", "{}"))
    if "log_psd_offset" not in const and metadata.get("log_psd_offset", {}).get(
        "applied"
    ):
        raise ValueError(
            "This legacy file omitted log_psd_offset; regenerate it from the original fit."
        )
    dense = n_time_dense is not None or n_freq_dense is not None
    if dense and (structure != "tensor" or np.any(offset)):
        raise ValueError(
            "Dense reconstruction requires a tensor model without log_psd_offset. "
            "Use the stored analysis grid for offset or stationary-plus-interaction fits."
        )
    if structure == "stationary_plus_interaction":
        samples = _posterior_samples(idata, _NESTED_SITES)
        basis_time = np.asarray(const["basis_interaction_time"])
        basis_freq = np.asarray(const["basis_nested_freq"])
        log_mean, log_lower, log_upper = nested_surface_summaries(
            samples["g"].reshape(-1, basis_freq.shape[1]),
            samples["h"].reshape(-1, basis_time.shape[1], basis_freq.shape[1]),
            samples["sigma_interaction"],
            basis_time,
            basis_freq,
            lower_pct=lower_pct,
            upper_pct=upper_pct,
        )
    elif structure == "tensor":
        whitened = {k: np.asarray(const[k].values) for k in _WHITENED_KEYS}
        samples = _posterior_samples(idata)
        eig_samples = reconstruct_eig_coeff_samples(samples, whitened, config)
        if dense:
            return _dense_surface(
                const, config, whitened, eig_samples, n_time_dense, n_freq_dense
            )
        log_mean, log_lower, log_upper = surface_summaries(
            eig_samples,
            np.asarray(const["basis_eig_time"]),
            np.asarray(const["basis_eig_freq"]),
            lower_pct=lower_pct,
            upper_pct=upper_pct,
        )
    else:
        raise ValueError(f"Unsupported residual_structure: {structure!r}")
    log_mean = log_mean + offset
    log_lower = log_lower + offset
    log_upper = log_upper + offset
    surface = {
        "time_grid": np.asarray(const["time_grid"].values),
        "freq_grid": np.asarray(const["freq_grid"].values),
        "log_psd_mean": log_mean,
        "log_psd_lower": log_lower,
        "log_psd_upper": log_upper,
        "psd_geometric_mean": np.exp(log_mean),
        "psd_mean": np.exp(log_mean),
        "psd_lower": np.exp(log_lower),
        "psd_upper": np.exp(log_upper),
    }
    if "likelihood_mask" in const:
        surface["likelihood_mask"] = np.asarray(
            const["likelihood_mask"].values, dtype=bool
        )
    return surface


def _dense_surface(const, config, whitened, eig_samples, n_time_dense, n_freq_dense):
    time_grid = np.asarray(const["time_grid"].values)
    freq_grid = np.asarray(const["freq_grid"].values)
    n_t = n_time_dense or time_grid.size
    n_f = n_freq_dense or freq_grid.size
    dense_time = np.linspace(time_grid[0], time_grid[-1], n_t)
    dense_freq = np.linspace(freq_grid[0], freq_grid[-1], n_f)
    dense_freq_unit = dense_freq / np.maximum(freq_grid[-1], 1e-12)

    B_time = evaluate_bspline_basis(
        dense_time, np.asarray(const["knots_time"].values), degree=config.degree_time
    )
    B_freq = evaluate_bspline_basis(
        dense_freq_unit,
        np.asarray(const["knots_freq"].values),
        degree=config.degree_freq,
    )
    # Posterior-mean coefficient matrix in the original (un-whitened) basis.
    W_mean = whitened["U_time"] @ eig_samples.mean(axis=0) @ whitened["U_freq"].T
    dense_log_psd = B_time @ W_mean @ B_freq.T
    return {
        "time_grid": dense_time,
        "freq_grid": dense_freq,
        "log_psd_mean": dense_log_psd,
        "psd_geometric_mean": np.exp(dense_log_psd),
        "psd_mean": np.exp(dense_log_psd),
    }


def _as_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


__all__ = ["results_to_idata", "save_run", "load_run", "surface_from_idata"]
