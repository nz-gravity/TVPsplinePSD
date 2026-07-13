"""Persist a fit as a small ArviZ ``InferenceData`` (NetCDF) and reload it.

Only the tiny posterior sites (``s``, ``phi_time``, ``phi_freq``) are stored -- the
``log S(t, f)`` surface is regenerated from them on demand. With the whitening
matrices and eigen-bases kept in ``constant_data``, the full posterior surface
(mean and credible interval) is reconstructed exactly, so a saved run supports
trace plots, divergence diagnostics, loss plots, and surface replots from a file
that is megabytes rather than gigabytes.

Layout of the saved tree:

* ``posterior`` -- ``s``, ``phi_time``, ``phi_freq`` (chain, draw, ...).
* ``sample_stats`` -- ``diverging``, ``acceptance_rate``, ``n_steps``, ``lp``.
* ``constant_data`` -- grids, knots, eigen-bases, whitening (everything needed to
  rebuild the surface) plus ``power`` and an optional ``true_psd``.
* ``vi`` -- ELBO ``loss`` trace and the VI point-estimate sites (when VI was run).
* root ``attrs`` -- ``config`` (JSON), ``nuts_runtime_s``, ``vi_runtime_s``,
  ``mse_nuts``, ``vi_mse``, ``divergences``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

from .config import PSplineConfig
from .inference import reconstruct_eig_coeff_samples, surface_summaries
from .metrics import mse_log_psd
from .provenance import provenance
from .splines import evaluate_bspline_basis

# Names threaded through reconstruct_eig_coeff_samples / surface_summaries.
_WHITENED_KEYS = ("U_time", "U_freq", "lam_time", "lam_freq", "joint_null")
_POSTERIOR_SITES = ("s", "phi_time", "phi_freq")


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
            stored and used to record ``mse_nuts`` / ``vi_mse``.
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

    basis_eig_time = np.asarray(results["B_time"]) @ np.asarray(whitened["U_time"])
    basis_eig_freq = np.asarray(results["B_freq"]) @ np.asarray(whitened["U_freq"])
    const = {
        "time_grid": ("time", np.asarray(results["time_grid"])),
        "freq_grid": ("freq", np.asarray(results["freq_grid"])),
        "knots_time": ("knot_time", np.asarray(results["knots_time"])),
        "knots_freq": ("knot_freq", np.asarray(results["knots_freq"])),
        "basis_eig_time": (("time", "eig_time"), basis_eig_time),
        "basis_eig_freq": (("freq", "eig_freq"), basis_eig_freq),
        "U_time": (("basis_time", "eig_time"), np.asarray(whitened["U_time"])),
        "U_freq": (("basis_freq", "eig_freq"), np.asarray(whitened["U_freq"])),
        "lam_time": ("eig_time", np.asarray(whitened["lam_time"])),
        "lam_freq": ("eig_freq", np.asarray(whitened["lam_freq"])),
        "joint_null": (("eig_time", "eig_freq"), np.asarray(whitened["joint_null"])),
        "power": (("time", "freq"), np.asarray(results["power"])),
    }
    if true_psd is not None:
        const["true_psd"] = (("time", "freq"), np.asarray(true_psd))
    idata["constant_data"] = xr.Dataset(const)

    if results.get("vi_losses") is not None:
        vi_vars = {"loss": ("vi_step", np.asarray(results["vi_losses"]))}
        idata["vi"] = xr.Dataset(vi_vars)

    attrs: dict[str, object] = {
        "config": json.dumps(asdict(config)),
        "provenance": json.dumps(results.get("provenance", provenance(config=config))),
        "nuts_runtime_s": _as_float(results.get("nuts_runtime_s")),
        "vi_runtime_s": _as_float(results.get("vi_runtime_s")),
        "divergences": int(results.get("divergences", 0)),
    }
    if true_psd is not None:
        attrs["mse_nuts"] = mse_log_psd(true_psd, np.asarray(results["psd_mean"]))
        if results.get("vi_psd_mean") is not None:
            attrs["vi_mse"] = mse_log_psd(true_psd, np.asarray(results["vi_psd_mean"]))
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
    return PSplineConfig(**json.loads(idata.attrs["config"]))


def _posterior_samples(idata: az.InferenceData) -> dict[str, np.ndarray]:
    """Posterior sites as ``(n_samples, ...)`` arrays (chains stacked)."""
    post = idata["posterior"].dataset
    stacked = post[list(_POSTERIOR_SITES)].stack(sample=("chain", "draw"))
    out = {}
    for name in _POSTERIOR_SITES:
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

    On the stored analysis grid this is exact. Passing ``n_time_dense`` /
    ``n_freq_dense`` re-evaluates the B-spline bases on a denser plotting grid
    (posterior mean only, matching ``evaluate_dense_posterior_mean``).

    Returns a dict with ``time_grid``, ``freq_grid``, ``log_psd_mean`` and
    ``psd_geometric_mean`` (always; with the deprecated ``psd_mean`` alias),
    plus ``log_psd_lower`` / ``log_psd_upper`` /
    ``psd_lower`` / ``psd_upper`` on the native analysis grid.
    """
    config = _config_from_idata(idata)
    const = idata["constant_data"].dataset
    whitened = {k: np.asarray(const[k].values) for k in _WHITENED_KEYS}
    samples = _posterior_samples(idata)
    eig_samples = reconstruct_eig_coeff_samples(samples, whitened, config)

    if n_time_dense is not None or n_freq_dense is not None:
        return _dense_surface(
            const, config, whitened, eig_samples, n_time_dense, n_freq_dense
        )

    basis_eig_time = np.asarray(const["basis_eig_time"].values)
    basis_eig_freq = np.asarray(const["basis_eig_freq"].values)
    log_mean, log_lower, log_upper = surface_summaries(
        eig_samples, basis_eig_time, basis_eig_freq,
        lower_pct=lower_pct, upper_pct=upper_pct,
    )
    return {
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
        dense_freq_unit, np.asarray(const["knots_freq"].values),
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
