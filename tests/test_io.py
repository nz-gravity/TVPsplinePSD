"""Round-trip: a saved fit reloads and regenerates the surface exactly."""

from __future__ import annotations

import json

import numpy as np

from tv_pspline_psd import (
    PSplineConfig,
    load_run,
    run_wdm_psd_mcmc,
    save_run,
    surface_from_idata,
)


def test_save_load_regenerates_surface(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal(576)
    config = PSplineConfig(
        n_interior_knots_time=4, n_interior_knots_freq=4, adaptive_time_knots=False
    )
    res = run_wdm_psd_mcmc(
        data, dt=0.1, nt=24, config=config,
        n_warmup=10, n_samples=10, random_seed=0,
    )
    # The per-sample surface must never be stored -- only the tiny sites are kept.
    assert "log_psd" not in res["samples"]
    assert isinstance(res["nuts_runtime_s"], float)

    true_psd = np.ones_like(res["psd_mean"])
    path = save_run(res, tmp_path / "run.nc", true_psd=true_psd)
    assert path.stat().st_size < 5_000_000  # small artifact

    idata = load_run(path)
    assert set(idata.children) == {
        "posterior", "sample_stats", "constant_data", "observed_data"
    }
    assert "diverging" in idata["sample_stats"].dataset.data_vars
    assert idata.attrs["nuts_runtime_s"] > 0
    assert "mse_nuts" in idata.attrs
    metadata = json.loads(idata.attrs["provenance"])
    assert metadata["seed"] == 0
    assert metadata["dt"] == 0.1
    assert metadata["nt"] == 24

    # Regenerating the surface from the saved sites reproduces the fit exactly.
    surf = surface_from_idata(idata)
    np.testing.assert_allclose(surf["log_psd_mean"], res["log_psd_mean"], atol=1e-9)
    np.testing.assert_allclose(surf["log_psd_lower"], res["log_psd_lower"], atol=1e-9)
    np.testing.assert_allclose(surf["log_psd_upper"], res["log_psd_upper"], atol=1e-9)
    np.testing.assert_allclose(surf["psd_geometric_mean"], res["psd_mean"])

    dense = surface_from_idata(idata, n_time_dense=40, n_freq_dense=40)
    assert dense["psd_mean"].shape == (40, 40)
