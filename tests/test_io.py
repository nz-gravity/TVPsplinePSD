"""Round-trip: a saved fit reloads and regenerates the surface exactly."""

from __future__ import annotations

import json

import numpy as np
import pytest

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
        n_interior_knots_time=4, n_interior_knots_freq=4, freq_knot_strategy="linear"
    )
    res = run_wdm_psd_mcmc(
        data,
        dt=0.1,
        nt=24,
        config=config,
        n_warmup=10,
        n_samples=10,
        random_seed=0,
        time_bin=2,
        binning_metadata={"time": {"method": "fixed", "requested_width": 2}},
    )
    # The per-sample surface must never be stored -- only the tiny sites are kept.
    assert "log_psd" not in res["samples"]
    assert isinstance(res["nuts_runtime_s"], float)

    true_psd = np.ones_like(res["psd_mean"])
    path = save_run(res, tmp_path / "run.nc", true_psd=true_psd)
    assert path.stat().st_size < 5_000_000  # small artifact

    idata = load_run(path)
    assert set(idata.children) == {
        "posterior",
        "sample_stats",
        "constant_data",
        "observed_data",
    }
    assert "diverging" in idata["sample_stats"].dataset.data_vars
    assert idata.attrs["nuts_runtime_s"] > 0
    assert "mse_nuts" in idata.attrs
    metadata = json.loads(idata.attrs["provenance"])
    assert metadata["seed"] == 0
    assert metadata["dt"] == 0.1
    assert metadata["nt"] == 24
    assert metadata["binning"]["input_shape"] == [22, 23]
    assert metadata["binning"]["output_shape"] == [11, 23]
    assert metadata["binning"]["time"]["widths"] == [2] * 11
    assert metadata["binning"]["selector"]["time"]["requested_width"] == 2
    assert metadata["likelihood_mask"]["applied"] is False
    np.testing.assert_array_equal(
        idata["constant_data"]["likelihood_mask"].values,
        res["likelihood_mask"],
    )

    # Regenerating the surface from the saved sites reproduces the fit exactly.
    surf = surface_from_idata(idata)
    np.testing.assert_allclose(surf["log_psd_mean"], res["log_psd_mean"], atol=1e-9)
    np.testing.assert_allclose(surf["log_psd_lower"], res["log_psd_lower"], atol=1e-9)
    np.testing.assert_allclose(surf["log_psd_upper"], res["log_psd_upper"], atol=1e-9)
    np.testing.assert_allclose(surf["psd_geometric_mean"], res["psd_mean"])
    np.testing.assert_array_equal(surf["likelihood_mask"], res["likelihood_mask"])

    idata.attrs.pop("schema_version")
    idata.attrs.pop("residual_structure")
    idata["constant_data"] = idata["constant_data"].dataset.drop_vars("log_psd_offset")
    legacy = surface_from_idata(idata)
    np.testing.assert_allclose(legacy["log_psd_mean"], res["log_psd_mean"], atol=1e-9)

    dense = surface_from_idata(idata, n_time_dense=40, n_freq_dense=40)
    assert dense["psd_mean"].shape == (40, 40)


@pytest.mark.parametrize(
    ("structure", "centered"),
    [("tensor", False), ("tensor", True), ("stationary_plus_interaction", True)],
)
def test_offset_and_nested_round_trip(tmp_path, structure, centered):
    from tv_pspline_psd import evaluate_dense_posterior_mean, fit_log_pspline_surface

    time = np.linspace(0.0, 1.0, 8)
    frequency = np.linspace(0.01, 0.1, 9)
    offset = 0.4 + time[:, None] + 2.0 * frequency[None, :]
    coeffs = np.random.default_rng(3).normal(size=(1, 8, 9)) * np.exp(offset / 2)
    mask = np.ones((8, 9), dtype=bool)
    mask[2, 3] = False
    config = PSplineConfig(
        n_interior_knots_time=1,
        n_interior_knots_freq=1,
        freq_knot_strategy="linear",
        centered=centered,
    )
    result = fit_log_pspline_surface(
        coeffs,
        time,
        frequency,
        config=config,
        log_psd_offset=offset,
        likelihood_mask=mask,
        residual_structure=structure,
        interaction_time_knots=0,
        n_warmup=5,
        n_samples=5,
        num_chains=2,
        random_seed=4,
        progress_bar=False,
    )
    idata = load_run(save_run(result, tmp_path / "offset.nc"))
    assert idata.attrs["schema_version"] == 2
    assert idata.attrs["residual_structure"] == structure
    np.testing.assert_array_equal(idata["constant_data"]["log_psd_offset"], offset)
    surface = surface_from_idata(idata)
    for name in (
        "log_psd_mean",
        "log_psd_lower",
        "log_psd_upper",
        "psd_geometric_mean",
    ):
        np.testing.assert_allclose(surface[name], result[name], atol=1e-10)
    np.testing.assert_array_equal(surface["likelihood_mask"], mask)
    with pytest.raises(ValueError, match="Dense reconstruction"):
        surface_from_idata(idata, n_time_dense=12)
    with pytest.raises(ValueError, match="Dense reconstruction"):
        evaluate_dense_posterior_mean(result)
    if structure == "tensor":
        idata["constant_data"] = idata["constant_data"].dataset.drop_vars(
            "log_psd_offset"
        )
        with pytest.raises(ValueError, match="legacy file omitted"):
            surface_from_idata(idata)
