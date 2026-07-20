"""Persistent-chain warm start for blocked signal/noise samplers."""

from __future__ import annotations

import numpy as np

from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface


def test_initial_state_advances_persistent_chain_without_warmup() -> None:
    nt, nf = 32, 8
    rng = np.random.default_rng(0)
    time_grid = np.linspace(0.0, 1.0, nt)
    freq_grid = np.linspace(1e-3, 5e-3, nf)
    coeffs = rng.standard_normal((1, nt, nf))

    config = PSplineConfig(
        n_interior_knots_time=4,
        n_interior_knots_freq=4,
        freq_knot_strategy="linear",
    )
    common = dict(
        time_grid=time_grid, freq_grid=freq_grid, config=config,
        n_samples=20, random_seed=0, progress_bar=False,
    )
    first = fit_log_pspline_surface(coeffs=coeffs, n_warmup=50, **common)
    assert first["last_state"] is not None
    assert first["psd_last_draw"].shape == (nt, nf)
    assert np.all(first["psd_last_draw"] > 0)

    # A blocked sweep hands the noise kernel a *new* residual but the same
    # spline setup; the chain must continue from the adapted state.
    new_coeffs = coeffs + 0.1 * rng.standard_normal(coeffs.shape)
    second = fit_log_pspline_surface(
        coeffs=new_coeffs, n_warmup=0, initial_state=first["last_state"], **common
    )
    assert second["psd_last_draw"].shape == (nt, nf)
    assert not np.allclose(second["psd_last_draw"], first["psd_last_draw"])
    # The continued chain starts where the first left off, so the first
    # retained draws should be in the same posterior region.
    assert np.isfinite(second["log_psd_mean"]).all()
    s_first = first["samples"]["s"]
    s_second = second["samples"]["s"]
    assert s_first.shape == s_second.shape
    assert not np.array_equal(s_first, s_second)

    # The continued chain must target the NEW data's likelihood, not a cached
    # potential: handing it 10x-amplitude coefficients (100x power) must pull
    # the sampled surface up by roughly log(100).
    loud = fit_log_pspline_surface(
        coeffs=10.0 * coeffs, n_warmup=0, initial_state=first["last_state"], **common
    )
    level_shift = loud["log_psd_mean"].mean() - first["log_psd_mean"].mean()
    assert level_shift > 2.0

