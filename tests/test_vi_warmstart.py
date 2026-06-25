"""Staged analytic -> VI -> NUTS initialisation for the P-spline surface."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface, save_figure


def _toy_surface(nt: int, nf: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A smooth, separable log-PSD surface and its rescaled grids."""
    time_grid = np.linspace(0.0, 1.0, nt)
    freq_grid = np.linspace(1e-3, 5e-3, nf)
    f_unit = (freq_grid - freq_grid[0]) / (freq_grid[-1] - freq_grid[0])
    log_psd = (
        -2.0
        + 0.8 * np.sin(2.0 * np.pi * time_grid)[:, None]
        - 1.5 * f_unit[None, :]
    )
    return time_grid, freq_grid, log_psd


@pytest.fixture
def toy_data():
    nt, nf = 28, 24
    time_grid, freq_grid, true_log_psd = _toy_surface(nt, nf)
    rng = np.random.default_rng(0)
    # One real component per cell drawn at the true PSD: c ~ N(0, exp(log_psd)).
    coeffs = rng.standard_normal((1, nt, nf)) * np.exp(0.5 * true_log_psd)[None]
    return time_grid, freq_grid, true_log_psd, coeffs


def test_vi_warmstart_runs_and_improves(toy_data, plot_outdir):
    time_grid, freq_grid, true_log_psd, coeffs = toy_data
    config = PSplineConfig(
        n_interior_knots_time=10,
        n_interior_knots_freq=10,
        adaptive_time_knots=False,
    )

    result = fit_log_pspline_surface(
        coeffs, time_grid, freq_grid, config=config,
        n_warmup=40, n_samples=40, random_seed=0,
        use_vi=True, vi_steps=400,
    )

    # The staged path returns both VI and NUTS surfaces.
    vi_log_psd = np.asarray(result["vi_log_psd"])
    nuts_log_psd = np.asarray(result["log_psd_mean"])
    losses = np.asarray(result["vi_losses"])

    assert vi_log_psd.shape == true_log_psd.shape == nuts_log_psd.shape
    assert np.isfinite(vi_log_psd).all()
    assert np.isfinite(nuts_log_psd).all()
    assert result["divergences"] == 0
    # VI optimised: the ELBO loss decreased over the run.
    assert losses[-1] < losses[0]

    # Both stages recover the surface far better than the raw periodogram.
    raw_log = np.log(coeffs[0] ** 2 + 1e-12)
    raw_mse = np.mean((raw_log - true_log_psd) ** 2)
    vi_mse = np.mean((vi_log_psd - true_log_psd) ** 2)
    nuts_mse = np.mean((nuts_log_psd - true_log_psd) ** 2)
    assert vi_mse < raw_mse
    assert nuts_mse < raw_mse

    # Plot: true | VI estimate | NUTS estimate (shared colour scale).
    fields = [
        (true_log_psd, "True log PSD"),
        (vi_log_psd, f"VI estimate (MSE={vi_mse:.3f})"),
        (nuts_log_psd, f"NUTS estimate (MSE={nuts_mse:.3f})"),
    ]
    vmin = min(f.min() for f, _ in fields)
    vmax = max(f.max() for f, _ in fields)
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4), constrained_layout=True, sharey=True
    )
    for ax, (field, title) in zip(axes, fields):
        mesh = ax.pcolormesh(
            time_grid, freq_grid, field.T,
            shading="auto", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Rescaled time")
        fig.colorbar(mesh, ax=ax, label="log PSD")
    axes[0].set_ylabel("Frequency [Hz]")
    fig.suptitle("Staged init: analytic -> VI -> NUTS")
    out = save_figure(fig, Path(plot_outdir) / "vi_warmstart_surface.png")
    assert out.exists()


def test_vi_disabled_by_default(toy_data):
    time_grid, freq_grid, _, coeffs = toy_data
    config = PSplineConfig(
        n_interior_knots_time=10,
        n_interior_knots_freq=10,
        adaptive_time_knots=False,
    )
    result = fit_log_pspline_surface(
        coeffs, time_grid, freq_grid, config=config,
        n_warmup=20, n_samples=20, random_seed=0,
    )
    assert result["vi_losses"] is None
    assert result["vi_log_psd"] is None
    assert result["vi_psd_mean"] is None
