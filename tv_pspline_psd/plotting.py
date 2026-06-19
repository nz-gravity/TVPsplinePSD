"""Plotting helpers for WDM log-P-spline PSD results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = 160) -> Path:
    """Save and close a figure, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_surface_comparison(
    results: dict[str, object],
    reference_psd: np.ndarray,
    *,
    freq_scale: float = 1.0,
    freq_label: str = "Frequency",
    path: str | Path,
) -> Path:
    """Raw power, posterior-mean and reference log-surfaces side by side."""
    time_grid = np.asarray(results["time_grid"])
    freq_grid = np.asarray(results["freq_grid"]) * freq_scale

    raw = np.log(np.asarray(results["power"]) + 1e-12)
    post = np.log(np.asarray(results["psd_mean"]) + 1e-12)
    ref = np.log(np.asarray(reference_psd) + 1e-12)
    vmin = min(raw.min(), post.min(), ref.min())
    vmax = max(raw.max(), post.max(), ref.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True, sharey=True)
    for ax, field, title in [
        (axes[0], raw, "Raw WDM log power"),
        (axes[1], post, "Posterior mean log S"),
        (axes[2], ref, "Reference E[w^2]"),
    ]:
        mesh = ax.pcolormesh(
            time_grid, freq_grid, field.T, shading="nearest", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Rescaled WDM time")
        fig.colorbar(mesh, ax=ax, label="log local power")
    axes[0].set_ylabel(freq_label)
    return save_figure(fig, path)


def plot_channel_slice(
    results: dict[str, object],
    reference_psd: np.ndarray,
    channel: int,
    *,
    true_psd: np.ndarray | None = None,
    freq_scale: float = 1.0,
    freq_label: str = "Frequency",
    path: str | Path,
) -> Path:
    """Time profile of one frequency channel with the posterior 90% band."""
    time_grid = np.asarray(results["time_grid"])
    freq_grid = np.asarray(results["freq_grid"])

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    if true_psd is not None:
        ax.plot(time_grid, np.asarray(true_psd)[:, channel], color="tab:green",
                lw=2.0, label="Analytic S(u, f)")
    ax.plot(time_grid, np.asarray(reference_psd)[:, channel], color="black",
            lw=1.5, ls="--", label="Monte Carlo E[w^2]")
    ax.plot(time_grid, np.asarray(results["power"])[:, channel], color="tab:orange",
            lw=1.0, alpha=0.55, label="Raw squared coeffs")
    ax.plot(time_grid, np.asarray(results["psd_mean"])[:, channel], color="tab:blue",
            lw=2.0, label="Posterior mean")
    ax.fill_between(
        time_grid,
        np.asarray(results["psd_lower"])[:, channel],
        np.asarray(results["psd_upper"])[:, channel],
        color="tab:blue", alpha=0.2, label="Posterior 90% interval",
    )
    ax.set_title(
        f"{freq_label} channel f = {freq_grid[channel] * freq_scale:.3g}"
    )
    ax.set_xlabel("Rescaled WDM time")
    ax.set_ylabel("Local power")
    ax.legend(loc="upper right")
    return save_figure(fig, path)
