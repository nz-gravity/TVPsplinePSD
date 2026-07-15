"""Plotting helpers for WDM log-P-spline PSD results."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .model import power_floor


def set_paper_style() -> None:
    """Apply a publication style matching the PRD/ApJ reference figures.

    Computer-Modern math (no system LaTeX needed) with a serif body font,
    inward major+minor ticks on all four spines, frameless legends, and no
    gridlines -- the conventions used by Digman & Cornish (2022) and Rosati &
    Littenberg (2024). Call once at the top of a figure script.
    """
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 1.8,
        "savefig.dpi": 200,
        "figure.dpi": 120,
    })


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = 160) -> Path:
    """Save and close a figure, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_surface_knots(
    ax: plt.Axes,
    results: dict[str, object],
    *,
    time_transform=None,
    freq_transform=None,
):
    """Overlay the realized interior tensor-product knots on a surface axis.

    Optional transforms map the fit coordinates into the coordinates displayed
    by the plot (for example, normalized time to days or a warped frequency
    coordinate back to Hz).
    """
    config = results["config"]
    time_knots = np.asarray(results["knots_time_physical"])[
        config.degree_time + 1:-(config.degree_time + 1)
    ]
    freq_knots = np.asarray(results["knots_freq_physical"])[
        config.degree_freq + 1:-(config.degree_freq + 1)
    ]
    if time_transform is not None:
        time_knots = np.asarray(time_transform(time_knots))
    if freq_transform is not None:
        freq_knots = np.asarray(freq_transform(freq_knots))
    knot_t, knot_f = np.meshgrid(time_knots, freq_knots, indexing="ij")
    return ax.scatter(
        knot_t.ravel(), knot_f.ravel(), s=12, facecolors="none",
        edgecolors="red", linewidths=0.7, zorder=3,
    )


def quicklook(idata, *, path: str | Path | None = None) -> plt.Figure | Path:
    """One-glance summary of a saved fit (see :mod:`tv_pspline_psd.io`).

    Builds a self-contained matplotlib figure with scalar-parameter trace plots
    and the regenerated posterior-mean PSD surface, straight from the stored
    sites, so no per-sample surface needs to have been kept. ``idata`` may be a
    NetCDF path or a loaded ``InferenceData``.

    Args:
        idata: Path to a saved ``.nc`` or a loaded ArviZ tree.
        path: If given, save the figure there and close it; otherwise return it.
    """
    import arviz as az

    from .io import surface_from_idata

    if isinstance(idata, (str, Path)):
        idata = az.from_netcdf(str(idata))

    post = idata["posterior"].dataset
    scalar_vars = [
        v for v in post.data_vars
        if set(post[v].dims) <= {"chain", "draw"}
    ]
    surf = surface_from_idata(idata)

    n_trace = len(scalar_vars)
    n_cols = n_trace + 1  # trace panels plus the surface
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.0))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, scalar_vars):
        for chain in post.coords.get("chain", [0]):
            y = np.asarray(post[name].sel(chain=int(chain)).values).reshape(-1)
            ax.plot(y, lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("draw")

    mesh = axes[n_trace].pcolormesh(
        surf["time_grid"], surf["freq_grid"], surf["log_psd_mean"].T, shading="auto"
    )
    axes[n_trace].set_title("posterior-mean log PSD")
    axes[n_trace].set_xlabel("time")
    axes[n_trace].set_ylabel("frequency")
    fig.colorbar(mesh, ax=axes[n_trace], fraction=0.046)

    attrs = idata.attrs
    bits = [f"div={attrs.get('divergences', '?')}"]
    if attrs.get("nuts_runtime_s") is not None:
        bits.append(f"NUTS {attrs['nuts_runtime_s']:.1f}s")
    if attrs.get("mse_nuts") is not None:
        bits.append(f"MSE {attrs['mse_nuts']:.3f}")
    fig.suptitle("  |  ".join(bits), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if path is not None:
        return save_figure(fig, path)
    return fig


def plot_surface_comparison(
    results: dict[str, object],
    reference_psd: np.ndarray,
    *,
    freq_scale: float = 1.0,
    freq_label: str = "Frequency",
    show_knots: bool = False,
    path: str | Path,
) -> Path:
    """Raw power, posterior-mean and reference log-surfaces side by side.

    When ``show_knots`` is true, red open circles mark the tensor-product
    interior knots on the posterior-mean panel.
    """
    time_grid = np.asarray(results["time_grid"])
    freq_grid = np.asarray(results["freq_grid"]) * freq_scale

    raw_power = np.asarray(results["power"])
    floor = power_floor(raw_power)
    raw = np.log(raw_power + floor)
    post = np.log(np.asarray(results["psd_mean"]) + floor)
    ref = np.log(np.asarray(reference_psd) + floor)
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
    if show_knots:
        plot_surface_knots(
            axes[1], results, freq_transform=lambda values: values * freq_scale,
        )
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
