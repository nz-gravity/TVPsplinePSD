"""Raw WDM log-power spectrograms of the Mojito processed TDI noise.

Loads the processed (X, Y, Z) second-generation TDI noise from the Mojito
pipeline, forms the orthogonal (A, E, T) combinations, and renders the raw WDM
log-power surface of each channel. The point is to see how the TDI
transfer-function nulls (the horizontal stripes in log |w|^2) sit in the
time-frequency plane and whether they drift over the span -- the same "raw WDM
log power" panel produced by ``fit_ollie_tdi.py``, but for the Mojito data set
and without any fit.

The data are ~716 days of X/Y/Z TDI at fs = 0.2 Hz (dt = 5 s), Tukey-tapered at
the edges (alpha = 0.05 -> the first/last ~18 days ramp up from zero). By
default the analysis window is cropped to the untapered interior. Use ``--days``
(and optionally ``--start-day``) to zoom into a shorter stretch, e.g. a 30-day
window to compare against the 30-day reference simulation.

Run:
    python studies/ollie_tdi/check_XYZ.py                 # full untapered span
    python studies/ollie_tdi/check_XYZ.py --days 30       # first 30 clean days
    python studies/ollie_tdi/check_XYZ.py --days 30 --start-day 300
    python studies/ollie_tdi/check_XYZ.py --nt 256 --days 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tv_pspline_psd import PSplineConfig, set_paper_style, wdm_analysis_coefficients

set_paper_style()

REPO = Path(__file__).resolve().parents[2]
# The 6.8 GB Mojito file lives in the sibling MojitoProcessor checkout.
DATA = (
    REPO.parent
    / "MojitoProcessor"
    / "Mojito_Data"
    / "processed_segments_noise_no_segmentation.h5"
)
RESULTS_DIR = REPO / "studies" / "results" / "ollie_tdi"

DT = 5.0  # s (fs = 0.2 Hz, Nyquist 0.1 Hz)
TAPER_FRAC = 0.025  # Tukey alpha=0.05 tapers alpha/2 of the series at each end


def load_xyz(
    path: Path, nt: int, days: float | None, start_day: float | None
) -> tuple[dict[str, np.ndarray], float, float]:
    """Load an X/Y/Z window, cropped so the length is WDM-valid.

    The window is ``[start_day, start_day + days]`` in mission days; both default
    to the untapered interior (``days=None`` runs to the trailing taper). The WDM
    transform needs ``nt`` and ``nf = N / nt`` both even, so the sample count is
    cropped to the largest ``nt * nf`` that fits with ``nf`` even.

    Returns ``(xyz, start_used_days, t_full_days)``.
    """
    with h5py.File(path, "r") as h:
        grp = h["processed/segment0"]
        n_full = grp["X"].shape[0]
        t_full = n_full * DT / 86400.0
        taper = TAPER_FRAC * t_full

        s0 = taper if start_day is None else start_day
        e0 = (t_full - taper) if days is None else (s0 + days)
        s0, e0 = max(0.0, s0), min(t_full, e0)
        if e0 <= s0:
            raise ValueError(f"empty window: start={s0:.1f} d >= end={e0:.1f} d")
        if s0 < taper - 1e-6 or e0 > t_full - taper + 1e-6:
            print(
                f"[warn] window [{s0:.1f}, {e0:.1f}] d overlaps the Tukey taper "
                f"(first/last {taper:.1f} d); edge power is attenuated"
            )

        i0 = int(round(s0 * 86400.0 / DT))
        n_target = int(round(e0 * 86400.0 / DT)) - i0
        nf = n_target // nt
        nf -= nf % 2  # nf must be even
        if nf < 2:
            raise ValueError(
                f"window of {n_target} samples too short for nt={nt} "
                f"(need >= {2 * nt}); reduce --nt or widen --days"
            )
        n_use = nt * nf
        xyz = {c: grp[c][i0 : i0 + n_use] for c in ("X", "Y", "Z")}
        start_used = i0 * DT / 86400.0
    return xyz, start_used, t_full


def orthogonal_aet(xyz: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Standard orthogonal TDI combinations from X/Y/Z."""
    X, Y, Z = xyz["X"], xyz["Y"], xyz["Z"]
    return {
        "A": (Z - X) / np.sqrt(2.0),
        "E": (X - 2.0 * Y + Z) / np.sqrt(6.0),
        "T": (X + Y + Z) / np.sqrt(3.0),
    }


def wdm_log_power(
    x: np.ndarray, nt: int, config: PSplineConfig, t_window: float, t0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mission_day, freq_Hz, log_power) on the trimmed WDM grid.

    ``time_grid`` is normalised to [0, 1] over the window, so the absolute
    mission day is ``t0 + time_grid * t_window``.
    """
    coeffs, time_grid, freq_grid = wdm_analysis_coefficients(x, DT, nt, config)
    power = coeffs**2  # WDM has one real coeff per cell
    # Scale-free guard for log(): fill any exact-zero cells with the smallest
    # positive power in this channel rather than an absolute floor (the TDI
    # amplitudes are ~1e-20, so a fixed floor would flatten the whole surface).
    if not power.all():
        power = np.where(power > 0.0, power, power[power > 0.0].min())
    return t0 + time_grid * t_window, freq_grid, np.log(power)


def plot_panels(
    channels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    out: Path,
    ylim: tuple[float, float],
) -> None:
    """One row of shared-scale log-power spectrograms, one per channel."""
    names = list(channels)
    # Shared, robust color scale across the row so relative power is comparable.
    stacked = np.concatenate([channels[n][2].ravel() for n in names])
    vmin, vmax = np.percentile(stacked, [2.0, 99.5])

    fig, axes = plt.subplots(
        1, len(names), figsize=(2.55 * len(names) + 0.8, 2.7),
        constrained_layout=True, sharey=True,
    )
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, name in zip(axes, names):
        tg, fg, lp = channels[name]
        mesh = ax.pcolormesh(
            tg, fg, lp.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlabel("time [days]")
        ax.set_title(rf"raw WDM log power: {name}")
    axes[0].set_ylabel("f [Hz]")
    fig.colorbar(mesh, ax=axes, shrink=0.9, label=r"$\log |w|^2$")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nt", type=int, default=1024, help="WDM time bins (even)")
    parser.add_argument(
        "--days", type=float, default=None,
        help="analysis-window length in days (default: full untapered span)",
    )
    parser.add_argument(
        "--start-day", type=float, default=None, dest="start_day",
        help="window start in mission days (default: end of the leading taper)",
    )
    parser.add_argument(
        "--fmin", type=float, default=1e-4, help="lower frequency limit [Hz] for the y-axis",
    )
    args = parser.parse_args()
    if args.nt % 2:
        parser.error("--nt must be even")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    xyz, start_used, t_full = load_xyz(DATA, args.nt, args.days, args.start_day)
    aet = orthogonal_aet(xyz)
    n_use = xyz["X"].size
    t_window = n_use * DT / 86400.0
    end_used = start_used + t_window
    windowed = args.days is not None or args.start_day is not None
    print(
        f"[data] N={n_use}, window days {start_used:.1f}-{end_used:.1f} "
        f"({t_window:.1f} d), nt={args.nt}, nf={n_use // args.nt}, dt={DT:.0f}s"
    )

    # The taper is excluded by the data window (see load_xyz), so only a few WDM
    # edge bins (finite wavelet support at the window cut) need trimming. Drop the
    # DC channel so the frequency axis can be log-scaled.
    config = PSplineConfig(
        trim_time_bins=4, trim_low_freq_channels=1, trim_high_freq_channels=0,
    )

    ylim = (args.fmin, 1.0 / (2.0 * DT))  # up to Nyquist
    tag = f"_d{start_used:.0f}-{end_used:.0f}" if windowed else ""
    span = f"days {start_used:.0f}-{end_used:.0f}" if windowed else f"{t_window:.0f} d"

    surfaces: dict[str, dict] = {"XYZ": {}, "AET": {}}
    for label, group in (("XYZ", xyz), ("AET", aet)):
        for name, series in group.items():
            surfaces[label][name] = wdm_log_power(
                series, args.nt, config, t_window, start_used
            )
        print(f"[wdm] {label} done")

    for label in ("XYZ", "AET"):
        out = RESULTS_DIR / f"mojito_wdm_logpower_{label}{tag}.png"
        plot_panels(surfaces[label], f"Mojito TDI noise -- {label} ({span})", out, ylim)


if __name__ == "__main__":
    main()
