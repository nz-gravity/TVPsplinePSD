"""Linear-frequency zoom on the first Michelson TDI null, with the predicted
null frequency 1/(2<L>) overlaid, for an arbitrary mission window.

The Michelson-X null near 0.06 Hz sits at f = 1/(2L), where L is the one-way
light-travel-time of the arms adjacent to the reference spacecraft. Because the
constellation breathes, L -- and hence the null -- drifts by ~1.4% over the
orbit. That drift is sub-pixel on a log axis (see ``check_XYZ.py``); here we zoom
in on a linear frequency axis and overlay 1/(2<L(t)>) computed from the
simulation's own light-travel-times (``raw/segment0/ltts``), averaged over the
four arms touching the channel's spacecraft.

Run:
    python studies/ollie_tdi/null_zoom.py                       # full span, X
    python studies/ollie_tdi/null_zoom.py --days 7              # 7 d mid-mission
    python studies/ollie_tdi/null_zoom.py --days 30 --start-day 343
    python studies/ollie_tdi/null_zoom.py --channel Y --days 30
"""

from __future__ import annotations

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from check_XYZ import DATA, DT, RESULTS_DIR, orthogonal_aet
from scipy.ndimage import uniform_filter

from tv_pspline_psd import PSplineConfig, set_paper_style, wdm_analysis_coefficients

set_paper_style()

# Arms adjacent to each Michelson channel's reference spacecraft (both directions
# on the two local arms). Their mean one-way light time sets that channel's null.
SC_ARMS = {
    "X": ("12", "21", "13", "31"),  # S/C 1
    "Y": ("23", "32", "21", "12"),  # S/C 2
    "Z": ("31", "13", "32", "23"),  # S/C 3
}


def load_window(
    nt: int, days: float | None, start_day: float | None
) -> tuple[dict[str, np.ndarray], float]:
    """Load an X/Y/Z window (default: full span, centred if only ``days`` given).

    Cropped to the largest WDM-valid ``nt * nf`` (both even). Returns the channel
    dict plus the actual window start in mission days.
    """
    with h5py.File(DATA, "r") as h:
        grp = h["processed/segment0"]
        n_full = grp["X"].shape[0]
        t_full = n_full * DT / 86400.0

        if days is None:
            s0 = 0.0 if start_day is None else start_day
            e0 = t_full
        else:
            s0 = (t_full - days) / 2.0 if start_day is None else start_day
            e0 = s0 + days
        s0, e0 = max(0.0, s0), min(t_full, e0)

        i0 = int(round(s0 * 86400.0 / DT))
        nf = (int(round(e0 * 86400.0 / DT)) - i0) // nt
        nf -= nf % 2
        if nf < 2:
            raise ValueError(f"window too short for nt={nt}; reduce --nt or widen --days")
        n_use = nt * nf
        xyz = {c: grp[c][i0 : i0 + n_use] for c in ("X", "Y", "Z")}
    return xyz, i0 * DT / 86400.0


def predicted_null(channel: str, t_axis: np.ndarray) -> np.ndarray | None:
    """f = 1/(2<L(t)>) sampled on ``t_axis`` (mission days); None for A/E/T."""
    if channel not in SC_ARMS:
        return None
    with h5py.File(DATA, "r") as h:
        g = h["raw/segment0/ltts"]
        stride = max(1, g["times"].shape[0] // 4000)
        tl = (g["times"][::stride] - g["times"][0]) / 86400.0
        lbar = np.mean([g[a][::stride] for a in SC_ARMS[channel]], axis=0)
    return 1.0 / (2.0 * np.interp(t_axis, tl, lbar))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="X", choices=[*"XYZ", *"AET"])
    parser.add_argument("--nt", type=int, default=128, help="WDM time bins (even)")
    parser.add_argument("--days", type=float, default=None, help="window length [days]")
    parser.add_argument(
        "--start-day", type=float, default=None, dest="start_day",
        help="window start [mission days] (default: centred / start of run)",
    )
    parser.add_argument(
        "--fband", type=float, nargs=2, default=(0.045, 0.075),
        metavar=("FLO", "FHI"), help="frequency zoom band [Hz]",
    )
    args = parser.parse_args()
    if args.nt % 2:
        parser.error("--nt must be even")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    xyz, start_used = load_window(args.nt, args.days, args.start_day)
    series = xyz[args.channel] if args.channel in xyz else orthogonal_aet(xyz)[args.channel]
    n_use = series.size
    t_window = n_use * DT / 86400.0
    end_used = start_used + t_window
    print(
        f"[data] channel {args.channel}, days {start_used:.1f}-{end_used:.1f} "
        f"({t_window:.1f} d), nt={args.nt}, nf={n_use // args.nt}, dt={DT:.0f}s"
    )

    config = PSplineConfig(
        trim_time_bins=2, trim_low_freq_channels=1, trim_high_freq_channels=0,
    )
    coeffs, time_grid, freq_grid = wdm_analysis_coefficients(series, DT, args.nt, config)
    tdays = start_used + time_grid * t_window
    logp = np.log(coeffs**2)

    lo, hi = args.fband
    band = (freq_grid >= lo) & (freq_grid <= hi)
    if band.sum() < 4:
        parser.error(f"only {band.sum()} WDM channels in {args.fband} Hz; lower --nt")
    # Light box smoothing tames the per-cell chi^2 speckle so the notch reads
    # cleanly; time width scales with the number of time bins.
    smooth_t = max(3, round(args.nt / 120))
    logp_s = uniform_filter(logp[:, band], size=(smooth_t, 3))
    vmin, vmax = np.percentile(logp_s, [2.0, 99.8])

    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        tdays, freq_grid[band], logp_s.T, shading="auto", cmap="viridis",
        vmin=vmin, vmax=vmax,
    )
    fnull = predicted_null(args.channel, tdays)
    if fnull is not None:
        ax.plot(tdays, fnull, "r-", lw=1.2, label=r"$1/2\langle L\rangle$")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(tdays.min(), tdays.max())
    ax.set_ylim(lo, hi)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("f [Hz]")
    ax.set_title(
        f"{args.channel}: first TDI null, days "
        f"{start_used:.0f}-{end_used:.0f} ({t_window:.0f} d)"
    )
    fig.colorbar(mesh, ax=ax, shrink=0.9, label=r"$\log |w|^2$")

    out = RESULTS_DIR / f"mojito_null_zoom_{args.channel}_d{start_used:.0f}-{end_used:.0f}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
