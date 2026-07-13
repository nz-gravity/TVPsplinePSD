"""TV-PSD noise estimation on a short segment of the Mojito TDI data.

Selects a ``--days``-long window (default 7 d) starting at ``--start-day`` from
the processed Mojito X/Y/Z (or A/E/T) noise, WDM-transforms it, restricts the
analysis to ``[--fmin, --fmax]`` Hz (default 1e-4..1e-1), and fits the
time-varying log-P-spline PSD surface. This is the building block for studying
how the (weakly non-stationary) TDI null folds into the noise estimate: fit
different 7-day windows along the mission and compare the recovered surfaces.

Over a single 7-day window the ~1.4% annual null drift is negligible, so this
first run is effectively a stationary noise estimate / null test. Longer or
edge-of-orbit windows are where the non-stationarity will show up.

Run:
    python studies/ollie_tdi/fit_mojito_segment.py --start-day 355
    python studies/ollie_tdi/fit_mojito_segment.py --start-day 355 --days 7 --channel A
    python studies/ollie_tdi/fit_mojito_segment.py --start-day 100 --end-day 107
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from check_XYZ import DATA, DT, orthogonal_aet
from scipy.signal import welch

from tv_pspline_psd import (
    PSplineConfig,
    run_wdm_psd_mcmc,
    set_paper_style,
    summarize_mcmc_diagnostics,
)
from tv_pspline_psd.datasets import wdm_white_noise_calibration

set_paper_style()

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "studies" / "results" / "ollie_tdi"


def load_segment(
    channel: str, nt: int, start_day: float, days: float | None
) -> tuple[np.ndarray, float]:
    """Load one channel window at an exact, WDM-valid length.

    Finite windows are cropped to the largest ``nt * nf`` no longer than the
    request, with ``nf`` even.  ``days=None`` loads the full series (and ignores
    ``start_day``).  Invalid starts and requests extending beyond the data are
    rejected rather than silently clamped or truncated.
    """
    if channel not in {"X", "Y", "Z", "A", "E", "T"}:
        raise ValueError(f"unknown channel {channel!r}; expected one of XYZ/AET")
    if nt <= 0:
        raise ValueError("nt must be positive")

    with h5py.File(DATA, "r") as h:
        grp = h["processed/segment0"]
        n_full = grp["X"].shape[0]
        t_full = n_full * DT / 86400.0
        if days is None:
            s0 = 0.0
            n_requested = n_full
        else:
            if not np.isfinite(start_day) or start_day < 0.0 or start_day >= t_full:
                raise ValueError(
                    f"start_day={start_day!r} is outside the data span [0, {t_full:.6g})"
                )
            if not np.isfinite(days) or days <= 0.0:
                raise ValueError("days must be finite and positive")
            s0 = start_day
            n_requested = int(round(days * 86400.0 / DT))

        i0 = int(round(s0 * 86400.0 / DT))
        if i0 + n_requested > n_full:
            end_day = (i0 + n_requested) * DT / 86400.0
            raise ValueError(
                f"requested window [{s0:.6g}, {end_day:.6g}] days extends "
                f"past EOF at day {t_full:.6g}"
            )

        nf = n_requested // nt
        nf -= nf % 2
        if nf < 2:
            raise ValueError(f"window too short for nt={nt}; reduce --nt or widen --days")
        n_use = nt * nf
        xyz = {c: grp[c][i0 : i0 + n_use] for c in ("X", "Y", "Z")}

    lengths = {c: values.size for c, values in xyz.items()}
    if any(length != n_use for length in lengths.values()):
        raise ValueError(
            f"HDF5 slice was truncated: requested {n_use} samples, got {lengths}"
        )
    series = xyz[channel] if channel in xyz else orthogonal_aet(xyz)[channel]
    if series.size != n_use:
        raise ValueError(
            f"derived channel {channel} has {series.size} samples; expected {n_use}"
        )
    return series, i0 * DT / 86400.0


def band_trims(n_use: int, nt: int, fmin: float, fmax: float) -> tuple[int, int]:
    """Map [fmin, fmax] Hz to (trim_low, trim_high) WDM channel counts.

    WDM channels are linear: ``f_k = k * df`` with ``df = 1/(2 nf dt)`` and top
    channel ``nf`` at Nyquist ``1/(2 dt)``. Drops channels below fmin (never the
    DC channel is kept anyway) and above fmax.
    """
    nf = n_use // nt
    df = 1.0 / (2.0 * nf * DT)
    trim_low = max(1, int(np.ceil(fmin / df)))
    trim_high = max(0, nf - int(np.floor(fmax / df)))
    return trim_low, trim_high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="X", choices=[*"XYZ", *"AET"])
    parser.add_argument("--start-day", type=float, default=355.0, dest="start_day")
    parser.add_argument("--days", type=float, default=7.0, help="segment length [days]")
    parser.add_argument(
        "--end-day", type=float, default=None, dest="end_day",
        help="segment end [days]; overrides --days if given",
    )
    parser.add_argument("--fmin", type=float, default=1e-4, help="low frequency [Hz]")
    parser.add_argument("--fmax", type=float, default=1e-1, help="high frequency [Hz]")
    parser.add_argument("--nt", type=int, default=32, help="WDM time bins (even)")
    parser.add_argument("--freq-knots", type=int, default=30, dest="freq_knots")
    parser.add_argument("--time-knots", type=int, default=8, dest="time_knots")
    parser.add_argument("--n-warmup", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=2)
    args = parser.parse_args()
    if args.nt % 2:
        parser.error("--nt must be even")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    days = args.days if args.end_day is None else (args.end_day - args.start_day)
    series, start_used = load_segment(args.channel, args.nt, args.start_day, days)
    n_total = series.size
    end_used = start_used + n_total * DT / 86400.0

    trim_low, trim_high = band_trims(n_total, args.nt, args.fmin, args.fmax)
    nf = n_total // args.nt
    df = 1.0 / (2.0 * nf * DT)
    f_lo, f_hi = trim_low * df, (nf - trim_high) * df
    print(
        f"[data] {args.channel} days {start_used:.2f}-{end_used:.2f} "
        f"({days:.2f} d), N={n_total}, nt={args.nt}, nf={nf}, dt={DT:.0f}s"
    )
    print(
        f"[band] request [{args.fmin:.1e}, {args.fmax:.1e}] Hz -> keep channels "
        f"{trim_low}..{nf - trim_high} = [{f_lo:.2e}, {f_hi:.3f}] Hz ({nf - trim_high - trim_low + 1} chans)"
    )

    config = PSplineConfig(
        n_interior_knots_time=args.time_knots,
        n_interior_knots_freq=args.freq_knots,
        trim_time_bins=2,
        trim_low_freq_channels=trim_low,
        trim_high_freq_channels=trim_high,
        centered=True,  # large grid: centered keeps phi from freezing (see config)
    )

    t0 = time.perf_counter()
    res = run_wdm_psd_mcmc(
        series, dt=DT, nt=args.nt, config=config,
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        num_chains=args.num_chains, random_seed=0,
    )
    total_s = time.perf_counter() - t0
    diag = summarize_mcmc_diagnostics(res)
    print(
        f"[fit] wall={total_s:.0f}s sampling={res['nuts_runtime_s']:.0f}s "
        f"div={diag['divergences']} "
        f"rhat(phi)<= {max(diag['phi_time']['r_hat'], diag['phi_freq']['r_hat']):.3f}"
    )

    # WDM-coefficient power -> one-sided PSD [1/Hz]: E[w^2] = C_m * S_onesided/(2 dt).
    cal = wdm_white_noise_calibration(n_total, DT, args.nt, config)
    to_psd = 2.0 * DT / cal[None, :]
    S_est = res["psd_mean"] * to_psd
    S_lo = res["psd_lower"] * to_psd
    S_hi = res["psd_upper"] * to_psd
    tg_days = start_used + res["time_grid"] * n_total * DT / 86400.0
    fg = res["freq_grid"]

    f_w, P_w = welch(series, fs=1.0 / DT, nperseg=min(n_total, 2**16))

    tag = f"{args.channel}_d{start_used:.0f}-{end_used:.0f}"
    np.savez(
        RESULTS_DIR / f"mojito_fit_{tag}.npz",
        time_grid_days=tg_days, freq_grid=fg,
        psd_mean=S_est, psd_lower=S_lo, psd_upper=S_hi,
        welch_f=f_w, welch_psd=P_w,
        start_day=start_used, end_day=end_used, band=(f_lo, f_hi),
        runtime_s=total_s, nuts_runtime_s=res["nuts_runtime_s"],
        divergences=diag["divergences"],
    )
    with open(RESULTS_DIR / f"mojito_fit_{tag}_diag.json", "w") as fp:
        json.dump(diag, fp, indent=2, default=float)

    # --- Figures ---
    # 1) Recovered surface next to the raw WDM log-power.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True, sharey=True)
    raw_pow = np.log(res["power"] + 1e-300) + np.log(to_psd)
    m0 = axes[0].pcolormesh(tg_days, fg, raw_pow.T, shading="auto", cmap="viridis")
    axes[0].set_title("raw WDM log power")
    m1 = axes[1].pcolormesh(tg_days, fg, np.log(S_est).T, shading="auto", cmap="viridis",
                            vmin=m0.get_clim()[0], vmax=m0.get_clim()[1])
    axes[1].set_title(r"posterior mean $\log \hat S(t,f)$")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(f_lo, f_hi)
        ax.set_xlabel("time [days]")
    axes[0].set_ylabel("f [Hz]")
    fig.colorbar(m1, ax=axes, shrink=0.85, label=r"$\log S$ [1/Hz]")
    fig.suptitle(f"Mojito {args.channel}, days {start_used:.0f}-{end_used:.0f}", fontsize=10)
    fig.savefig(RESULTS_DIR / f"mojito_surface_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Time-averaged spectrum vs Welch.
    fig, ax = plt.subplots(figsize=(3.8, 2.9), constrained_layout=True)
    wband = (f_w >= f_lo) & (f_w <= f_hi)
    ax.loglog(f_w[wband], P_w[wband], color="0.5", lw=0.8, label="Welch (7 d)")
    ax.loglog(fg, S_est.mean(axis=0), color="tab:blue", label="TV fit (time avg)")
    ax.fill_between(fg, S_lo.mean(axis=0), S_hi.mean(axis=0), color="tab:blue", alpha=0.3)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel(r"$S(f)$ [1/Hz]")
    ax.legend(fontsize=8)
    fig.savefig(RESULTS_DIR / f"mojito_spectrum_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] figures + npz in {RESULTS_DIR} (tag {tag})")


if __name__ == "__main__":
    main()
