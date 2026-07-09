"""Full-band TV-PSD fit of AET channels built from Ollie's unequal-arm XYZ TDI.

The data are 30 days of X2/Y2/Z2 at fs = 4 Hz
(datasets/ollie_data/simulated_noise_30_days_L1_ext.h5) from a
lisainstrument+pytdi pipeline with unequal, drifting arm lengths; the two
7-day pre-processed segments (lisa_sim_processed_segments_30_days_ext.h5)
are available as cross-checks. The static (equal-arm) orthonormal AET
combination is applied,

    A = (Z - X)/sqrt(2),  E = (X - 2Y + Z)/sqrt(6),  T = (X + Y + Z)/sqrt(3),

and its residual cross-channel coherence is measured to quantify the
approximation. Each channel is brick-wall decimated to Nyquist = 0.125 Hz so
the analysis band is 1e-4..0.1 Hz, then fitted with the WDM log-P-spline
estimator.

The frequency knots are placed via a warped coordinate: log-spaced over the
smooth instrument spectrum below 0.02 Hz and linear above, where the TDI
transfer-function nulls (comb at 1/L ~ 0.030 Hz: 0.03, 0.06, 0.09 Hz) need
dense knots. The fit itself is coordinate-agnostic, so the warp only moves
knots.

Run:
    python studies/ollie_tdi/fit_aet_fullband.py                 # A, 30 days
    python studies/ollie_tdi/fit_aet_fullband.py --channel E --data seg1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import csd, welch

from tv_pspline_psd import (
    PSplineConfig,
    fit_log_pspline_surface,
    set_paper_style,
    summarize_mcmc_diagnostics,
    wdm_analysis_coefficients,
)
from tv_pspline_psd.splines import evaluate_bspline_basis
from datasets import wdm_white_noise_calibration

set_paper_style()

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "datasets" / "ollie_data"
DATA_FULL = DATA_DIR / "simulated_noise_30_days_L1_ext.h5"
DATA_SEGMENTS = DATA_DIR / "lisa_sim_processed_segments_30_days_ext.h5"
RESULTS_DIR = REPO / "studies" / "results" / "ollie_tdi"

DECIMATE = 15           # 0.25 s -> 3.75 s, Nyquist 0.133 Hz (band 1e-4..0.1 Hz);
                        # 15 keeps both the 30-day and 7-day lengths divisible by nt
F_LO, F_BREAK, F_HI = 1e-4, 0.02, 1.0 / (2 * 0.25 * 15)
# Log knots cover the smooth spectrum below the first null; linear knots at
# ~0.0015 Hz spacing (~20 per null period, comb at 1/L ~ 0.030 Hz) resolve
# the null flanks. The cores (6 decades deep) remain smoothed by design.
N_KNOTS_LOG, N_KNOTS_LIN = 24, 70
# WDM time bins / trimmed low channels per data source. nt keeps the time
# resolution at ~2.6-5.6 h; the low trim puts the band edge at ~1e-4 Hz.
GRID = {"full": (128, 4), "seg0": (64, 2), "seg1": (64, 2)}
# ~22 h per edge: the null channels are sensitive enough to see boundary
# wrap-around that the broadband cells hide.
TRIM_TIME_BINS = 4


def load_aet(source: str) -> tuple[dict[str, np.ndarray], float]:
    dt = 0.25
    if source == "full":
        with h5py.File(DATA_FULL) as h:
            if np.any(h["tdis/tdi_flags"][:]):
                raise ValueError("tdi_flags mark gaps; this fit assumes none")
            X, Y, Z = (h[f"tdis/{k}"][:] for k in ("X2", "Y2", "Z2"))
        # No filtering: the raw series is clean at 1e-4 Hz, and a 1e-5 Hz
        # filtfilt highpass injects day-long edge transients into the band.
        # WDM edge-bin trimming handles the series boundaries.
    else:
        with h5py.File(DATA_SEGMENTS) as h:
            g = h[f"processed/segment{source[-1]}"]
            X, Y, Z = g["X"][:], g["Y"][:], g["Z"][:]
    aet = {
        "A": (Z - X) / np.sqrt(2),
        "E": (X - 2 * Y + Z) / np.sqrt(6),
        "T": (X + Y + Z) / np.sqrt(3),
    }
    return aet, dt


def lisa_like_gaps(t_obs_s: float, seed: int = 1) -> list[tuple[float, float]]:
    """LISA-like gap schedule: (start, end) seconds within [0, t_obs_s].

    Scheduled antenna repointings (3.5 h every 14 days) plus unscheduled
    outages (Poisson, ~1/week, duration log-uniform 0.5..24 h).
    """
    gaps = [(t0, t0 + 3.5 * 3600) for t0 in np.arange(14 * 86400, t_obs_s, 14 * 86400)]
    rng = np.random.default_rng(seed)
    n_unsched = rng.poisson(t_obs_s / (7 * 86400))
    starts = rng.uniform(0, t_obs_s, n_unsched)
    durations = np.exp(rng.uniform(np.log(0.5 * 3600), np.log(24 * 3600), n_unsched))
    gaps += [(t0, min(t0 + d, t_obs_s)) for t0, d in zip(starts, durations)]
    return sorted(gaps)


def gate_gaps(
    data: np.ndarray, dt: float, gaps: list[tuple[float, float]],
    taper_s: float = 3600.0,
) -> np.ndarray:
    """Zero the gaps with cosine (Tukey-lobe) tapers into each edge.

    Tapering before the transform confines the leakage of the steep red
    spectrum to the gap's own time bins; a hard (rectangular) gate would
    smear it across the affected columns.
    """
    t = np.arange(data.size) * dt
    w = np.ones_like(data)
    for t0, t1 in gaps:
        w[(t >= t0) & (t <= t1)] = 0.0
        for edge, sgn in ((t0, -1.0), (t1, 1.0)):
            lobe = sgn * (t - edge)  # distance into the good data
            sel = (lobe > 0) & (lobe < taper_s)
            w[sel] = np.minimum(w[sel], 0.5 - 0.5 * np.cos(np.pi * lobe[sel] / taper_s))
    return data * w


def good_time_bins(
    time_grid: np.ndarray, t_obs_s: float, gaps: list[tuple[float, float]],
    nt: int, taper_s: float = 3600.0,
) -> np.ndarray:
    """Boolean mask of WDM time bins untouched by any tapered gap (+1 bin buffer)."""
    half = t_obs_s / nt  # one full bin as buffer on either side of the taper
    centers = time_grid * t_obs_s
    keep = np.ones(centers.size, dtype=bool)
    for t0, t1 in gaps:
        keep &= (centers < t0 - taper_s - half) | (centers > t1 + taper_s + half)
    return keep


def fft_decimate(x: np.ndarray, q: int) -> np.ndarray:
    """Brick-wall lowpass + downsample by cropping the rFFT."""
    n_new = x.size // q
    spec = np.fft.rfft(x[: n_new * q])
    return np.fft.irfft(spec[: n_new // 2 + 1], n=n_new) / q


def warp_freq(f: np.ndarray) -> np.ndarray:
    """Monotone coordinate in which uniform knots are log-spaced below
    F_BREAK and linear above (densities N_KNOTS_LOG : N_KNOTS_LIN)."""
    a = N_KNOTS_LOG / (N_KNOTS_LOG + N_KNOTS_LIN)
    f = np.asarray(f, dtype=float)
    low = a * np.log10(np.maximum(f, F_LO) / F_LO) / np.log10(F_BREAK / F_LO)
    high = a + (1 - a) * (f - F_BREAK) / (F_HI - F_BREAK)
    return np.where(f <= F_BREAK, low, high)


def coherence_matrix(aet: dict[str, np.ndarray], fs: float, nper: int):
    """Pairwise magnitude-squared coherence between the AET channels."""
    psd = {k: welch(v, fs, nperseg=nper)[1] for k, v in aet.items()}
    f = welch(aet["A"], fs, nperseg=nper)[0]
    coh = {}
    for a, b in (("A", "E"), ("A", "T"), ("E", "T")):
        _, pab = csd(aet[a], aet[b], fs, nperseg=nper)
        coh[a + b] = np.abs(pab) ** 2 / (psd[a] * psd[b])
    n_seg = aet["A"].size // (nper // 2) - 1
    return f, psd, coh, 1.0 / n_seg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="A", choices=["A", "E", "T"])
    parser.add_argument("--data", default="full", choices=list(GRID))
    parser.add_argument("--n-warmup", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--use-vi", action="store_true",
                        help="refine the warm start with VI before NUTS")
    parser.add_argument("--non-centered", action="store_true",
                        help="use the package's default non-centered prior "
                             "(centered is the right geometry at this grid size)")
    parser.add_argument("--gaps", action="store_true",
                        help="inject LISA-like gaps (gate + taper, mask WDM bins)")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.channel}_{args.data}"
    if args.gaps:
        tag += "_gaps"
    if args.n_samples < 100:
        tag += "_pilot"  # keep short test chains from clobbering production output
    nt, trim_low = GRID[args.data]

    aet_raw, dt_raw = load_aet(args.data)

    # --- Coherence of the static AET approximation (full-rate data). ---
    coh_f, coh_psd, coh, coh_floor = coherence_matrix(aet_raw, 1 / dt_raw, 2**18)
    band = (coh_f >= F_LO) & (coh_f <= F_HI)
    print("[coh] band medians:",
          {k: float(np.median(v[band])) for k, v in coh.items()},
          f"(Welch floor ~{coh_floor:.3f})")

    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    for (k, v), c in zip(coh.items(), ("tab:blue", "tab:orange", "tab:green")):
        ax.loglog(coh_f[band], v[band], lw=0.6, color=c, alpha=0.8,
                  label=f"{k[0]}--{k[1]}")
    ax.axhline(coh_floor, color="black", ls=":", lw=0.8, label="estimator floor")
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel(r"coherence $|C_{ij}|^2/(S_i S_j)$")
    ax.set_ylim(1e-4, 1.5)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(RESULTS_DIR / f"aet_coherence_{args.data}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # --- WDM TV-PSD fit over the full LISA band. ---
    data = fft_decimate(aet_raw[args.channel], DECIMATE)
    dt = dt_raw * DECIMATE
    n_total = data.size
    t_obs_s = n_total * dt
    gaps = lisa_like_gaps(t_obs_s) if args.gaps else []
    if gaps:
        data = gate_gaps(data, dt, gaps)
        print(f"[gaps] {len(gaps)} gaps, "
              f"{sum(t1 - t0 for t0, t1 in gaps) / 3600:.1f} h gated")
    print(f"[data] {tag}: {n_total} samples, dt={dt}s, "
          f"Nyquist={1/(2*dt):.2f} Hz, T={n_total*dt/86400:.1f} days")

    # Low trim puts the band edge at ~1e-4 Hz; 2 edge time bins cover the
    # Tukey taper and WDM wrap-around.
    config = PSplineConfig(
        n_interior_knots_freq=N_KNOTS_LOG + N_KNOTS_LIN,
        trim_low_freq_channels=trim_low,
        trim_time_bins=TRIM_TIME_BINS,
        centered=not args.non_centered,
    )
    coeffs, time_grid, freq_grid = wdm_analysis_coefficients(data, dt, nt, config)
    if gaps:
        # The spline basis takes any time grid, so gap handling is just
        # dropping the corrupted rows: the knots still span the gaps and the
        # posterior widens there.
        keep = good_time_bins(time_grid, t_obs_s, gaps, nt)
        time_grid_all = time_grid  # full grid, for plotting across the gaps
        tg_full_days = time_grid_all * t_obs_s / 86400
        coeffs, time_grid = coeffs[keep], time_grid[keep]
        print(f"[gaps] dropped {np.count_nonzero(~keep)} of {keep.size} time bins")
    print(f"[wdm] grid {coeffs.shape[0]} x {coeffs.shape[1]}, "
          f"f = {freq_grid[0]:.2e}..{freq_grid[-1]:.2e} Hz")

    t0 = time.perf_counter()
    res = fit_log_pspline_surface(
        coeffs[None, :, :], time_grid, warp_freq(freq_grid), config=config,
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        num_chains=args.num_chains, random_seed=0, use_vi=args.use_vi,
    )
    total_s = time.perf_counter() - t0
    diag = summarize_mcmc_diagnostics(res)
    print(f"[fit] wall={total_s:.0f}s sampling={res['nuts_runtime_s']:.0f}s "
          f"div={diag['divergences']} "
          f"rhat(phi)<= {max(diag['phi_time']['r_hat'], diag['phi_freq']['r_hat']):.3f}")

    # WDM-coefficient power -> one-sided PSD in 1/Hz.
    cal = wdm_white_noise_calibration(n_total, dt, nt, config)
    to_psd = 2.0 * dt / cal[None, :]
    S_est = res["psd_mean"] * to_psd
    S_lo = res["psd_lower"] * to_psd
    S_hi = res["psd_upper"] * to_psd
    tg_days = time_grid * n_total * dt / 86400
    fg = freq_grid

    f_w, P_w = welch(data, fs=1 / dt, nperseg=2**18)

    np.savez(
        RESULTS_DIR / f"aet_fullband_{tag}.npz",
        time_grid_days=tg_days, freq_grid=fg,
        psd_mean=S_est, psd_lower=S_lo, psd_upper=S_hi,
        welch_f=f_w, welch_psd=P_w,
        coh_f=coh_f, coh_ae=coh["AE"], coh_at=coh["AT"], coh_et=coh["ET"],
        coh_floor=coh_floor,
        runtime_s=total_s, nuts_runtime_s=res["nuts_runtime_s"],
        divergences=diag["divergences"],
        gaps_s=np.asarray(gaps, dtype=float).reshape(-1, 2),
    )
    with open(RESULTS_DIR / f"aet_fullband_{tag}_diag.json", "w") as fp:
        json.dump(diag, fp, indent=2, default=float)

    # 1) Raw WDM log power vs recovered surface (log-f axis).
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True,
                             sharey=True)
    raw_pow = np.log(res["power"] + 1e-300) + np.log(to_psd)
    if gaps:
        # Raw panel: masked bins as holes rather than letting pcolormesh
        # stretch neighbours across them. Posterior panel: the surface is
        # defined inside the gaps, so evaluate the spline on the full grid.
        raw_full = np.full((keep.size, raw_pow.shape[1]), np.nan)
        raw_full[keep] = raw_pow
        mesh0 = axes[0].pcolormesh(tg_full_days, fg, raw_full.T, shading="auto",
                                   cmap="viridis")
        # Evaluate only inside the kept-bin range: outside it the spline
        # extrapolates without data support.
        in_support = (time_grid_all >= time_grid.min()) & (time_grid_all <= time_grid.max())
        B_t_full = evaluate_bspline_basis(
            time_grid_all[in_support], res["knots_time"], degree=config.degree_time)
        logS_full = (B_t_full @ res["W_mean"] @ res["B_freq"].T) + np.log(to_psd)
        axes[1].pcolormesh(tg_full_days[in_support], fg, logS_full.T,
                           shading="auto", cmap="viridis",
                           vmin=mesh0.get_clim()[0], vmax=mesh0.get_clim()[1])
        for t0_g, t1_g in gaps:
            axes[1].axvspan(t0_g / 86400, t1_g / 86400, color="white", alpha=0.2,
                            lw=0)
    else:
        mesh0 = axes[0].pcolormesh(tg_days, fg, raw_pow.T, shading="auto",
                                   cmap="viridis")
        axes[1].pcolormesh(tg_days, fg, np.log(S_est).T, shading="auto",
                           cmap="viridis", vmin=mesh0.get_clim()[0],
                           vmax=mesh0.get_clim()[1])
    axes[0].set_title("raw WDM log power")
    axes[1].set_title(r"posterior mean $\log \hat S(t,f)$")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("time [days]")
    axes[0].set_ylabel("f [Hz]")
    fig.colorbar(mesh0, ax=axes, shrink=0.85, label=r"$\log S$ [1/Hz]")
    fig.savefig(RESULTS_DIR / f"aet_surface_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Time-averaged spectrum vs Welch across the full band.
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    wb = (f_w >= fg[0]) & (f_w <= fg[-1])
    ax.loglog(f_w[wb], P_w[wb], color="0.6", lw=0.5, label="Welch")
    ax.loglog(fg, S_est.mean(axis=0), color="tab:blue", lw=1.0,
              label="TV fit (time avg)")
    ax.fill_between(fg, S_lo.mean(axis=0), S_hi.mean(axis=0),
                    color="tab:blue", alpha=0.3)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel(r"$S(f)$ [1/Hz]")
    ax.legend(fontsize=8)
    fig.savefig(RESULTS_DIR / f"aet_spectrum_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) Time variation at null flanks vs a mid-band control frequency.
    fig, ax = plt.subplots(figsize=(3.6, 2.4), constrained_layout=True)
    for f_target, lab in ((0.03, "null 0.03 Hz"), (0.06, "null 0.06 Hz"),
                          (0.01, "control 0.01 Hz")):
        j = int(np.argmin(np.abs(fg - f_target)))
        med = np.median(S_est[:, j])
        ax.plot(tg_days, S_est[:, j] / med, label=lab)
        ax.fill_between(tg_days, S_lo[:, j] / med, S_hi[:, j] / med, alpha=0.2)
    for t0, t1 in gaps:
        ax.axvspan(t0 / 86400, t1 / 86400, color="0.85", zorder=0)
    ax.axhline(1.0, color="black", ls=":")
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$\hat S(t,f)/\mathrm{med}_t\,\hat S$")
    ax.legend(fontsize=7)
    fig.savefig(RESULTS_DIR / f"aet_nulldrift_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] figures + npz in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
