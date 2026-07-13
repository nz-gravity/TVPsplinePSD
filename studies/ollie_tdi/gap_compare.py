"""Gapped vs ungapped whitening + null-tracking figure for the paper.

Post-processes the two existing full-band fits (aet_fullband_A_full.npz and
aet_fullband_A_full_gaps.npz; no refit): pooled whitened-coefficient histogram,
per-time-bin whitening statistic z-bar^2, fitted null trajectories against
the parameter-free 1/Lbar(t) prediction, and a stationary-model comparison:
whitening the same data with the time-averaged spectrum leaves opposite-signed
z-bar^2 trends on the two sides of each drifting null (they cancel when pooled
over the band, so the flank sides must be separated to expose the failure).

Run after fit_aet_fullband.py (with and without --gaps):
    python studies/ollie_tdi/gap_compare.py
"""

from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tv_pspline_psd import PSplineConfig, set_paper_style, wdm_analysis_coefficients
from datasets import wdm_white_noise_calibration

from fit_aet_fullband import (
    DATA_FULL, DECIMATE, GRID, N_KNOTS_LIN, N_KNOTS_LOG, RESULTS_DIR,
    TRIM_TIME_BINS, fft_decimate, gate_gaps, good_time_bins, lisa_like_gaps,
    load_aet,
)

set_paper_style()

STYLES = {"ungapped": dict(color="tab:blue"), "gapped": dict(color="tab:orange")}


def null_track(tg_days, fg, S_est, f0, half_width=0.004):
    """Depth-weighted valley centroid of the null near f0 (cf. verify_aet_fit)."""
    win = (fg > f0 - half_width) & (fg < f0 + half_width)
    logS_win = np.log(S_est[:, win])
    depth = np.maximum(0.0, np.percentile(logS_win, 75, axis=1,
                                          keepdims=True) - logS_win)
    return (fg[win] * depth).sum(axis=1) / depth.sum(axis=1)


def main() -> None:
    channel = "A"
    fits = {tag: np.load(RESULTS_DIR / f"aet_fullband_{channel}_full{sfx}.npz")
            for tag, sfx in (("ungapped", ""), ("gapped", "_gaps"))}
    gaps_s = fits["gapped"]["gaps_s"]

    aet, dt_raw = load_aet("full")
    data = fft_decimate(aet[channel], DECIMATE)
    dt = dt_raw * DECIMATE
    t_obs_s = data.size * dt
    nt, trim_low = GRID["full"]
    config = PSplineConfig(
        n_interior_knots_freq=N_KNOTS_LOG + N_KNOTS_LIN,
        trim_low_freq_channels=trim_low, trim_time_bins=TRIM_TIME_BINS,
    )
    cal = wdm_white_noise_calibration(data.size, dt, nt, config)

    # Whitened coefficients per fit: the gapped fit whitens the gated data on
    # the kept bins, exactly as fitted (same schedule: lisa_like_gaps default).
    z2_t, z_pooled = {}, {}
    for tag, fit in fits.items():
        if tag == "gapped":
            gaps = [tuple(g) for g in gaps_s]
            assert np.allclose(lisa_like_gaps(t_obs_s), gaps_s), \
                "stored gap schedule does not match lisa_like_gaps default"
            coeffs, time_grid, _ = wdm_analysis_coefficients(
                gate_gaps(data, dt, gaps), dt, nt, config)
            keep = good_time_bins(time_grid, t_obs_s, gaps, nt)
            coeffs = coeffs[keep]
        else:
            coeffs, _, _ = wdm_analysis_coefficients(data, dt, nt, config)
        S_wdm = fit["psd_mean"] * cal[None, :] / (2.0 * dt)
        z = coeffs / np.sqrt(S_wdm)
        z_pooled[tag] = z.reshape(-1)
        z2_t[tag] = np.mean(z**2, axis=1)
        print(f"[whiten] {tag}: mean z^2 = {np.mean(z**2):.4f}, "
              f"std z = {np.std(z):.4f}")

    # Stationary comparison (ungapped data): whiten with the time-averaged
    # fitted spectrum -- a stationary model of identical frequency resolution.
    # The failure lives on the null flanks and is opposite-signed on the two
    # sides (the null drifts up, so fixed-f power falls on the left flank and
    # rises on the right), cancelling in band-pooled statistics; separate them.
    fg_u = fits["ungapped"]["freq_grid"]
    coeffs_u, _, _ = wdm_analysis_coefficients(data, dt, nt, config)
    S_tv = fits["ungapped"]["psd_mean"] * cal[None, :] / (2.0 * dt)
    nulls = np.arange(0.03, fg_u.max(), 0.03)
    sd = fg_u[:, None] - nulls[None, :]
    sdn = sd[np.arange(fg_u.size), np.abs(sd).argmin(axis=1)]
    flanks = {"left": (sdn > -2e-3) & (sdn < -0.3e-3),
              "right": (sdn > 0.3e-3) & (sdn < 2e-3)}
    z2_flank = {
        model: {side: (coeffs_u[:, m] ** 2 / S[:, m]).mean(axis=1)
                for side, m in flanks.items()}
        for model, S in (("TV", S_tv),
                         ("stationary", S_tv.mean(axis=0, keepdims=True)))
    }

    # Armlength prediction for the null tracks.
    with h5py.File(DATA_FULL) as h:
        ltts = np.stack([h[f"ltts/ltt_{k}"][:] for k in
                         ("12", "13", "21", "23", "31", "32")])
    L_bar = ltts.mean(axis=0)
    tg_u = fits["ungapped"]["time_grid_days"]
    t_L = np.linspace(0, tg_u[-1] + tg_u[0], L_bar.size)

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.6), constrained_layout=True)

    ax = axes[0]
    x = np.linspace(-4, 4, 200)
    for tag in fits:
        ax.hist(z_pooled[tag], bins=200, range=(-4, 4), density=True,
                histtype="step", lw=1.0, **STYLES[tag], label=tag)
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1,
            label=r"$\mathcal{N}(0,1)$")
    ax.set_xlabel(r"$w/\sqrt{\hat S}$")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)

    ax = axes[1]
    nf = fits["ungapped"]["freq_grid"].size
    ax.axhspan(*(1 + 3 * np.sqrt(2 / nf) * np.array([-1, 1])), color="0.92",
               lw=0, label=r"$3\sigma$ $\chi^2$ band")
    for tag, fit in fits.items():
        ax.plot(fit["time_grid_days"], z2_t[tag], ".", ms=3, alpha=0.8,
                **STYLES[tag], label=tag)
    ax.axhline(1.0, color="black", ls=":", lw=0.7)
    for t0, t1 in gaps_s:
        ax.axvspan(t0 / 86400, t1 / 86400, color="0.8", alpha=0.5, lw=0)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$\overline{z^2}$ per time bin")

    # Flank asymmetry: A(t) = zbar^2(low side) - zbar^2(high side), each side
    # normalised by its time mean. The drifting nulls give the stationary
    # model an odd (opposite-signed) miscalibration across each null, so A(t)
    # ramps through zero for the stationary spectrum and stays flat for the
    # TV fit; band-pooled statistics cancel this signature entirely.
    ax = axes[2]
    tg_days = fits["ungapped"]["time_grid_days"]
    n_flank = flanks["left"].sum()
    block = 8  # ~2-day block means: 120 bins x 5.6 h over the month
    nblk = tg_days.size // block
    blk = lambda y: y[: nblk * block].reshape(nblk, block).mean(axis=1)
    ax.axhspan(*(2 / np.sqrt(n_flank * block) * 3 * np.array([-1, 1])),
               color="0.92", lw=0, label=r"$3\sigma$ $\chi^2$ band")
    for model, color in (("TV", "tab:blue"), ("stationary", "tab:red")):
        yl, yr = (z2_flank[model][s] for s in ("left", "right"))
        a = yl / yl.mean() - yr / yr.mean()
        ax.plot(blk(tg_days), blk(a), "o-", color=color, ms=2.5, lw=1.0,
                label=f"{model} PSD")
        print(f"[stationary-compare] {model}: asymmetry range "
              f"{a.min():+.3f} to {a.max():+.3f} "
              f"(left mean z^2 {yl.mean():.3f}, right {yr.mean():.3f})")
    ax.axhline(0.0, color="black", ls=":", lw=0.7)
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"flank $\overline{z^2}$ asymmetry")
    ax.legend(fontsize=6.5)

    ax = axes[3]
    for f0, ls in ((0.06, "-"), (0.12, "-.")):
        for tag, fit in fits.items():
            tg, fg = fit["time_grid_days"], fit["freq_grid"]
            fnull = null_track(tg, fg, fit["psd_mean"], f0)
            lam = np.interp(tg, t_L, L_bar / L_bar[0])
            pred = fnull.mean() * lam.mean() / lam
            r = np.corrcoef(fnull, pred)[0, 1]
            ax.plot(tg, 1e6 * (fnull - fnull.mean()), ls, lw=1.0, **STYLES[tag])
            print(f"[null-track] {tag} @ {f0:.2f} Hz: drift "
                  f"{fnull.max() - fnull.min():.2e} Hz, corr = {r:+.3f}")
            if tag == "ungapped":
                ax.plot(tg, 1e6 * (pred - pred.mean()), ls, color="black",
                        lw=0.8, alpha=0.6)
    for tag in fits:
        ax.plot([], [], "-", label=tag, **STYLES[tag])
    ax.plot([], [], "-", color="black", lw=0.8, alpha=0.6,
            label=r"$\propto 1/\bar L(t)$")
    ax.plot([], [], "-", color="0.5", label="0.06 Hz")
    ax.plot([], [], "-.", color="0.5", label="0.12 Hz")
    for t0, t1 in gaps_s:
        ax.axvspan(t0 / 86400, t1 / 86400, color="0.85", lw=0)
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$f_{\rm null} - \langle f_{\rm null}\rangle$ [$\mu$Hz]")
    ax.legend(fontsize=6.5, ncol=2)

    out = RESULTS_DIR / "tdi_gap_compare.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
