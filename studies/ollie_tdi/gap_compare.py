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
from fit_aet_fullband import (
    DATA_FULL,
    DECIMATE,
    GRID,
    N_KNOTS_LIN,
    N_KNOTS_LOG,
    RESULTS_DIR,
    TRIM_TIME_BINS,
    fft_decimate,
    gate_gaps,
    good_time_bins,
    lisa_like_gaps,
    load_aet,
)
from scipy.signal import welch

from tv_pspline_psd import PSplineConfig, set_paper_style, wdm_analysis_coefficients
from tv_pspline_psd.datasets import wdm_white_noise_calibration

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
    z2_t, z_mat = {}, {}
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
        z_mat[tag] = z
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
    # Whitened coefficients away from the null comb (+-2 mHz excluded): the
    # only whitening blemish is the intentionally smoothed null cores, so this
    # subset should be (and is) indistinguishable from N(0,1). The histogram
    # panel shows the null-excluded z for both fits; full-band numbers are
    # printed for the text.
    nullreg = np.abs(sdn) < 2e-3
    from scipy.stats import kurtosis
    z_hist = {}
    for tag, z in z_mat.items():
        z_hist[tag] = z[:, ~nullreg].ravel()
        print(f"[null-excl] {tag}: {nullreg.sum()}/{fg_u.size} channels "
              f"excluded; kurtosis {kurtosis(z.ravel()):.3f} -> "
              f"{kurtosis(z_hist[tag]):.3f}, std z {z_hist[tag].std():.4f}, "
              f"mean z^2 {np.mean(z_hist[tag]**2):.4f}")
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

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.6), constrained_layout=True)

    # Per-time-bin whitening statistic: quoted in the text, no longer a panel.
    nf = fits["ungapped"]["freq_grid"].size
    band = 3 * np.sqrt(2 / nf)
    for tag in fits:
        y = z2_t[tag]
        print(f"[z2-per-bin] {tag}: range [{y.min():.3f}, {y.max():.3f}], "
              f"3sig band 1 +- {band:.4f}, "
              f"trend {np.polyfit(fits[tag]['time_grid_days'], y, 1)[0] * 30:+.4f}/month")

    ax = axes[0]
    x = np.linspace(-4, 4, 200)
    for tag in fits:
        ax.hist(z_hist[tag], bins=200, range=(-4, 4), density=True,
                histtype="step", lw=1.0, **STYLES[tag], label=tag)
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1,
            label=r"$\mathcal{N}(0,1)$")
    ax.set_xlabel(r"$w/\sqrt{\hat S}$")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)

    # Flank asymmetry A(t) = zbar^2(low side) - zbar^2(high side), each side
    # normalised by its time mean: the pooled residual-level statistic quoted
    # in the text. The drifting nulls give the stationary model an odd
    # (opposite-signed) miscalibration across each null, so A(t) ramps through
    # zero for the stationary spectrum and stays flat for the TV fit;
    # band-pooled statistics cancel this signature entirely.
    for model in ("TV", "stationary"):
        yl, yr = (z2_flank[model][s] for s in ("left", "right"))
        a = yl / yl.mean() - yr / yr.mean()
        print(f"[stationary-compare] {model}: asymmetry range "
              f"{a.min():+.3f} to {a.max():+.3f} "
              f"(left mean z^2 {yl.mean():.3f}, right {yr.mean():.3f})")

    # Middle panel: the same failure shown directly in the data, at the
    # steepest flank of the 0.06 Hz null. Spline-free 3-day chunked Welch
    # power vs the fitted TV band vs the stationary (time-averaged) level.
    tg_days = fits["ungapped"]["time_grid_days"]
    S_lo = fits["ungapped"]["psd_lower"] * cal[None, :] / (2.0 * dt)
    S_hi = fits["ungapped"]["psd_upper"] * cal[None, :] / (2.0 * dt)
    win = (fg_u > 0.055) & (fg_u < 0.065)
    swing = S_tv[:, win].max(axis=0) / S_tv[:, win].min(axis=0)
    jbest = np.where(win)[0][np.argmax(swing)]
    group = np.arange(jbest - 2, jbest + 3)  # 5 flank channels, ~120 uHz
    f_star = fg_u[group].mean()

    fs = 1 / dt
    nchunk = int(3.0 * 86400 * fs)  # 3-day chunks
    chunked = data[: (data.size // nchunk) * nchunk].reshape(-1, nchunk)
    f_c, P_c = welch(chunked, fs, nperseg=nchunk // 4, axis=-1)
    t_c = (np.arange(chunked.shape[0]) + 0.5) * 3.0
    df = fg_u[1] - fg_u[0]
    jc = (f_c >= fg_u[group[0]] - df / 2) & (f_c <= fg_u[group[-1]] + df / 2)
    pc = P_c[:, jc].mean(axis=1)
    rel = 1 / np.sqrt(7 * np.count_nonzero(jc))  # 7 welch segments per chunk

    ax = axes[1]
    med = None
    for tag in fits:
        Sg_t = (fits[tag]["psd_mean"] * cal[None, :] / (2.0 * dt))[:, group].mean(axis=1)
        lo_t = (fits[tag]["psd_lower"] * cal[None, :] / (2.0 * dt))[:, group].mean(axis=1)
        hi_t = (fits[tag]["psd_upper"] * cal[None, :] / (2.0 * dt))[:, group].mean(axis=1)
        if med is None:  # common normalisation: the ungapped time median
            med = np.median(Sg_t)
        tgt = fits[tag]["time_grid_days"]
        ax.fill_between(tgt, lo_t / med, hi_t / med, alpha=0.2, lw=0,
                        **STYLES[tag])
        ax.plot(tgt, Sg_t / med, lw=1.1, **STYLES[tag], label=tag)
    ax.errorbar(t_c, pc / np.median(pc), yerr=rel * pc / np.median(pc),
                fmt="o", ms=3, mfc="none", color="0.25", lw=0.9,
                label="chunked Welch (3 d)")
    # Arm-length prediction of the power trajectory: shift the raw full-run
    # Welch spectrum by the recorded arm breathing, S(t, f) = W(f Lbar(t)/L0),
    # no free parameters (cf. verify_aet_fit).
    W_f, W_S = fits["ungapped"]["welch_f"], fits["ungapped"]["welch_psd"]
    lam_c = np.interp(t_c, t_L, L_bar / L_bar[0])
    pred = np.exp(np.mean([np.interp(np.log(fg_u[j] * lam_c), np.log(W_f[1:]),
                                     np.log(W_S[1:])) for j in group], axis=0))
    ax.plot(t_c, pred / np.median(pred), "--", color="black", lw=0.9,
            label="arm-length prediction")
    Sg = S_tv[:, group].mean(axis=1)
    ax.axhline(Sg.mean() / med, color="tab:red", ls="--", lw=1.2,
               label="stationary PSD")
    r = np.corrcoef(np.log(pc), np.log(np.interp(t_c, tg_days, Sg)))[0, 1]
    print(f"[flank-trajectory] f* = {f_star*1e3:.1f} mHz: fit swing "
          f"x{Sg.max()/Sg.min():.2f}, welch swing x{pc.max()/pc.min():.2f}, "
          f"corr(log welch, log fit) = {r:+.3f}, "
          f"corr(log welch, log pred) = "
          f"{np.corrcoef(np.log(pc), np.log(pred))[0, 1]:+.3f}")
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$S(t,f_\star)\,/\,\mathrm{med}_t\,\hat S$")
    ax.legend(fontsize=6, title=f"$f_\\star = {f_star*1e3:.1f}$ mHz",
              title_fontsize=6)

    # Null tracks with 90% credible bands (per-posterior-draw centroids,
    # saved by fit_aet_fullband). The two nulls are labelled in-plot; the
    # legend carries only the fit/prediction distinction.
    ax = axes[2]
    for f0, key in ((0.06, "null_track_006"), (0.12, "null_track_012")):
        for tag, fit in fits.items():
            tg = fit["time_grid_days"]
            lo, fnull, hi = fit[key]
            lam = np.interp(tg, t_L, L_bar / L_bar[0])
            pred = fnull.mean() * lam.mean() / lam
            r = np.corrcoef(fnull, pred)[0, 1]
            ax.fill_between(tg, 1e6 * (lo - fnull.mean()),
                            1e6 * (hi - fnull.mean()),
                            color=STYLES[tag]["color"], alpha=0.2, lw=0)
            ax.plot(tg, 1e6 * (fnull - fnull.mean()), lw=1.0, **STYLES[tag])
            print(f"[null-track] {tag} @ {f0:.2f} Hz: drift "
                  f"{fnull.max() - fnull.min():.2e} Hz, corr = {r:+.3f}")
            if tag == "ungapped":
                ax.plot(tg, 1e6 * (pred - pred.mean()), color="black",
                        lw=0.8, alpha=0.6)
                ax.annotate(f"{f0:.2f} Hz", (tg[-1], 1e6 * (fnull[-1] - fnull.mean())),
                            textcoords="offset points", xytext=(-2, 8),
                            fontsize=7, ha="right", va="bottom")
    for tag in fits:
        ax.plot([], [], "-", label=tag, **STYLES[tag])
    ax.plot([], [], "-", color="black", lw=0.8, alpha=0.6,
            label="arm-length prediction")
    for t0, t1 in gaps_s:
        ax.axvspan(t0 / 86400, t1 / 86400, color="0.85", lw=0)
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$f_{\rm null} - \langle f_{\rm null}\rangle$ [$\mu$Hz]")
    ax.legend(fontsize=6.5, loc="upper left")

    out = RESULTS_DIR / "tdi_gap_compare.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
