"""Three independent checks of the full-band AET TV-PSD fit.

1. Chunked Welch: per-chunk PSD at the null-flank frequencies vs the fitted
   S(t, f) drift curves — empirical, spline-free confirmation of the drift.
2. Armlength prediction: the TDI null comb scales as f ~ 1/Lbar(t), so
   S(t, f) ~ Sbar(f * Lbar(t)/Lbar_0). The light travel times in the file
   predict the flank drift with no free parameters.
3. Whitening: if S-hat is right, w / sqrt(S-hat(t, f)) ~ N(0, 1) across all
   cells (PIT-style histogram and per-time-bin reduced chi^2).

Run after fit_aet_fullband.py:
    python studies/ollie_tdi/verify_aet_fit.py                # A_full_pilot
    python studies/ollie_tdi/verify_aet_fit.py --tag A_full
"""

from __future__ import annotations

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from tv_pspline_psd import PSplineConfig, set_paper_style, wdm_analysis_coefficients
from datasets import wdm_white_noise_calibration

from fit_aet_fullband import (
    DATA_FULL, GRID, N_KNOTS_LIN, N_KNOTS_LOG, RESULTS_DIR,
    fft_decimate, load_aet, DECIMATE,
)

set_paper_style()

FLANKS_HZ = (0.30, 0.72)
CONTROL_HZ = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="A_full_pilot")
    args = parser.parse_args()
    channel = args.tag.split("_")[0]

    fit = np.load(RESULTS_DIR / f"aet_fullband_{args.tag}.npz")
    tg_days, fg = fit["time_grid_days"], fit["freq_grid"]
    S_est, S_lo, S_hi = fit["psd_mean"], fit["psd_lower"], fit["psd_upper"]

    aet, dt_raw = load_aet("full")
    data = fft_decimate(aet[channel], DECIMATE)
    dt = dt_raw * DECIMATE
    fs = 1 / dt

    # --- 1) Chunked Welch at the flank/control frequencies. ---
    chunk_days = 3.0
    nchunk = int(chunk_days * 86400 * fs)
    chunks = data[: (data.size // nchunk) * nchunk].reshape(-1, nchunk)
    f_c, P_c = welch(chunks, fs, nperseg=nchunk // 4, axis=-1)
    t_c = (np.arange(chunks.shape[0]) + 0.5) * chunk_days

    # --- 2) Armlength-scaling prediction. ---
    with h5py.File(DATA_FULL) as h:
        ltts = np.stack([h[f"ltts/ltt_{k}"][:] for k in
                         ("12", "13", "21", "23", "31", "32")])
    L_bar = ltts.mean(axis=0)
    t_L = np.linspace(0, tg_days[-1] + tg_days[0], L_bar.size)
    # Fractional armlength change, interpolated to the fit's time grid.
    lam_t = np.interp(tg_days, t_L, L_bar / L_bar[0])
    S_ref = S_est.mean(axis=0)

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    colors = dict(zip(FLANKS_HZ + (CONTROL_HZ,), ("tab:orange", "tab:blue", "tab:green")))
    for f_t in FLANKS_HZ + (CONTROL_HZ,):
        j = int(np.argmin(np.abs(fg - f_t)))
        med = np.median(S_est[:, j])
        c = colors[f_t]
        ax.plot(tg_days, S_est[:, j] / med, color=c, label=f"fit {fg[j]:.2f} Hz")
        ax.fill_between(tg_days, S_lo[:, j] / med, S_hi[:, j] / med,
                        color=c, alpha=0.2)
        # chunked Welch (averaged over the fit channel's WDM bandwidth)
        jc = np.abs(f_c - fg[j]) <= (fg[1] - fg[0]) / 2
        pc = P_c[:, jc].mean(axis=1)
        ax.plot(t_c, pc / np.median(pc), "o", ms=3, color=c, mfc="none")
        # armlength prediction: S(t, f) = Sbar(f * L(t)/L0), no free parameters
        pred = np.exp(np.interp(np.log(fg[j] * lam_t),
                                np.log(fg), np.log(S_ref)))
        ax.plot(tg_days, pred / np.median(pred), "--", color=c, lw=0.9)
    ax.plot([], [], "o", ms=3, color="0.3", mfc="none", label="chunked Welch")
    ax.plot([], [], "--", color="0.3", lw=0.9, label="armlength prediction")
    ax.axhline(1.0, color="black", ls=":", lw=0.7)
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$S(t,f)/\mathrm{med}_t\,S$")
    ax.legend(fontsize=6.5, ncol=2)
    fig.savefig(RESULTS_DIR / f"verify_drift_{args.tag}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # --- 3) Whitening of the WDM coefficients by the fitted surface. ---
    nt, trim_low = GRID["full"]
    config = PSplineConfig(
        n_interior_knots_freq=N_KNOTS_LOG + N_KNOTS_LIN,
        trim_low_freq_channels=trim_low, trim_time_bins=2,
    )
    coeffs, tg_w, fg_w = wdm_analysis_coefficients(data, dt, nt, config)
    assert coeffs.shape == S_est.shape, "fit tag does not match this data/config"
    cal = wdm_white_noise_calibration(data.size, dt, nt, config)
    S_wdm = S_est * cal[None, :] / (2.0 * dt)  # back to WDM-coefficient units
    z = coeffs / np.sqrt(S_wdm)

    chi2_t = np.mean(z**2, axis=1)  # reduced chi^2 per time bin (nf dof)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), constrained_layout=True)
    x = np.linspace(-4, 4, 200)
    axes[0].hist(z.reshape(-1), bins=200, range=(-4, 4), density=True,
                 color="tab:blue", alpha=0.7)
    axes[0].plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1)
    axes[0].set_xlabel(r"$w/\sqrt{\hat S}$")
    axes[0].set_ylabel("density")
    axes[1].plot(tg_days, chi2_t, color="tab:blue")
    nf = z.shape[1]
    band = 1 + 3 * np.sqrt(2 / nf) * np.array([-1, 1])
    axes[1].axhline(1.0, color="black", ls=":")
    for b in band:
        axes[1].axhline(b, color="0.6", ls=":", lw=0.7)
    axes[1].set_xlabel("time [days]")
    axes[1].set_ylabel(r"$\overline{z^2}$ per time bin")
    fig.savefig(RESULTS_DIR / f"verify_whitening_{args.tag}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    print(f"[whiten] mean z^2 = {np.mean(z**2):.4f} (target 1), "
          f"std z = {np.std(z):.4f}, "
          f"time bins outside 3-sigma band: "
          f"{np.count_nonzero((chi2_t < band[0]) | (chi2_t > band[1]))}/{chi2_t.size}")
    print(f"[out] verify_drift_{args.tag}.png, verify_whitening_{args.tag}.png")


if __name__ == "__main__":
    main()
