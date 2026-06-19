"""Joint galactic-binary + non-stationary-noise demo (wavelet-domain global fit).

A galactic binary is injected into the modulated LISA confusion-plus-instrument
noise. We fit the WDM coefficients two ways: (i) noise only, which lets the
signal leak into the PSD estimate, and (ii) jointly, ``w ~ N(a g_c + b g_s, S)``,
which recovers the signal amplitudes and an unbiased noise PSD simultaneously.

Saves ``notes/figures/lisa_signal_*.png``. Requires the project installed.

    python notes/scripts/make_lisa_signal_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import (
    LISANoiseConfig,
    gb_quadratures,
    gb_signal,
    normalization_constant,
    simulate_tv_lisa_noise,
    true_psd_lisa,
    wdm_white_noise_calibration,
)
from wdm_psd import (
    PSplineConfig,
    fit_log_pspline_surface,
    run_joint_signal_noise_mcmc,
    save_figure,
    wdm_analysis_coefficients,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

DT, NT, N_TOTAL = 167.0, 24, 768
LISA = LISANoiseConfig(tobs_key="1yr", n_modulation_cycles=3.0)
PSPLINE = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10,
                        trim_low_freq_channels=2)
GB_F0, GB_FDOT, GB_PHI0, GB_AMP = 1.5e-3, 0.0, 0.7, 1.0  # ~SNR 20 vs normalized noise


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    noise, _ = simulate_tv_lisa_noise(N_TOTAL, dt=DT, rng=rng, config=LISA)
    signal, (a_true, b_true) = gb_signal(N_TOTAL, DT, f0=GB_F0, fdot=GB_FDOT,
                                         amp=GB_AMP, phi0=GB_PHI0)
    data = noise + signal

    w_data, tg, fg = wdm_analysis_coefficients(data, DT, NT, PSPLINE)
    c_t, s_t = gb_quadratures(N_TOTAL, DT, f0=GB_F0, fdot=GB_FDOT)
    g_c, _, _ = wdm_analysis_coefficients(c_t, DT, NT, PSPLINE)
    g_s, _, _ = wdm_analysis_coefficients(s_t, DT, NT, PSPLINE)
    templates = np.stack([g_c, g_s], axis=0)

    # Calibrated true noise PSD on the WDM grid.
    cal = wdm_white_noise_calibration(N_TOTAL, DT, NT, PSPLINE)
    norm = normalization_constant(N_TOTAL, DT, LISA)
    true_noise = cal[None, :] * true_psd_lisa(tg, fg, LISA, norm_ref=norm)

    # (i) noise-only fit on signal-contaminated data; (ii) joint fit.
    noise_only = fit_log_pspline_surface(w_data[None, :, :], tg, fg, config=PSPLINE,
                                         n_warmup=400, n_samples=400, random_seed=1)
    joint = run_joint_signal_noise_mcmc(w_data, templates, tg, fg, config=PSPLINE,
                                        n_warmup=400, n_samples=400, random_seed=1)

    a_hat, b_hat = joint["beta_mean"]
    da, db = joint["beta_std"]
    snr = float(np.sqrt(np.sum((a_true * g_c + b_true * g_s) ** 2 / joint["psd_mean"])))
    gb_chan = int(np.argmin(np.abs(fg - GB_F0)))
    print(f"injected SNR ~ {snr:.1f}, GB channel f = {fg[gb_chan] * 1e3:.2f} mHz")
    print(f"true (a,b)      = ({a_true:.3f}, {b_true:.3f})")
    print(f"recovered (a,b) = ({a_hat:.3f} +/- {da:.3f}, {b_hat:.3f} +/- {db:.3f})")
    print(f"true |A| = {np.hypot(a_true, b_true):.3f}, "
          f"recovered |A| = {np.hypot(a_hat, b_hat):.3f}")
    print(f"divergences: noise-only={noise_only['divergences']} joint={joint['divergences']}")

    # Figure 1: data / recovered signal / residual in the WDM plane.
    recovered_signal = a_hat * g_c + b_hat * g_s
    residual = w_data - recovered_signal
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True, sharey=True)
    for ax, field, title, cmap in [
        (axes[0], np.log(w_data**2 + 1e-12), "Data WDM log power", "viridis"),
        (axes[1], np.log(recovered_signal**2 + 1e-12), "Recovered GB log power", "viridis"),
        (axes[2], np.log(residual**2 + 1e-12), "Residual log power", "viridis"),
    ]:
        mesh = ax.pcolormesh(tg, fg * 1e3, field.T, shading="nearest", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Rescaled time $u$")
        fig.colorbar(mesh, ax=ax)
    axes[0].set_ylabel("Frequency [mHz]")
    save_figure(fig, FIG_DIR / "lisa_signal_decomposition.png")

    # Figure 2: noise PSD in the GB channel -- bias from ignoring the signal.
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(tg, true_noise[:, gb_chan], color="black", lw=2.0, label="True noise PSD")
    ax.plot(tg, noise_only["psd_mean"][:, gb_chan], color="tab:red", lw=2.0,
            label="Noise-only fit (signal ignored)")
    ax.fill_between(tg, noise_only["psd_lower"][:, gb_chan],
                    noise_only["psd_upper"][:, gb_chan], color="tab:red", alpha=0.15)
    ax.plot(tg, joint["psd_mean"][:, gb_chan], color="tab:blue", lw=2.0,
            label="Joint fit (signal modelled)")
    ax.fill_between(tg, joint["psd_lower"][:, gb_chan],
                    joint["psd_upper"][:, gb_chan], color="tab:blue", alpha=0.15)
    ax.set_title(f"Noise PSD in the GB channel (f = {fg[gb_chan] * 1e3:.2f} mHz)")
    ax.set_xlabel("Rescaled time $u$")
    ax.set_ylabel("Local power")
    ax.legend(loc="upper right")
    save_figure(fig, FIG_DIR / "lisa_signal_psd_bias.png")
    print(f"Saved signal-demo figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
