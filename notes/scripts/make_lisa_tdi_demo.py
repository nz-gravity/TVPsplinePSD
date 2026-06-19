"""Realistic LISA TDI demo: joint galactic-binary + non-stationary-noise fit.

Uses matched gen-2 TDI-X conventions throughout (Sec. lisa_tdi): the instrument
noise (lisatools ``XYZ2SensitivityMatrix``) and the seasonally modulated Galactic
confusion are drawn in TDI-X units, and the galactic-binary signal is the jaxGB
TDI-X response at the same generation. Over a full observation year (default; use
``--quick`` for a short toy), a resolvable binary is recovered jointly with the
non-stationary noise PSD; a noise-only fit instead absorbs the signal and biases
the PSD. At year scale the posterior surface is reconstructed from the
eigen-coefficients (``store_log_psd_samples=False``) to bound memory.

Requires the [lisa] extra. Saves ``notes/figures/lisa_tdi_*.png``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import (
    LISANoiseConfig,
    gb_tdi_signal,
    lisa_tdi_confusion_psd,
    lisa_tdi_noise_psd,
    monte_carlo_reference,
    simulate_tv_lisa_tdi,
)
from tv_pspline_psd import (
    PSplineConfig,
    fit_log_pspline_surface,
    run_joint_signal_noise_mcmc,
    save_figure,
    wdm_analysis_coefficients,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
YEAR = 365.25 * 86400.0
TDI_GEN, TOBS_YEARS = 2, 1.0
GB_PARAMS = dict(f0=1.5e-3, fdot=0.0, A=1e-21, ra=1.0, dec=0.3, psi=0.5, iota=0.7, phi0=0.2)


def _snr_total(signal, dt, S_total):
    freq_pos = np.fft.rfftfreq(len(signal), d=dt)
    h = dt * np.fft.rfft(signal)
    df = 1.0 / (len(signal) * dt)
    mask = freq_pos > 0
    return float(np.sqrt(np.sum(4.0 * np.abs(h[mask]) ** 2 / S_total[mask] * df)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="short 1.5-day toy instead of the full year")
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        nt, nf, dt = 24, 32, 167.0
        n_mod_cycles, store, n_draws, warm, samp, target_snr = 3.0, True, 40, 200, 200, 15.0
    else:
        # One observation year: ~1 time bin/day, dt fixed to keep the confusion
        # band (Nyquist ~3 mHz). The surface samples are reconstructed (store=False).
        # A loud, resolvable binary (SNR ~80) so the line and the noise-PSD bias are
        # visible at the fine year-long frequency resolution.
        nt, nf = 366, 500
        dt = YEAR / (nt * nf)
        n_mod_cycles, store, n_draws, warm, samp, target_snr = 1.0, False, 25, 300, 300, 80.0
    N_TOTAL = nt * nf
    NT, DT = nt, dt
    MOD = LISANoiseConfig(tobs_key="1yr", n_modulation_cycles=n_mod_cycles)
    PSPLINE = PSplineConfig(n_interior_knots_time=10, n_interior_knots_freq=12,
                            trim_low_freq_channels=2)
    print(f"duration {N_TOTAL * DT / YEAR:.2f} yr  ({N_TOTAL} samples, dt={DT:.1f}s, "
          f"grid {nt}x{nf})  store_surface={store}")
    t_start = time.time()

    freq = np.fft.rfftfreq(N_TOTAL, d=DT)
    s_inst = lisa_tdi_noise_psd(freq, tdi_gen=TDI_GEN)
    s_conf = lisa_tdi_confusion_psd(freq, tobs_years=TOBS_YEARS, tdi_gen=TDI_GEN)
    s_total = s_inst + s_conf  # <m^2> = 1 stationary-equivalent for SNR/scale
    band = (freq > 3e-4) & (freq < 2.8e-3)
    ref = float(np.sqrt(np.median(s_total[band])))  # normalise data to O(1)

    # Scale the binary amplitude to the target SNR against the total noise.
    unit_signal = gb_tdi_signal(N_TOTAL, DT, {**GB_PARAMS, "A": 1e-21}, tdi_gen=float(TDI_GEN))
    amp = 1e-21 * target_snr / _snr_total(unit_signal, DT, s_total)
    signal = gb_tdi_signal(N_TOTAL, DT, {**GB_PARAMS, "A": amp}, tdi_gen=float(TDI_GEN))
    snr = _snr_total(signal, DT, s_total)

    # Two jaxGB quadrature templates (phi0 = 0, pi/2) span the (amplitude, phase)
    # of the monochromatic binary; we fit the quadrature amplitudes (a, b).
    g_c = gb_tdi_signal(N_TOTAL, DT, {**GB_PARAMS, "A": amp, "phi0": 0.0}, tdi_gen=float(TDI_GEN))
    g_s = gb_tdi_signal(N_TOTAL, DT, {**GB_PARAMS, "A": amp, "phi0": np.pi / 2}, tdi_gen=float(TDI_GEN))
    basis = np.stack([g_c, g_s], axis=1)
    ab_true, *_ = np.linalg.lstsq(basis, signal, rcond=None)
    proj_resid = np.linalg.norm(signal - basis @ ab_true) / np.linalg.norm(signal)
    print(f"quadrature projection residual: {proj_resid:.3e} (span valid if small)")

    rng = np.random.default_rng(7)
    noise, _ = simulate_tv_lisa_tdi(N_TOTAL, DT, rng, mod_config=MOD,
                                    tobs_years=TOBS_YEARS, tdi_gen=TDI_GEN)
    data = noise + signal

    w_data, tg, fg = wdm_analysis_coefficients(data / ref, DT, NT, PSPLINE)
    g_c_w, _, _ = wdm_analysis_coefficients(g_c / ref, DT, NT, PSPLINE)
    g_s_w, _, _ = wdm_analysis_coefficients(g_s / ref, DT, NT, PSPLINE)
    templates = np.stack([g_c_w, g_s_w], axis=0)  # fit quadrature amplitudes (a, b)

    # Monte Carlo true noise E[w^2] (no signal), same normalisation.
    mc_ref = monte_carlo_reference(
        lambda r: simulate_tv_lisa_tdi(N_TOTAL, DT, r, mod_config=MOD,
                                       tobs_years=TOBS_YEARS, tdi_gen=TDI_GEN)[0] / ref,
        n_draws=n_draws, n_total=N_TOTAL, dt=DT, nt=NT, config=PSPLINE, seed=321,
    )

    noise_only = fit_log_pspline_surface(w_data[None, :, :], tg, fg, config=PSPLINE,
                                         n_warmup=warm, n_samples=samp, random_seed=1,
                                         store_log_psd_samples=store)
    joint = run_joint_signal_noise_mcmc(w_data, templates, tg, fg, config=PSPLINE,
                                        n_warmup=warm, n_samples=samp, random_seed=1,
                                        store_log_psd_samples=store)

    a_hat, b_hat = joint["beta_mean"]
    amp_true = float(np.hypot(*ab_true))
    amp_hat = float(np.hypot(a_hat, b_hat))
    gb_chan = int(np.argmin(np.abs(fg - GB_PARAMS["f0"])))
    print(f"GB at {fg[gb_chan] * 1e3:.2f} mHz, total-noise SNR = {snr:.1f}")
    print(f"instrument vs confusion at GB: {np.interp(GB_PARAMS['f0'], freq, s_inst):.2e}"
          f" vs {np.interp(GB_PARAMS['f0'], freq, s_conf):.2e}")
    print(f"true (a,b)      = ({ab_true[0]:.3f}, {ab_true[1]:.3f})  |A|={amp_true:.3f}")
    print(f"recovered (a,b) = ({a_hat:.3f}, {b_hat:.3f})  |A|={amp_hat:.3f}  "
          f"(ratio {amp_hat / amp_true:.3f})")
    print(f"divergences: noise-only={noise_only['divergences']} joint={joint['divergences']}")

    # Figure 1: data / recovered signal / residual in the WDM plane.
    recovered = a_hat * g_c_w + b_hat * g_s_w
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True, sharey=True)
    for ax, field, title in [
        (axes[0], np.log(w_data**2 + 1e-12), "Data WDM log power"),
        (axes[1], np.log(recovered**2 + 1e-12), "Recovered GB log power"),
        (axes[2], np.log((w_data - recovered) ** 2 + 1e-12), "Residual log power"),
    ]:
        mesh = ax.pcolormesh(tg, fg * 1e3, field.T, shading="nearest", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Rescaled time $u$")
        fig.colorbar(mesh, ax=ax)
    axes[0].set_ylabel("Frequency [mHz]")
    save_figure(fig, FIG_DIR / "lisa_tdi_decomposition.png")

    # Figure 2: noise PSD bias in the GB channel.
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(tg, mc_ref[:, gb_chan], color="black", lw=2.0, label="True noise $E[w^2]$ (MC)")
    ax.plot(tg, noise_only["psd_mean"][:, gb_chan], color="tab:red", lw=2.0,
            label="Noise-only fit (signal ignored)")
    ax.fill_between(tg, noise_only["psd_lower"][:, gb_chan],
                    noise_only["psd_upper"][:, gb_chan], color="tab:red", alpha=0.15)
    ax.plot(tg, joint["psd_mean"][:, gb_chan], color="tab:blue", lw=2.0,
            label="Joint fit (signal modelled)")
    ax.fill_between(tg, joint["psd_lower"][:, gb_chan],
                    joint["psd_upper"][:, gb_chan], color="tab:blue", alpha=0.15)
    ax.set_title(f"TDI noise PSD in the binary's channel (f = {fg[gb_chan] * 1e3:.2f} mHz, "
                 f"SNR {snr:.0f})")
    ax.set_xlabel("Rescaled time $u$")
    ax.set_ylabel("Local power (normalized TDI)")
    ax.legend(loc="upper right")
    save_figure(fig, FIG_DIR / "lisa_tdi_psd_bias.png")
    print(f"Saved LISA TDI figures to {FIG_DIR}  (total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
