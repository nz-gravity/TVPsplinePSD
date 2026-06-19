"""Non-stationary row across all three representations: stationary / WDM / STFT.

One realisation of physically consistent non-stationary LISA TDI data (jaxGB GB
signal + instrument noise + seasonally modulated Galactic confusion) is analysed
three ways, all recovering the SAME injected binary:

  * Stationary baseline -- a time-invariant Whittle spectrum (the traditional
    LISA noise model); biased on non-stationary noise.
  * WDM blocked-Gibbs joint fit -- non-stationary wavelet-domain noise + GB.
  * STFT blocked-Gibbs joint fit -- non-stationary moving-Fourier (R=2,
    phase-retaining) noise + GB.

Each method's recovered GB amplitude is reported as the ratio to its own injected
truth (comparable across representations despite different coefficient units), and
each non-stationary fit's recovered PSD in the GB channel is shown tracking the
true seasonal modulation. Needs the [lisa] extra.
Saves ``notes/figures/lisa_representation_comparison.png``.

    uv run python notes/scripts/make_lisa_representation_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import LISANoiseConfig
from datasets.lisa_tdi import (
    gb_tdi_signal,
    lisa_tdi_noise_psd,
    optimal_snr,
    simulate_tdi_noise,
    simulate_tv_lisa_tdi,
    true_tv_lisa_tdi_psd,
)
from tv_pspline_psd import (
    PSplineConfig,
    run_gibbs_signal_noise_mcmc,
    run_gibbs_stft_signal_noise_mcmc,
    run_stationary_psd_mcmc,
    save_figure,
    wdm_analysis_coefficients,
)
from tv_pspline_psd.stft import moving_stft

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

N, DT, NT = 768, 167.0, 24
NPERSEG = 32  # STFT segment length -> n_seg = N // NPERSEG segments
MOD = LISANoiseConfig(n_modulation_cycles=3.0, modulation_depth=0.85)
CFG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10, trim_low_freq_channels=2)
GBP = dict(f0=1.5e-3, fdot=0.0, A=2.5e-20, ra=1.0, dec=0.3, psi=0.5, iota=0.7, phi0=0.2)


def _wdm(series, ref):
    return wdm_analysis_coefficients(series / ref, DT, NT, CFG)


def _stft(series, ref):
    tg, fg, coeffs = moving_stft(series / ref, DT, nperseg=NPERSEG)
    keep = np.arange(CFG.trim_low_freq_channels, coeffs.shape[2] - CFG.trim_high_freq_channels)
    return coeffs[:, :, keep], tg, fg[keep]


def _ratio(beta_mean, beta_true):
    return float(np.hypot(*beta_mean) / np.hypot(*beta_true))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    noise, _ = simulate_tv_lisa_tdi(N, DT, rng, mod_config=MOD)
    signal = gb_tdi_signal(N, DT, GBP, tdi_gen=2.0)
    data = noise + signal
    ref = float(np.std(data))
    snr = optimal_snr(signal, DT, tdi_gen=2)

    # ---- WDM front end ----
    w_data, tg_w, fg_w = _wdm(data, ref)
    wg1, _, _ = _wdm(gb_tdi_signal(N, DT, {**GBP, "phi0": 0.0}, tdi_gen=2.0), ref)
    wg2, _, _ = _wdm(gb_tdi_signal(N, DT, {**GBP, "phi0": np.pi / 2}, tdi_gen=2.0), ref)
    w_templates = np.stack([wg1, wg2], axis=0)
    w_sig, _, _ = _wdm(signal, ref)
    w_beta_true, *_ = np.linalg.lstsq(
        np.stack([wg1.ravel(), wg2.ravel()], axis=1), w_sig.ravel(), rcond=None)

    cal_rng = np.random.default_rng(99)
    inst_pow = np.mean([_wdm(simulate_tdi_noise(N, DT, cal_rng, tdi_gen=2), ref)[0] ** 2
                        for _ in range(60)], axis=0)
    cal_w = inst_pow.mean(axis=0) / lisa_tdi_noise_psd(fg_w, tdi_gen=2)
    true_psd_w = cal_w[None, :] * true_tv_lisa_tdi_psd(tg_w, fg_w, mod_config=MOD)

    # ---- STFT front end ----
    s_data, tg_s, fg_s = _stft(data, ref)
    sg1, _, _ = _stft(gb_tdi_signal(N, DT, {**GBP, "phi0": 0.0}, tdi_gen=2.0), ref)
    sg2, _, _ = _stft(gb_tdi_signal(N, DT, {**GBP, "phi0": np.pi / 2}, tdi_gen=2.0), ref)
    s_templates = np.stack([sg1, sg2], axis=0)  # (2, 2, n_seg, n_freq)
    s_sig, _, _ = _stft(signal, ref)
    s_beta_true, *_ = np.linalg.lstsq(
        np.stack([sg1.ravel(), sg2.ravel()], axis=1), s_sig.ravel(), rcond=None)
    # Calibrate STFT power to the physical TDI PSD scale the SAME way as WDM:
    # transform ref-normalized colored instrument-noise realizations and average
    # the per-component power per channel (the model fits S = mean_r c_r^2).
    cal_rng_s = np.random.default_rng(123)
    inst_pow_s = np.mean(
        [np.mean(_stft(simulate_tdi_noise(N, DT, cal_rng_s, tdi_gen=2), ref)[0] ** 2, axis=0)
         for _ in range(60)], axis=0)  # (n_seg, n_freq)
    cal_s = inst_pow_s.mean(axis=0) / lisa_tdi_noise_psd(fg_s, tdi_gen=2)
    true_psd_s = cal_s[None, :] * true_tv_lisa_tdi_psd(tg_s, fg_s, mod_config=MOD)

    # ---- Fits ----
    gibbs_w = run_gibbs_signal_noise_mcmc(
        w_data, w_templates, tg_w, fg_w, config=CFG,
        n_sweeps=80, n_burn_sweeps=30, block_warmup=40, block_samples=8, random_seed=1)
    gibbs_s = run_gibbs_stft_signal_noise_mcmc(
        s_data, s_templates, tg_s, fg_s, config=CFG,
        n_sweeps=60, n_burnin_sweeps=20, noise_steps=8, signal_steps=8,
        noise_warmup=120, signal_warmup=60, random_seed=1)
    stationary = run_stationary_psd_mcmc(
        w_data, fg_w, config=CFG, n_warmup=400, n_samples=400, random_seed=1)

    r_w = _ratio(gibbs_w["beta_mean"], w_beta_true)
    r_s = _ratio(gibbs_s["beta_mean"], s_beta_true)
    print(f"injected GB: A={GBP['A']:.2e} optimal SNR={snr:.1f}")
    print(f"WDM-Gibbs  |A|/|A|_true = {r_w:.3f}  div={gibbs_w['divergences']}")
    print(f"STFT-Gibbs |A|/|A|_true = {r_s:.3f}  div={gibbs_s['divergences']}")
    print(f"stationary divergences   = {stationary['divergences']}")

    gb_w = int(np.argmin(np.abs(fg_w - GBP["f0"])))
    gb_s = int(np.argmin(np.abs(fg_s - GBP["f0"])))

    fig, (ax_w, ax_s, ax_a) = plt.subplots(
        1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [2, 2, 1]},
        constrained_layout=True)

    # WDM panel: stationary (biased) vs WDM-Gibbs vs truth.
    ax_w.plot(tg_w, true_psd_w[:, gb_w], color="black", lw=2.0, label="True PSD")
    ax_w.plot(tg_w, stationary["psd_mean_surface"][:, gb_w], color="tab:red", lw=2.0,
              label="Stationary (biased)")
    ax_w.plot(tg_w, gibbs_w["psd_mean"][:, gb_w], color="tab:blue", lw=2.0, label="WDM Gibbs")
    ax_w.fill_between(tg_w, gibbs_w["psd_lower"][:, gb_w], gibbs_w["psd_upper"][:, gb_w],
                      color="tab:blue", alpha=0.15)
    ax_w.set_title(f"WDM noise PSD (f={fg_w[gb_w]*1e3:.2f} mHz)")
    ax_w.set_xlabel("Rescaled time $u$"); ax_w.set_ylabel("Local power"); ax_w.legend(fontsize=8)

    # STFT panel: STFT-Gibbs vs truth (own scale).
    ax_s.plot(tg_s, true_psd_s[:, gb_s], color="black", lw=2.0, label="True PSD")
    ax_s.plot(tg_s, gibbs_s["psd_mean"][:, gb_s], color="tab:green", lw=2.0, label="STFT Gibbs")
    ax_s.fill_between(tg_s, gibbs_s["psd_lower"][:, gb_s], gibbs_s["psd_upper"][:, gb_s],
                      color="tab:green", alpha=0.15)
    ax_s.set_title(f"STFT noise PSD (f={fg_s[gb_s]*1e3:.2f} mHz)")
    ax_s.set_xlabel("Rescaled time $u$"); ax_s.legend(fontsize=8)

    # Amplitude recovery ratio across representations.
    labels = ["WDM\nGibbs", "STFT\nGibbs"]
    ratios = [r_w, r_s]
    ax_a.axhline(1.0, color="black", ls="--", lw=2.0, label="injected")
    ax_a.bar(np.arange(len(labels)), ratios, color=["tab:blue", "tab:green"], alpha=0.85, width=0.6)
    ax_a.set_xticks(np.arange(len(labels))); ax_a.set_xticklabels(labels)
    ax_a.set_ylabel(r"recovered $|A|\,/\,|A|_{\rm true}$")
    ax_a.set_ylim(0.0, 1.4); ax_a.set_title("GB amplitude recovery"); ax_a.legend(fontsize=8)

    save_figure(fig, FIG_DIR / "lisa_representation_comparison.png")
    print(f"Saved figure to {FIG_DIR / 'lisa_representation_comparison.png'}")


if __name__ == "__main__":
    main()
