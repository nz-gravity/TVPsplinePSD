"""Joint GB + non-stationary-noise fit on physically consistent LISA TDI data.

Signal (jaxGB) and noise (lisatools) share the TDI-X, generation-2 convention,
validated so the injected SNR is physical. A galactic binary is injected into the
instrument-plus-modulated-confusion noise and we recover its amplitude and the
non-stationary noise PSD jointly; a noise-only fit instead absorbs the signal.

Needs the [lisa] extra. Saves ``notes/figures/lisa_tdi_*.png``.

    python notes/scripts/make_lisa_tdi_signal_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import LISANoiseConfig
from datasets.lisa_tdi import (
    gb_tdi_signal,
    lisa_tdi_confusion_psd,
    lisa_tdi_noise_psd,
    simulate_tdi_noise,
    simulate_tv_lisa_tdi,
    true_tv_lisa_tdi_psd,
)
from tv_pspline_psd import (
    PSplineConfig,
    fit_log_pspline_surface,
    run_joint_signal_noise_mcmc,
    save_figure,
    wdm_analysis_coefficients,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

N, DT, NT = 768, 167.0, 24
MOD = LISANoiseConfig(n_modulation_cycles=3.0, modulation_depth=0.85)
CFG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10, trim_low_freq_channels=2)
GBP = dict(f0=1.5e-3, fdot=0.0, A=2.5e-20, ra=1.0, dec=0.3, psi=0.5, iota=0.7, phi0=0.2)


def _wdm(series, ref):
    coeffs, tg, fg = wdm_analysis_coefficients(series / ref, DT, NT, CFG)
    return coeffs, tg, fg


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    noise, _ = simulate_tv_lisa_tdi(N, DT, rng, mod_config=MOD)
    signal = gb_tdi_signal(N, DT, GBP, tdi_gen=2.0)
    data = noise + signal
    ref = float(np.std(data))

    f = np.fft.rfftfreq(N, d=DT)
    s_floor = lisa_tdi_noise_psd(f, tdi_gen=2) + lisa_tdi_confusion_psd(f, tdi_gen=2)
    snr = float(np.sqrt(np.sum(4 * np.abs(DT * np.fft.rfft(signal))[1:] ** 2 / s_floor[1:] / (N * DT))))
    print(f"injected GB SNR (vs instrument+confusion) = {snr:.1f}")

    w_data, tg, fg = _wdm(data, ref)
    # Two templates: jaxGB at two phases span amplitude + phase.
    g1, _, _ = _wdm(gb_tdi_signal(N, DT, {**GBP, "phi0": 0.0}, tdi_gen=2.0), ref)
    g2, _, _ = _wdm(gb_tdi_signal(N, DT, {**GBP, "phi0": np.pi / 2}, tdi_gen=2.0), ref)
    templates = np.stack([g1, g2], axis=0)
    w_sig, _, _ = _wdm(signal, ref)
    beta_true, *_ = np.linalg.lstsq(
        np.stack([g1.ravel(), g2.ravel()], axis=1), w_sig.ravel(), rcond=None
    )

    # Per-channel calibration to the physical TDI PSD scale, via instrument noise.
    cal_rng = np.random.default_rng(99)
    inst_pow = np.mean([_wdm(simulate_tdi_noise(N, DT, cal_rng, tdi_gen=2), ref)[0] ** 2
                        for _ in range(60)], axis=0)
    cal = inst_pow.mean(axis=0) / lisa_tdi_noise_psd(fg, tdi_gen=2)
    true_psd = cal[None, :] * true_tv_lisa_tdi_psd(tg, fg, mod_config=MOD)

    noise_only = fit_log_pspline_surface(w_data[None, :, :], tg, fg, config=CFG,
                                         n_warmup=400, n_samples=400, random_seed=1)
    joint = run_joint_signal_noise_mcmc(w_data, templates, tg, fg, config=CFG,
                                        n_warmup=400, n_samples=400, random_seed=1)

    print(f"true beta      = {np.round(beta_true, 3)}")
    print(f"recovered beta = {np.round(joint['beta_mean'], 3)} +/- {np.round(joint['beta_std'], 3)}")
    print(f"|A| true={np.hypot(*beta_true):.3f} recovered={np.hypot(*joint['beta_mean']):.3f}")
    print(f"divergences: noise-only={noise_only['divergences']} joint={joint['divergences']}")

    gb_chan = int(np.argmin(np.abs(fg - GBP["f0"])))
    recovered = joint["beta_mean"][0] * g1 + joint["beta_mean"][1] * g2

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True, sharey=True)
    for ax, fld, ttl in [(axes[0], w_data, "Data WDM log power"),
                         (axes[1], recovered, "Recovered GB log power"),
                         (axes[2], w_data - recovered, "Residual log power")]:
        mesh = ax.pcolormesh(tg, fg * 1e3, np.log(fld ** 2 + 1e-30).T, shading="nearest", cmap="viridis")
        ax.set_title(ttl); ax.set_xlabel("Rescaled time $u$")
        fig.colorbar(mesh, ax=ax)
    axes[0].set_ylabel("Frequency [mHz]")
    save_figure(fig, FIG_DIR / "lisa_tdi_decomposition.png")

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(tg, true_psd[:, gb_chan], color="black", lw=2.0, label="True noise PSD")
    ax.plot(tg, noise_only["psd_mean"][:, gb_chan], color="tab:red", lw=2.0, label="Noise-only fit")
    ax.plot(tg, joint["psd_mean"][:, gb_chan], color="tab:blue", lw=2.0, label="Joint fit")
    ax.fill_between(tg, joint["psd_lower"][:, gb_chan], joint["psd_upper"][:, gb_chan],
                    color="tab:blue", alpha=0.15)
    ax.set_title(f"TDI noise PSD in the GB channel (f = {fg[gb_chan] * 1e3:.2f} mHz)")
    ax.set_xlabel("Rescaled time $u$"); ax.set_ylabel("Local power"); ax.legend()
    save_figure(fig, FIG_DIR / "lisa_tdi_psd_bias.png")
    print(f"Saved TDI signal-demo figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
