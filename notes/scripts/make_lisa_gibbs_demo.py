"""LISA §5 analysis: blocked-Gibbs joint GB+noise fit vs a stationary baseline.

Physically consistent LISA TDI data (jaxGB signal + lisatools noise, TDI-X gen-2)
with a seasonally modulated Galactic confusion foreground and one injected
Galactic binary. We run:

  * the blocked Gibbs sampler (``run_gibbs_signal_noise_mcmc``) -- a NUTS update of
    the non-stationary noise PSD alternating with a NUTS update of the GB
    amplitudes; and
  * a stationary Whittle baseline (``run_stationary_psd_mcmc``) on the same data.

The figures show (i) the recovered amplitude and (ii) that assuming stationarity
biases the noise PSD, while the non-stationary Gibbs fit tracks the modulated
truth. Needs the [lisa] extra. Saves ``notes/figures/lisa_gibbs_*.png``.

    python notes/scripts/make_lisa_gibbs_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import LISANoiseConfig
from datasets.lisa_tdi import (
    gb_tdi_signal,
    lisa_tdi_noise_psd,
    simulate_tdi_noise,
    simulate_tv_lisa_tdi,
    true_tv_lisa_tdi_psd,
)
from tv_pspline_psd import (
    PSplineConfig,
    run_gibbs_signal_noise_mcmc,
    run_stationary_psd_mcmc,
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

    w_data, tg, fg = _wdm(data, ref)
    g1, _, _ = _wdm(gb_tdi_signal(N, DT, {**GBP, "phi0": 0.0}, tdi_gen=2.0), ref)
    g2, _, _ = _wdm(gb_tdi_signal(N, DT, {**GBP, "phi0": np.pi / 2}, tdi_gen=2.0), ref)
    templates = np.stack([g1, g2], axis=0)
    w_sig, _, _ = _wdm(signal, ref)
    beta_true, *_ = np.linalg.lstsq(
        np.stack([g1.ravel(), g2.ravel()], axis=1), w_sig.ravel(), rcond=None)

    # Per-channel calibration to the physical TDI PSD scale, via instrument noise.
    cal_rng = np.random.default_rng(99)
    inst_pow = np.mean([_wdm(simulate_tdi_noise(N, DT, cal_rng, tdi_gen=2), ref)[0] ** 2
                        for _ in range(60)], axis=0)
    cal = inst_pow.mean(axis=0) / lisa_tdi_noise_psd(fg, tdi_gen=2)
    true_psd = cal[None, :] * true_tv_lisa_tdi_psd(tg, fg, mod_config=MOD)

    # Non-stationary blocked Gibbs joint fit, and the stationary baseline.
    gibbs = run_gibbs_signal_noise_mcmc(
        w_data, templates, tg, fg, config=CFG,
        n_sweeps=80, n_burn_sweeps=30, block_warmup=40, block_samples=8, random_seed=1)
    stationary = run_stationary_psd_mcmc(
        w_data, fg, config=CFG, n_warmup=400, n_samples=400, random_seed=1)

    print(f"true beta      = {np.round(beta_true, 3)}")
    print(f"recovered beta = {np.round(gibbs['beta_mean'], 3)} +/- {np.round(gibbs['beta_std'], 3)}")
    print(f"|A| true={np.hypot(*beta_true):.3f} recovered={np.hypot(*gibbs['beta_mean']):.3f}")
    print(f"divergences: gibbs={gibbs['divergences']} stationary={stationary['divergences']}")

    gb_chan = int(np.argmin(np.abs(fg - GBP["f0"])))

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(tg, true_psd[:, gb_chan], color="black", lw=2.0, label="True noise PSD")
    ax.plot(tg, stationary["psd_mean_surface"][:, gb_chan], color="tab:red", lw=2.0,
            label="Stationary fit")
    ax.fill_between(tg, stationary["psd_lower"][gb_chan], stationary["psd_upper"][gb_chan],
                    color="tab:red", alpha=0.12)
    ax.plot(tg, gibbs["psd_mean"][:, gb_chan], color="tab:blue", lw=2.0,
            label="Non-stationary Gibbs fit")
    ax.fill_between(tg, gibbs["psd_lower"][:, gb_chan], gibbs["psd_upper"][:, gb_chan],
                    color="tab:blue", alpha=0.15)
    ax.set_title(f"TDI noise PSD in the GB channel (f = {fg[gb_chan] * 1e3:.2f} mHz)")
    ax.set_xlabel("Rescaled time $u$"); ax.set_ylabel("Local power"); ax.legend()
    save_figure(fig, FIG_DIR / "lisa_gibbs_psd_bias.png")
    print(f"Saved Gibbs demo figure to {FIG_DIR}")


if __name__ == "__main__":
    main()
