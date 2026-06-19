"""LISA 2x3 grid: noise scenario x representation, sensible fits as expected.

Rows are the two noise scenarios -- stationary and non-stationary (seasonally
modulated Galactic confusion) -- and columns are the three analyses:

  * Frequency domain  -- a stationary Whittle noise model (the traditional
    approach; cannot represent time variation), with the GB amplitudes recovered
    under that fixed spectrum;
  * WDM               -- the non-stationary wavelet-domain blocked-Gibbs joint fit;
  * STFT              -- the non-stationary moving-Fourier (R=2) blocked-Gibbs joint fit.

Each panel shows the recovered noise PSD in the injected binary's channel over
rescaled time against the truth (calibrated to that representation's scale), with
the recovered GB amplitude ratio |A|/|A|_true annotated. The *expected* outcome:
every cell is a sensible fit EXCEPT (non-stationary x frequency domain), where the
stationary model cannot follow the modulation and is biased.

Needs the [lisa] extra. Saves ``notes/figures/lisa_2x3_grid.png``.

    uv run python notes/scripts/make_lisa_2x3_grid.py
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS

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
from tv_pspline_psd.joint import _signal_amplitude_model
from tv_pspline_psd.stft import moving_stft

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

N, DT, NT, NPERSEG = 768, 167.0, 24, 32
CFG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10, trim_low_freq_channels=2)
GBP = dict(f0=1.5e-3, fdot=0.0, A=2.5e-20, ra=1.0, dec=0.3, psi=0.5, iota=0.7, phi0=0.2)
SCENARIOS = {
    "Stationary noise": LISANoiseConfig(n_modulation_cycles=3.0, modulation_depth=0.0),
    "Non-stationary noise": LISANoiseConfig(n_modulation_cycles=3.0, modulation_depth=0.85),
}


def _wdm(series, ref):
    return wdm_analysis_coefficients(series / ref, DT, NT, CFG)


def _stft(series, ref):
    tg, fg, coeffs = moving_stft(series / ref, DT, nperseg=NPERSEG)
    keep = np.arange(CFG.trim_low_freq_channels, coeffs.shape[2] - CFG.trim_high_freq_channels)
    return coeffs[:, :, keep], tg, fg[keep]


def _ratio(beta_mean, beta_true):
    return float(np.hypot(*np.atleast_1d(beta_mean)) / np.hypot(*np.atleast_1d(beta_true)))


def _amplitudes_under_psd(coeffs, templates, log_psd_2d, amp_scale, seed=1):
    """GB quadrature amplitudes under a FIXED noise log-PSD surface."""
    kernel = NUTS(_signal_amplitude_model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=300, num_samples=300, progress_bar=False)
    mcmc.run(random.PRNGKey(seed), jnp.asarray(coeffs), jnp.asarray(templates),
             jnp.asarray(log_psd_2d), float(amp_scale), extra_fields=("diverging",))
    beta = np.asarray(mcmc.get_samples()["beta"])
    return beta.mean(axis=0), int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())


def _freq_domain_joint(coeffs, templates, freq_grid, amp_scale, *, n_iter=4, seed=1):
    """Joint stationary (frequency-domain) fit: alternate a stationary noise-PSD
    update on the signal-subtracted data with a GB-amplitude update under that
    fixed, time-invariant spectrum. Subtracting the signal keeps the stationary
    noise estimate from absorbing the GB, so the ONLY limitation left is the
    stationarity assumption itself (it cannot follow time variation).
    """
    n_time = coeffs.shape[0]
    beta = np.zeros(templates.shape[0])
    div = 0
    stat = None
    for it in range(n_iter):
        resid = coeffs - np.tensordot(beta, templates, axes=1)
        stat = run_stationary_psd_mcmc(resid, freq_grid, config=CFG,
                                       n_warmup=250, n_samples=250, random_seed=seed + it)
        log_psd = np.broadcast_to(np.log(stat["psd_mean"])[None, :], (n_time, freq_grid.size))
        beta, sig_div = _amplitudes_under_psd(coeffs, templates, log_psd, amp_scale, seed=seed + it)
        div += stat["divergences"] + sig_div
    return stat, beta, div


def _fit_scenario(mod):
    """Run all three representations for one noise scenario; return per-cell curves."""
    rng = np.random.default_rng(7)
    noise, _ = simulate_tv_lisa_tdi(N, DT, rng, mod_config=mod)
    signal = gb_tdi_signal(N, DT, GBP, tdi_gen=2.0)
    data = noise + signal
    ref = float(np.std(data))

    # WDM front end + physical-scale calibration.
    w_data, tg_w, fg_w = _wdm(data, ref)
    wg1, _, _ = _wdm(gb_tdi_signal(N, DT, {**GBP, "phi0": 0.0}, tdi_gen=2.0), ref)
    wg2, _, _ = _wdm(gb_tdi_signal(N, DT, {**GBP, "phi0": np.pi / 2}, tdi_gen=2.0), ref)
    w_templates = np.stack([wg1, wg2], axis=0)
    w_sig, _, _ = _wdm(signal, ref)
    w_beta_true, *_ = np.linalg.lstsq(np.stack([wg1.ravel(), wg2.ravel()], axis=1),
                                      w_sig.ravel(), rcond=None)
    cal_rng = np.random.default_rng(99)
    inst_w = np.mean([_wdm(simulate_tdi_noise(N, DT, cal_rng, tdi_gen=2), ref)[0] ** 2
                      for _ in range(60)], axis=0)
    cal_w = inst_w.mean(axis=0) / lisa_tdi_noise_psd(fg_w, tdi_gen=2)
    true_w = cal_w[None, :] * true_tv_lisa_tdi_psd(tg_w, fg_w, mod_config=mod)
    amp_w = float(np.sqrt(np.mean(w_data ** 2)))

    # STFT front end + matching calibration.
    s_data, tg_s, fg_s = _stft(data, ref)
    sg1, _, _ = _stft(gb_tdi_signal(N, DT, {**GBP, "phi0": 0.0}, tdi_gen=2.0), ref)
    sg2, _, _ = _stft(gb_tdi_signal(N, DT, {**GBP, "phi0": np.pi / 2}, tdi_gen=2.0), ref)
    s_templates = np.stack([sg1, sg2], axis=0)
    s_sig, _, _ = _stft(signal, ref)
    s_beta_true, *_ = np.linalg.lstsq(np.stack([sg1.ravel(), sg2.ravel()], axis=1),
                                      s_sig.ravel(), rcond=None)
    cal_rng_s = np.random.default_rng(123)
    inst_s = np.mean([np.mean(_stft(simulate_tdi_noise(N, DT, cal_rng_s, tdi_gen=2), ref)[0] ** 2, axis=0)
                      for _ in range(60)], axis=0)
    cal_s = inst_s.mean(axis=0) / lisa_tdi_noise_psd(fg_s, tdi_gen=2)
    true_s = cal_s[None, :] * true_tv_lisa_tdi_psd(tg_s, fg_s, mod_config=mod)

    gb_w = int(np.argmin(np.abs(fg_w - GBP["f0"])))
    gb_s = int(np.argmin(np.abs(fg_s - GBP["f0"])))

    # --- Fits ---
    stat, f_beta, f_div = _freq_domain_joint(w_data, w_templates, fg_w, amp_w)
    gibbs_w = run_gibbs_signal_noise_mcmc(
        w_data, w_templates, tg_w, fg_w, config=CFG,
        n_sweeps=60, n_burn_sweeps=20, block_warmup=40, block_samples=8, random_seed=1)
    gibbs_s = run_gibbs_stft_signal_noise_mcmc(
        s_data, s_templates, tg_s, fg_s, config=CFG,
        n_sweeps=50, n_burnin_sweeps=15, noise_steps=8, signal_steps=8,
        noise_warmup=120, signal_warmup=60, random_seed=1)

    return {
        "Frequency domain": {
            "u": tg_w, "true": true_w[:, gb_w],
            "mean": stat["psd_mean_surface"][:, gb_w],
            "lo": np.broadcast_to(stat["psd_lower"][gb_w], tg_w.shape),
            "hi": np.broadcast_to(stat["psd_upper"][gb_w], tg_w.shape),
            "ratio": _ratio(f_beta, w_beta_true), "div": stat["divergences"] + f_div,
            "color": "tab:red",
        },
        "WDM": {
            "u": tg_w, "true": true_w[:, gb_w], "mean": gibbs_w["psd_mean"][:, gb_w],
            "lo": gibbs_w["psd_lower"][:, gb_w], "hi": gibbs_w["psd_upper"][:, gb_w],
            "ratio": _ratio(gibbs_w["beta_mean"], w_beta_true), "div": gibbs_w["divergences"],
            "color": "tab:blue",
        },
        "STFT": {
            "u": tg_s, "true": true_s[:, gb_s], "mean": gibbs_s["psd_mean"][:, gb_s],
            "lo": gibbs_s["psd_lower"][:, gb_s], "hi": gibbs_s["psd_upper"][:, gb_s],
            "ratio": _ratio(gibbs_s["beta_mean"], s_beta_true), "div": gibbs_s["divergences"],
            "color": "tab:green",
        },
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    snr = optimal_snr(gb_tdi_signal(N, DT, GBP, tdi_gen=2.0), DT, tdi_gen=2)
    columns = ["Frequency domain", "WDM", "STFT"]
    rows = list(SCENARIOS)

    results = {}
    for row in rows:
        print(f"=== {row} ===")
        results[row] = _fit_scenario(SCENARIOS[row])
        for col in columns:
            c = results[row][col]
            print(f"  {col:<18} |A|/|A|_true={c['ratio']:.3f}  div={c['div']}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for i, row in enumerate(rows):
        for j, col in enumerate(columns):
            ax = axes[i, j]
            c = results[row][col]
            ax.plot(c["u"], c["true"], color="black", lw=2.0, label="True PSD")
            ax.plot(c["u"], c["mean"], color=c["color"], lw=2.0, label="Recovered")
            ax.fill_between(c["u"], c["lo"], c["hi"], color=c["color"], alpha=0.15)
            ax.set_title(f"{col}\n$|A|/|A|_{{\\rm true}}={c['ratio']:.2f}$, div={c['div']}",
                         fontsize=10)
            if i == 1:
                ax.set_xlabel("Rescaled time $u$")
            if j == 0:
                ax.set_ylabel(f"{row}\nLocal power")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"LISA noise PSD in the GB channel (SNR={snr:.0f}): scenario x representation",
                 fontsize=13)
    save_figure(fig, FIG_DIR / "lisa_2x3_grid.png")
    print(f"Saved {FIG_DIR / 'lisa_2x3_grid.png'}")


if __name__ == "__main__":
    main()
