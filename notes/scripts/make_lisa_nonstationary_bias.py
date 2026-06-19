"""LISA non-stationary row: stationary-Whittle GB fit vs WDM blocked Gibbs.

One realisation of physically consistent LISA TDI data (jaxGB galactic-binary
signal + lisatools instrument noise + seasonally modulated Galactic confusion
foreground, TDI-X gen-2). On this *single* dataset we run two analyses that
recover the SAME injected galactic binary, differing only in their noise model:

  1. Non-stationary blocked-Gibbs joint fit (``run_gibbs_signal_noise_mcmc``):
     a NUTS update of the time-varying P-spline noise PSD alternating with a
     NUTS update of the GB quadrature amplitudes. The noise surface is free to
     track the seasonal modulation.

  2. Traditional stationary-Whittle baseline (form (a)): we first fit a single
     time-INVARIANT noise spectrum ``S(f)`` with ``run_stationary_psd_mcmc``
     (the time-averaged Whittle spectrum, the standard LISA noise model), then
     recover the GB quadrature amplitudes under that fixed stationary PSD by
     sampling the linear-amplitude block (``_signal_amplitude_model``) with the
     stationary spectrum broadcast across all time rows. This is exactly the
     standard LISA approach -- a fixed/known stationary PSD, fit the signal
     under it -- and because the ONLY difference from the Gibbs fit is that the
     noise surface is forced time-invariant, it isolates the cost of assuming
     stationarity.

The figure shows, in the GB channel: the true modulated noise PSD, the
stationary fit (flat, biased), and the non-stationary Gibbs fit (tracks the
modulation); plus the recovered |A| under each method against the injected
truth. Needs the [lisa] extra. Saves ``notes/figures/lisa_nonstationary_bias.png``.

    uv run python notes/scripts/make_lisa_nonstationary_bias.py
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
    run_stationary_psd_mcmc,
    save_figure,
    wdm_analysis_coefficients,
)
from tv_pspline_psd.joint import _signal_amplitude_model

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

N, DT, NT = 768, 167.0, 24
MOD = LISANoiseConfig(n_modulation_cycles=3.0, modulation_depth=0.85)
CFG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10, trim_low_freq_channels=2)
GBP = dict(f0=1.5e-3, fdot=0.0, A=2.5e-20, ra=1.0, dec=0.3, psi=0.5, iota=0.7, phi0=0.2)


def _wdm(series, ref):
    coeffs, tg, fg = wdm_analysis_coefficients(series / ref, DT, NT, CFG)
    return coeffs, tg, fg


def _stationary_signal_amplitudes(coeffs, templates, log_psd_1d, n_time, amp_scale,
                                  *, n_warmup=400, n_samples=400, seed=1):
    """Recover GB quadrature amplitudes under a FIXED stationary noise PSD.

    ``log_psd_1d`` is the time-invariant ``log S(f)`` from the stationary fit;
    it is broadcast across all ``n_time`` rows so the Whittle weighting is the
    same in every time bin (the defining assumption of a stationary analysis).
    The signal block is the identical linear-amplitude model used inside the
    Gibbs sampler, so the only difference between the two recoveries is the
    noise model.
    """
    log_psd = jnp.broadcast_to(jnp.asarray(log_psd_1d)[None, :], (n_time, log_psd_1d.size))
    kernel = NUTS(_signal_amplitude_model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, progress_bar=False)
    mcmc.run(random.PRNGKey(seed), jnp.asarray(coeffs), jnp.asarray(templates),
             log_psd, float(amp_scale), extra_fields=("diverging",))
    beta = np.asarray(mcmc.get_samples()["beta"])
    divs = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return beta.mean(axis=0), beta.std(axis=0), divs


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    noise, _ = simulate_tv_lisa_tdi(N, DT, rng, mod_config=MOD)
    signal = gb_tdi_signal(N, DT, GBP, tdi_gen=2.0)
    data = noise + signal
    ref = float(np.std(data))
    snr = optimal_snr(signal, DT, tdi_gen=2)

    w_data, tg, fg = _wdm(data, ref)
    g1, _, _ = _wdm(gb_tdi_signal(N, DT, {**GBP, "phi0": 0.0}, tdi_gen=2.0), ref)
    g2, _, _ = _wdm(gb_tdi_signal(N, DT, {**GBP, "phi0": np.pi / 2}, tdi_gen=2.0), ref)
    templates = np.stack([g1, g2], axis=0)
    w_sig, _, _ = _wdm(signal, ref)
    beta_true, *_ = np.linalg.lstsq(
        np.stack([g1.ravel(), g2.ravel()], axis=1), w_sig.ravel(), rcond=None)
    amp_scale = float(np.sqrt(np.mean(w_data**2)))

    # Per-channel calibration of the WDM power to the physical TDI PSD scale.
    cal_rng = np.random.default_rng(99)
    inst_pow = np.mean([_wdm(simulate_tdi_noise(N, DT, cal_rng, tdi_gen=2), ref)[0] ** 2
                        for _ in range(60)], axis=0)
    cal = inst_pow.mean(axis=0) / lisa_tdi_noise_psd(fg, tdi_gen=2)
    true_psd = cal[None, :] * true_tv_lisa_tdi_psd(tg, fg, mod_config=MOD)

    # (1) Non-stationary blocked-Gibbs joint GB + noise fit.
    gibbs = run_gibbs_signal_noise_mcmc(
        w_data, templates, tg, fg, config=CFG,
        n_sweeps=80, n_burn_sweeps=30, block_warmup=40, block_samples=8, random_seed=1)

    # (2) Stationary baseline: fit a time-invariant S(f), then recover the GB
    #     amplitudes under that fixed stationary PSD.
    stationary = run_stationary_psd_mcmc(
        w_data, fg, config=CFG, n_warmup=400, n_samples=400, random_seed=1)
    stat_log_psd = np.log(stationary["psd_mean"])
    stat_beta_mean, stat_beta_std, stat_sig_divs = _stationary_signal_amplitudes(
        w_data, templates, stat_log_psd, w_data.shape[0], amp_scale, seed=1)

    a_true = float(np.hypot(*beta_true))
    a_gibbs = float(np.hypot(*gibbs["beta_mean"]))
    a_stat = float(np.hypot(*stat_beta_mean))
    stat_divs = stationary["divergences"] + stat_sig_divs

    print(f"injected GB:    A={GBP['A']:.2e}  optimal SNR={snr:.1f}")
    print(f"true beta       = {np.round(beta_true, 3)}   |A|_true     = {a_true:.3f}")
    print(f"Gibbs beta      = {np.round(gibbs['beta_mean'], 3)} +/- "
          f"{np.round(gibbs['beta_std'], 3)}   |A|_gibbs    = {a_gibbs:.3f}")
    print(f"stationary beta = {np.round(stat_beta_mean, 3)} +/- "
          f"{np.round(stat_beta_std, 3)}   |A|_stat     = {a_stat:.3f}")
    a_gibbs_err = float(np.hypot(*gibbs["beta_std"]))
    a_stat_err = float(np.hypot(*stat_beta_std))
    print(f"|A| error: gibbs={100*(a_gibbs/a_true-1):+.1f}%  "
          f"stationary={100*(a_stat/a_true-1):+.1f}%")
    print(f"|A| uncertainty: gibbs={a_gibbs_err:.3f}  stationary={a_stat_err:.3f} "
          f"(stationary is {a_stat_err/a_gibbs_err:.1f}x wider)")
    print(f"divergences: gibbs={gibbs['divergences']} stationary={stat_divs}")

    gb_chan = int(np.argmin(np.abs(fg - GBP["f0"])))

    fig, (ax_psd, ax_amp) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.4, 1]},
        constrained_layout=True)

    ax_psd.plot(tg, true_psd[:, gb_chan], color="black", lw=2.2,
                label="True modulated noise PSD")
    ax_psd.plot(tg, stationary["psd_mean_surface"][:, gb_chan], color="tab:red", lw=2.0,
                label="Stationary fit (biased)")
    ax_psd.fill_between(tg, stationary["psd_lower"][gb_chan], stationary["psd_upper"][gb_chan],
                        color="tab:red", alpha=0.12)
    ax_psd.plot(tg, gibbs["psd_mean"][:, gb_chan], color="tab:blue", lw=2.0,
                label="Non-stationary Gibbs fit")
    ax_psd.fill_between(tg, gibbs["psd_lower"][:, gb_chan], gibbs["psd_upper"][:, gb_chan],
                        color="tab:blue", alpha=0.15)
    ax_psd.set_title(f"TDI noise PSD in the GB channel (f = {fg[gb_chan] * 1e3:.2f} mHz)")
    ax_psd.set_xlabel("Rescaled time $u$")
    ax_psd.set_ylabel("Local power")
    ax_psd.legend(loc="upper right", fontsize=9)

    labels = ["Stationary", "Non-stat.\nGibbs"]
    means = [a_stat, a_gibbs]
    errs = [a_stat_err, a_gibbs_err]
    colors = ["tab:red", "tab:blue"]
    x = np.arange(len(labels))
    ax_amp.axhline(a_true, color="black", lw=2.0, ls="--", label="Injected $|A|$")
    ax_amp.bar(x, means, yerr=errs, color=colors, alpha=0.85, capsize=5, width=0.6)
    ax_amp.set_xticks(x)
    ax_amp.set_xticklabels(labels)
    ax_amp.set_ylim(0.0, max(means) + max(errs) + 0.25)
    ax_amp.set_ylabel("Recovered GB amplitude $|A|$")
    ax_amp.set_title(
        f"GB recovery (SNR = {snr:.0f})\n"
        rf"$\sigma_{{|A|}}$: stationary {a_stat_err:.2f} vs Gibbs {a_gibbs_err:.2f}")
    ax_amp.legend(loc="lower right", fontsize=9)

    save_figure(fig, FIG_DIR / "lisa_nonstationary_bias.png")
    print(f"Saved figure to {FIG_DIR / 'lisa_nonstationary_bias.png'}")


if __name__ == "__main__":
    main()
