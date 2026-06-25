"""LISA non-stationary demonstration: one annual realisation, all figures.

A single, physically consistent year of LISA TDI-X (generation-2) data is
simulated once and analysed once; every non-stationary LISA figure in the
manuscript is rendered from that one realisation so the figures are mutually
consistent.

The Galactic-confusion foreground is modulated by the *cyclostationary* annual
law of Digman & Cornish (2022) -- ``S(u,f) = S_inst(f) + r(u) S_conf(f)`` with
``r`` the tabulated annual harmonics of LISA's antenna pattern sweeping the
Galaxy (their Table 1, A channel), so the non-stationarity is published physics
rather than an ad-hoc envelope. One resolvable Galactic binary (jaxGB) is
injected at ``f0 = 1.5 mHz``.

We run, on the same data:

  * the WDM blocked-Gibbs joint fit (non-stationary noise PSD + GB amplitudes),
  * the moving-STFT (R=2, phase-retaining) blocked-Gibbs joint fit, and
  * a stationary Whittle baseline (the traditional time-invariant LISA model).

Figures written to ``notes/figures/``:
  lisa_surface_comparison.png, lisa_gibbs_psd_bias.png,
  lisa_representation_comparison.png
and the intermediate arrays are cached to ``studies/results/lisa/lisa_demo.npz``.

Needs the [lisa] extra.

    uv run python notes/scripts/make_lisa_demo.py            # full annual run
    uv run python notes/scripts/make_lisa_demo.py --quick    # fast smoke test
"""

from __future__ import annotations

import argparse
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
    run_tang_dynamic_whittle_mcmc,
    save_figure,
    set_paper_style,
    wdm_analysis_coefficients,
)
from tv_pspline_psd.stft import moving_stft

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
CACHE = Path(__file__).resolve().parents[2] / "studies" / "results" / "lisa" / "lisa_demo.npz"
_YEAR = 365.25 * 86400.0


def make_config(quick: bool):
    """Return ``(N, dt, nt, nperseg, n_years, fits, mc, cal)`` for the chosen mode."""
    if quick:
        # Short fine-cadence window (Nyquist 3 mHz, GB in band) just to exercise
        # the code path; n_years only labels the modulation cycles here.
        nt, nf = 24, 32
        return dict(
            N=nt * nf, dt=167.0, nt=nt, nperseg=32, n_years=2.0, target_snr=30.0,
            wdm_fit=dict(n_sweeps=8, n_burn_sweeps=3, block_warmup=30, block_samples=6),
            stft_fit=dict(n_sweeps=8, n_burnin_sweeps=3, noise_steps=6, signal_steps=6,
                          noise_warmup=80, signal_warmup=50),
            stat_fit=dict(n_warmup=200, n_samples=200),
            dw_fit=dict(m=10, thin=2, n_time_grid=24, n_warmup=150, n_samples=150),
            mc_draws=20, cal_draws=20,
        )
    # Annual production grid: 1 year, Nyquist ~2.5 mHz. The binary SNR is spread
    # across the nt WDM time bins, so a coarser time grid (nt=256) keeps both
    # annual confusion peaks while giving enough per-bin SNR to recover the
    # source cleanly; SNR~200 is a bright, realistic LISA verification binary.
    nt, nf = 256, 616  # N = nt * nf; both even (WDM requirement)
    n_years = 1.0
    return dict(
        N=nt * nf, dt=_YEAR / (nt * nf), nt=nt, nperseg=512, n_years=n_years,
        target_snr=200.0,
        wdm_fit=dict(n_sweeps=70, n_burn_sweeps=25, block_warmup=60, block_samples=8,
                     target_accept_prob=0.95),
        stft_fit=dict(n_sweeps=55, n_burnin_sweeps=18, noise_steps=8, signal_steps=8,
                      noise_warmup=150, signal_warmup=80, target_accept_prob=0.95),
        stat_fit=dict(n_warmup=400, n_samples=400),
        dw_fit=dict(m=150, thin=2, n_time_grid=64, n_warmup=300, n_samples=300),
        mc_draws=80, cal_draws=80,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Fast smoke configuration.")
    parser.add_argument("--render-only", action="store_true",
                        help="Skip the fits; re-render figures from the cached npz.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.render_only:
        render(FIG_DIR, np.load(CACHE))
        return

    cfg = make_config(args.quick)
    N, DT, NT, NPERSEG = cfg["N"], cfg["dt"], cfg["nt"], cfg["nperseg"]
    n_years = cfg["n_years"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    # Physical cyclostationary modulation: Digman & Cornish A-channel annual law.
    mod = LISANoiseConfig(modulation_model="digman_cornish", dc_channel="A",
                          dc_tobs_key="1yr", n_year_cycles=n_years)
    pcfg = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10,
                         trim_low_freq_channels=2)

    # Target a clearly-resolvable but non-trivial binary; scale A to a fixed SNR.
    gbp = dict(f0=1.5e-3, fdot=0.0, A=1.0e-20, ra=1.0, dec=0.3, psi=0.5, iota=0.7, phi0=0.2)
    unit = gb_tdi_signal(N, DT, {**gbp, "A": 1.0}, tdi_gen=2.0)
    snr_unit = optimal_snr(unit, DT, tdi_gen=2)
    target_snr = cfg["target_snr"]
    gbp["A"] = target_snr / snr_unit
    print(f"[config] N={N} dt={DT:.1f}s T_obs={N*DT/_YEAR:.3f}yr nt={NT} nperseg={NPERSEG}")

    rng = np.random.default_rng(args.seed)
    noise, _ = simulate_tv_lisa_tdi(N, DT, rng, mod_config=mod, tobs_years=1.0)
    signal = gb_tdi_signal(N, DT, gbp, tdi_gen=2.0)
    data = noise + signal
    ref = float(np.std(data))
    snr = optimal_snr(signal, DT, tdi_gen=2)
    print(f"[inject] GB f0={gbp['f0']*1e3:.2f}mHz |A|={gbp['A']:.2e} optimal SNR={snr:.1f}")

    # ---------- WDM front end ----------
    def _wdm(series):
        return wdm_analysis_coefficients(series / ref, DT, NT, pcfg)

    g1_ts = gb_tdi_signal(N, DT, {**gbp, "phi0": 0.0}, tdi_gen=2.0)
    g2_ts = gb_tdi_signal(N, DT, {**gbp, "phi0": np.pi / 2}, tdi_gen=2.0)
    w_data, tg_w, fg_w = _wdm(data)
    wg1, _, _ = _wdm(g1_ts)
    wg2, _, _ = _wdm(g2_ts)
    w_templates = np.stack([wg1, wg2], axis=0)
    w_sig, _, _ = _wdm(signal)
    w_beta_true, *_ = np.linalg.lstsq(
        np.stack([wg1.ravel(), wg2.ravel()], axis=1), w_sig.ravel(), rcond=None)

    # Per-channel calibration of the WDM power scale to the physical TDI PSD.
    cal_rng = np.random.default_rng(99)
    inst_pow = np.mean([_wdm(simulate_tdi_noise(N, DT, cal_rng, tdi_gen=2))[0] ** 2
                        for _ in range(cfg["cal_draws"])], axis=0)
    cal_w = inst_pow.mean(axis=0) / lisa_tdi_noise_psd(fg_w, tdi_gen=2)
    true_psd_w = cal_w[None, :] * true_tv_lisa_tdi_psd(tg_w, fg_w, mod_config=mod, tobs_years=1.0)

    # Monte Carlo E[w^2] reference for the modulated noise (denoising target).
    mc_rng = np.random.default_rng(2024)
    ref_w = np.mean([_wdm(simulate_tv_lisa_tdi(N, DT, mc_rng, mod_config=mod, tobs_years=1.0)[0])[0] ** 2
                     for _ in range(cfg["mc_draws"])], axis=0)

    # ---------- STFT front end ----------
    def _stft(series):
        tg, fg, c = moving_stft(series / ref, DT, nperseg=NPERSEG)
        keep = np.arange(pcfg.trim_low_freq_channels, c.shape[2] - pcfg.trim_high_freq_channels)
        return c[:, :, keep], tg, fg[keep]

    s_data, tg_s, fg_s = _stft(data)
    sg1, _, _ = _stft(gb_tdi_signal(N, DT, {**gbp, "phi0": 0.0}, tdi_gen=2.0))
    sg2, _, _ = _stft(gb_tdi_signal(N, DT, {**gbp, "phi0": np.pi / 2}, tdi_gen=2.0))
    s_templates = np.stack([sg1, sg2], axis=0)
    s_sig, _, _ = _stft(signal)
    s_beta_true, *_ = np.linalg.lstsq(
        np.stack([sg1.ravel(), sg2.ravel()], axis=1), s_sig.ravel(), rcond=None)
    cal_rng_s = np.random.default_rng(123)
    inst_pow_s = np.mean(
        [np.mean(_stft(simulate_tdi_noise(N, DT, cal_rng_s, tdi_gen=2))[0] ** 2, axis=0)
         for _ in range(cfg["cal_draws"])], axis=0)
    cal_s = inst_pow_s.mean(axis=0) / lisa_tdi_noise_psd(fg_s, tdi_gen=2)
    true_psd_s = cal_s[None, :] * true_tv_lisa_tdi_psd(tg_s, fg_s, mod_config=mod, tobs_years=1.0)

    # ---------- Fits ----------
    print("[fit] WDM blocked Gibbs ...")
    gibbs_w = run_gibbs_signal_noise_mcmc(
        w_data, w_templates, tg_w, fg_w, config=pcfg, random_seed=1, **cfg["wdm_fit"])
    print("[fit] STFT blocked Gibbs ...")
    gibbs_s = run_gibbs_stft_signal_noise_mcmc(
        s_data, s_templates, tg_s, fg_s, config=pcfg, random_seed=1, **cfg["stft_fit"])
    # Stationary baseline on the SAME signal-subtracted coefficients the Gibbs
    # fit sees, so the only difference is the time-stationarity assumption (no
    # signal contamination in the comparison).
    recovered_w = gibbs_w["beta_mean"][0] * wg1 + gibbs_w["beta_mean"][1] * wg2
    print("[fit] stationary Whittle baseline ...")
    stationary = run_stationary_psd_mcmc(
        w_data - recovered_w, fg_w, config=pcfg, random_seed=1, **cfg["stat_fit"])

    # Dynamic-Whittle (power-only moving periodogram): noise-PSD only, so it runs
    # on the time series with the recovered binary subtracted (it cannot subtract a
    # coherent template itself). It is the power sibling of the STFT front end.
    print("[fit] dynamic Whittle (moving periodogram) ...")
    rec_ts = gibbs_w["beta_mean"][0] * g1_ts + gibbs_w["beta_mean"][1] * g2_ts
    dw = run_tang_dynamic_whittle_mcmc(
        data - rec_ts, dt=DT, config=pcfg, random_seed=1, **cfg["dw_fit"])
    gb_dw = int(np.argmin(np.abs(dw["freq_grid"] - gbp["f0"])))
    dw_slice = dw["psd_mean"][:, gb_dw]
    dw_rel = dw_slice / dw_slice.mean()
    print(f"[result] dynamic-Whittle div={dw['divergences']} "
          f"(GB channel f={dw['freq_grid'][gb_dw]*1e3:.2f} mHz)")

    # Recovered-amplitude posteriors (relative to each representation's truth).
    ratio_w = np.hypot(gibbs_w["beta_samples"][:, 0], gibbs_w["beta_samples"][:, 1]) / np.hypot(*w_beta_true)
    ratio_s = np.hypot(gibbs_s["beta_samples"][:, 0], gibbs_s["beta_samples"][:, 1]) / np.hypot(*s_beta_true)
    r_w, r_s = float(ratio_w.mean()), float(ratio_s.mean())
    print(f"[result] WDM  |A|/|A|_true = {r_w:.3f} +/- {ratio_w.std():.3f}  div={gibbs_w['divergences']}")
    print(f"[result] STFT |A|/|A|_true = {r_s:.3f} +/- {ratio_s.std():.3f}  div={gibbs_s['divergences']}")
    print(f"[result] stationary divergences = {stationary['divergences']}")

    gb_w = int(np.argmin(np.abs(fg_w - gbp["f0"])))
    gb_s = int(np.argmin(np.abs(fg_s - gbp["f0"])))

    np.savez_compressed(
        CACHE,
        tg_w=tg_w, fg_w=fg_w, w_data=w_data, recovered_w=recovered_w,
        true_psd_w=true_psd_w, ref_w=ref_w,
        gibbs_w_psd=gibbs_w["psd_mean"], gibbs_w_lo=gibbs_w["psd_lower"], gibbs_w_hi=gibbs_w["psd_upper"],
        stat_psd=stationary["psd_mean_surface"], stat_lo=stationary["psd_lower"], stat_hi=stationary["psd_upper"],
        tg_s=tg_s, fg_s=fg_s, true_psd_s=true_psd_s,
        gibbs_s_psd=gibbs_s["psd_mean"], gibbs_s_lo=gibbs_s["psd_lower"], gibbs_s_hi=gibbs_s["psd_upper"],
        gb_w=gb_w, gb_s=gb_s, n_years=n_years, r_w=r_w, r_s=r_s, snr=snr,
        ratio_w=ratio_w, ratio_s=ratio_s,
        dw_tg=dw["time_grid"], dw_rel=dw_rel, dw_gb_freq=float(dw["freq_grid"][gb_dw]),
    )
    print(f"[cache] wrote {CACHE}")

    render(FIG_DIR, np.load(CACHE))


def render(fig_dir: Path, d) -> None:
    """Render the three non-stationary LISA figures from cached arrays."""
    set_paper_style()
    tg_w, fg_w = d["tg_w"], d["fg_w"]
    yrs = d["n_years"]
    t_w = tg_w * yrs
    gb_w, gb_s = int(d["gb_w"]), int(d["gb_s"])

    # --- 1. Surface comparison: raw power | posterior mean logS | MC reference ---
    raw = np.log(d["w_data"] ** 2 + 1e-30)
    post = np.log(d["gibbs_w_psd"] + 1e-30)
    refl = np.log(d["ref_w"] + 1e-30)
    vmin = float(np.percentile(np.concatenate([post.ravel(), refl.ravel()]), 2))
    vmax = float(np.percentile(np.concatenate([post.ravel(), refl.ravel()]), 98))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4), constrained_layout=True, sharey=True)
    for ax, fld, ttl in [(axes[0], raw, "Raw WDM power (data)"),
                         (axes[1], post, r"Posterior-mean $\log\hat S(u,f)$"),
                         (axes[2], refl, r"Monte Carlo $\mathbb{E}[w^2]$")]:
        mesh = ax.pcolormesh(t_w, fg_w * 1e3, fld.T, shading="nearest", cmap="viridis",
                             vmin=vmin, vmax=vmax)
        ax.set_title(ttl); ax.set_xlabel("Time [yr]")
    axes[0].set_ylabel("Frequency [mHz]")
    fig.colorbar(mesh, ax=axes, label="log local power", shrink=0.9)
    save_figure(fig, fig_dir / "lisa_surface_comparison.png")

    # --- 2. PSD bias in the GB channel: true / stationary / Gibbs ---
    fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)
    ax.plot(t_w, d["true_psd_w"][:, gb_w], color="black", lw=2.0, label="True noise PSD")
    ax.plot(t_w, d["stat_psd"][:, gb_w], color="tab:red", lw=1.8, label="Stationary fit")
    ax.fill_between(t_w, d["stat_lo"][gb_w], d["stat_hi"][gb_w], color="tab:red", alpha=0.12)
    ax.plot(t_w, d["gibbs_w_psd"][:, gb_w], color="tab:blue", lw=1.8, label="Non-stationary Gibbs fit")
    ax.fill_between(t_w, d["gibbs_w_lo"][:, gb_w], d["gibbs_w_hi"][:, gb_w], color="tab:blue", alpha=0.18)
    ax.set_xlim(t_w.min(), t_w.max())
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel(rf"Local power at $f = {fg_w[gb_w]*1e3:.2f}\,$mHz")
    ax.legend(loc="upper right")
    save_figure(fig, fig_dir / "lisa_gibbs_psd_bias.png")

    # --- 3. Representation comparison: relative modulation across all three
    #         non-stationary front ends (WDM, STFT, dynamic Whittle) | amplitudes ---
    tg_s = d["tg_s"]; t_s = tg_s * yrs; fg_s = d["fg_s"]

    def _rel(slice_):  # normalise a GB-channel slice by its own time-average
        return slice_ / np.mean(slice_)

    fig, (ax_m, ax_a) = plt.subplots(
        1, 2, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [2.4, 1]},
        constrained_layout=True)

    ax_m.plot(t_w, _rel(d["true_psd_w"][:, gb_w]), color="black", lw=2.4, label="True modulation")
    ax_m.axhline(1.0, color="tab:red", lw=1.8, ls="-", label="Stationary fit")
    ax_m.plot(t_w, _rel(d["gibbs_w_psd"][:, gb_w]), color="tab:blue", lw=1.8, label="WDM (Gibbs)")
    ax_m.plot(t_s, _rel(d["gibbs_s_psd"][:, gb_s]), color="tab:green", lw=1.8, label="STFT (Gibbs)")
    ax_m.plot(np.asarray(d["dw_tg"]) * yrs, np.asarray(d["dw_rel"]), color="tab:purple",
              lw=1.8, ls="--", label="Dynamic Whittle")
    ax_m.set_xlim(t_w.min(), t_w.max())
    ax_m.set_ylim(top=ax_m.get_ylim()[1] * 1.30)  # headroom for the legend
    ax_m.set_xlabel("Time [yr]")
    ax_m.set_ylabel(r"Relative noise power $S(u)\,/\,\langle S\rangle$")
    ax_m.set_title(rf"Modulation recovery at $f \approx {fg_w[gb_w]*1e3:.2f}$ mHz")
    ax_m.legend(loc="upper center", ncol=3, fontsize=8, columnspacing=1.2,
                handlelength=1.6, borderaxespad=0.3)

    # Recovered-amplitude posteriors as violins (vs. the injected value).
    ratio_w = np.asarray(d["ratio_w"]); ratio_s = np.asarray(d["ratio_s"])
    parts = ax_a.violinplot([ratio_w, ratio_s], positions=[0, 1], widths=0.7,
                            showmeans=False, showextrema=False, showmedians=True)
    for body, col in zip(parts["bodies"], ("tab:blue", "tab:green")):
        body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.45)
    parts["cmedians"].set_color("black")
    ax_a.axhline(1.0, color="black", ls="--", lw=1.5, label="injected")
    ax_a.set_xticks([0, 1]); ax_a.set_xticklabels(["WDM\nGibbs", "STFT\nGibbs"])
    ax_a.set_ylabel(r"recovered $|A|\,/\,|A|_{\mathrm{true}}$")
    ax_a.set_xlim(-0.6, 1.6); ax_a.set_ylim(0.0, 1.5)
    ax_a.set_title("GB amplitude recovery"); ax_a.legend(loc="lower center")
    save_figure(fig, fig_dir / "lisa_representation_comparison.png")
    print(f"[figures] wrote 3 LISA figures to {fig_dir}")


if __name__ == "__main__":
    main()
