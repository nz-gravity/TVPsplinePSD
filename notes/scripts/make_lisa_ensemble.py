"""LISA A/E ensemble: amplitude-recovery distribution and PSD coverage.

Turns the single-realization LISA demonstration into a statistical result. For an
ensemble of independent noise realizations of one physical year of A and E TDI
data -- instrument noise plus the cyclostationary Galactic foreground modulated by
the channel-specific Digman & Cornish (2022) annual law -- we run the
multichannel joint fit (per-channel non-stationary noise PSD with one shared
Galactic-binary amplitude) and record, per realization:

  * the recovered binary amplitude |A|/|A|_true, and
  * the empirical coverage of the 90% PSD credible band (fraction of the
    time-frequency grid where the true S(u,f) lies inside the band), per channel.

Calibration, templates, the true surfaces, and the injected amplitude are
realization-independent and computed once; only the noise draw changes. Results
are saved incrementally to ``studies/results/lisa/lisa_ensemble.npz`` (survives
interruption) and rendered by ``--render-only``.

Needs the [lisa] extra.

    uv run python notes/scripts/make_lisa_ensemble.py            # full ensemble
    uv run python notes/scripts/make_lisa_ensemble.py --quick    # fast smoke test
    uv run python notes/scripts/make_lisa_ensemble.py --render-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import LISANoiseConfig
from datasets.lisa_tdi import (
    gb_ae_signal,
    optimal_snr_ae,
    simulate_tv_ae_tdi,
    true_tv_ae_tdi_psd,
)
from tv_pspline_psd import (
    PSplineConfig,
    run_multichannel_joint_mcmc,
    save_figure,
    set_paper_style,
    wdm_analysis_coefficients,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
CACHE = Path(__file__).resolve().parents[2] / "studies" / "results" / "lisa" / "lisa_ensemble.npz"
_YEAR = 365.25 * 86400.0
_CHANNELS = ("A", "E")


def make_config(quick: bool) -> dict:
    if quick:
        nt, nf = 24, 32
        return dict(N=nt * nf, dt=167.0, nt=nt, n_years=2.0, n_real=3, target_snr=30.0,
                    fit=dict(n_warmup=150, n_samples=150), cal_draws=15)
    nt, nf = 256, 616
    return dict(N=nt * nf, dt=_YEAR / (nt * nf), nt=nt, n_years=1.0, n_real=12,
                target_snr=200.0, fit=dict(n_warmup=300, n_samples=250), cal_draws=60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--n-real", type=int, default=None, help="Override realization count.")
    args = parser.parse_args()

    if args.render_only:
        render(FIG_DIR, np.load(CACHE))
        return

    cfg = make_config(args.quick)
    N, DT, NT, n_years = cfg["N"], cfg["dt"], cfg["nt"], cfg["n_years"]
    n_real = args.n_real or cfg["n_real"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    mod = {c: LISANoiseConfig(modulation_model="digman_cornish", dc_channel=c,
                              dc_tobs_key="1yr", n_year_cycles=n_years) for c in _CHANNELS}
    pcfg = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10,
                         trim_low_freq_channels=2)
    gbp = dict(f0=1.5e-3, fdot=0.0, A=1.0, ra=1.0, dec=0.3, psi=0.5, iota=0.7, phi0=0.2)

    # Injected amplitude for the target combined A+E SNR (realization-independent).
    uA = gb_ae_signal(N, DT, gbp, channel="A")
    uE = gb_ae_signal(N, DT, gbp, channel="E")
    gbp["A"] = cfg["target_snr"] / optimal_snr_ae(uA, uE, DT)
    sig = {c: gb_ae_signal(N, DT, gbp, channel=c) for c in _CHANNELS}
    print(f"[config] N={N} dt={DT:.1f}s T_obs={N*DT/_YEAR:.3f}yr nt={NT} "
          f"n_real={n_real} SNR={cfg['target_snr']:.0f}")

    # Fixed normalisation so templates/calibration are computed once; the |A| ratio
    # is dimensionless so the choice of ref cancels.
    ref_rng = np.random.default_rng(0)
    ref_noise = {c: simulate_tv_ae_tdi(N, DT, ref_rng, channel=c, mod_config=mod[c])[0]
                 for c in _CHANNELS}
    ref = float(np.std(np.concatenate([ref_noise[c] + sig[c] for c in _CHANNELS])))

    def _w(series):
        return wdm_analysis_coefficients(series / ref, DT, NT, pcfg)

    _, tg, fg = _w(sig["A"])
    templates, true_surf = {}, {}
    cal_rng = np.random.default_rng(99)
    for c in _CHANNELS:
        g1, _, _ = _w(gb_ae_signal(N, DT, {**gbp, "phi0": 0.0}, channel=c))
        g2, _, _ = _w(gb_ae_signal(N, DT, {**gbp, "phi0": np.pi / 2}, channel=c))
        templates[c] = np.stack([g1, g2], axis=0)
        # Smooth coverage target: the analytic E[w^2] = C_m * S(u,f). The per-channel
        # WDM normalisation C_m is calibrated from the (noise-only) Monte Carlo power
        # averaged over time -- using the noisy per-cell MC power directly as the
        # target would itself scatter ~18% (w^2 ~ S chi^2_1 over 60 draws) and
        # spuriously depress coverage.
        ref_w2 = np.mean(
            [_w(simulate_tv_ae_tdi(N, DT, cal_rng, channel=c, mod_config=mod[c])[0])[0] ** 2
             for _ in range(cfg["cal_draws"])], axis=0)
        s_analytic = true_tv_ae_tdi_psd(tg, fg, channel=c, mod_config=mod[c])
        cal_m = ref_w2.mean(axis=0) / s_analytic.mean(axis=0)  # per-frequency C_m
        true_surf[c] = cal_m[None, :] * s_analytic

    # True shared amplitude (projection of the signal onto the templates).
    sig_w = {c: _w(sig[c])[0] for c in _CHANNELS}
    M = np.stack([np.concatenate([templates[c][k].ravel() for c in _CHANNELS]) for k in (0, 1)], axis=1)
    beta_true, *_ = np.linalg.lstsq(
        M, np.concatenate([sig_w[c].ravel() for c in _CHANNELS]), rcond=None)
    amp_true = float(np.hypot(*beta_true))

    templ_stack = np.stack([templates[c] for c in _CHANNELS], axis=0)  # (C, K, nt, nf)

    ratios, cov_A, cov_E, divs = [], [], [], []
    for r in range(n_real):
        rng = np.random.default_rng(1000 + r)
        coeffs = np.stack([_w(simulate_tv_ae_tdi(N, DT, rng, channel=c, mod_config=mod[c])[0]
                              + sig[c])[0] for c in _CHANNELS], axis=0)
        res = run_multichannel_joint_mcmc(coeffs, templ_stack, tg, fg, config=pcfg,
                                          random_seed=1, **cfg["fit"])
        ratios.append(np.hypot(*res["beta_mean"]) / amp_true)
        cov = []
        for ci, c in enumerate(_CHANNELS):
            inside = (res["psd_lower"][ci] <= true_surf[c]) & (true_surf[c] <= res["psd_upper"][ci])
            cov.append(float(inside.mean()))
        cov_A.append(cov[0]); cov_E.append(cov[1]); divs.append(res["divergences"])
        print(f"[real {r:2d}/{n_real}] |A|/|A|_true={ratios[-1]:.3f}  "
              f"cov(A)={cov[0]:.2f} cov(E)={cov[1]:.2f}  div={res['divergences']}")
        np.savez_compressed(CACHE, ratios=np.array(ratios), cov_A=np.array(cov_A),
                            cov_E=np.array(cov_E), divergences=np.array(divs),
                            amp_true=amp_true, n_done=len(ratios), n_real=n_real,
                            snr=cfg["target_snr"])

    rr = np.array(ratios)
    print(f"[summary] |A|/|A|_true = {rr.mean():.3f} +/- {rr.std():.3f} "
          f"(median {np.median(rr):.3f}, n={len(rr)})")
    print(f"[summary] coverage A={np.mean(cov_A):.2f} E={np.mean(cov_E):.2f} "
          f"(nominal 0.90)")
    render(FIG_DIR, np.load(CACHE))


def render(fig_dir: Path, d) -> None:
    set_paper_style()
    ratios = np.asarray(d["ratios"]); cov_A = np.asarray(d["cov_A"]); cov_E = np.asarray(d["cov_E"])
    n = len(ratios)
    fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)

    # Amplitude recovery across realizations.
    parts = ax_a.violinplot([ratios], positions=[0], widths=0.6, showmeans=False,
                            showextrema=False, showmedians=True)
    parts["bodies"][0].set_facecolor("tab:blue"); parts["bodies"][0].set_alpha(0.45)
    parts["cmedians"].set_color("black")
    ax_a.scatter(np.random.default_rng(0).normal(0, 0.03, n), ratios, s=14,
                 color="tab:blue", alpha=0.7, zorder=3)
    ax_a.axhline(1.0, color="black", ls="--", lw=1.5, label="injected")
    ax_a.set_xticks([0]); ax_a.set_xticklabels([f"A+E joint\n($n={n}$)"])
    ax_a.set_ylabel(r"recovered $|A|\,/\,|A|_{\mathrm{true}}$")
    ax_a.set_xlim(-0.6, 0.6); ax_a.set_ylim(0.0, 1.4)
    ax_a.set_title(rf"Amplitude recovery ($\mu={ratios.mean():.2f}\pm{ratios.std():.2f}$)")
    ax_a.legend(loc="lower right")

    # PSD-band coverage across realizations.
    parts = ax_c.violinplot([cov_A, cov_E], positions=[0, 1], widths=0.6,
                            showmeans=False, showextrema=False, showmedians=True)
    for body, col in zip(parts["bodies"], ("tab:blue", "tab:green")):
        body.set_facecolor(col); body.set_alpha(0.45)
    parts["cmedians"].set_color("black")
    ax_c.axhline(0.9, color="black", ls=":", lw=1.5, label="nominal 90%")
    ax_c.set_xticks([0, 1]); ax_c.set_xticklabels(["A", "E"])
    ax_c.set_ylabel("90% PSD-band coverage")
    ax_c.set_xlim(-0.6, 1.6); ax_c.set_ylim(0.0, 1.05)
    ax_c.set_title("Non-stationary PSD coverage"); ax_c.legend(loc="lower right")
    save_figure(fig, fig_dir / "lisa_ensemble.png")
    print(f"[figure] wrote {fig_dir / 'lisa_ensemble.png'}")


if __name__ == "__main__":
    main()
