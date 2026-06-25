"""The demonstration: stationary-PSD vs time-varying-PSD on the same source.

Two datasets, each analysed two ways (a 2x2):

  Case A data: stationary LISA noise (no Galactic modulation) + the source.
  Case B data: non-stationary noise (cyclostationary modulation)   + the source.

  Stationary analysis: frequency-domain Whittle with a log_psplines PSD.
  TV-PSD analysis:      WDM tensor-product log-P-spline joint fit.

Expectation:
  * On Case A the two analyses should agree (a stationary truth -- the flexible
    TV-PSD does not hurt).
  * On Case B the TV-PSD should be better calibrated than the stationary PSD,
    which sees only a year-averaged spectrum and so mis-measures the source whose
    SNR accumulates where the confusion is at its annual maximum.

The compared parameter is the luminosity distance ``dL`` (amplitude/SNR) -- the
noise-model-sensitive parameter (Mc/tc are phase-pinned). The merger is placed at
the confusion-power maximum so the time-localization bites.

    uv run python -m studies.sobh_tvpsd.comparison            # smoke scale
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from datasets.lisa import (  # noqa: E402
    LISANoiseConfig,
    digman_cornish_power_modulation,
    lisa_instrument_psd,
    normalization_constant,
    simulate_tv_lisa_noise,
)
from datasets.sobh import SOBHParams, sobh_strain_td  # noqa: E402
from tv_pspline_psd import (  # noqa: E402
    PSplineConfig,
    build_sobh_wdm_grid,
    make_sobh_wdm_signal_fn,
    run_joint_dL_wdm_mcmc,
    wdm_analysis_coefficients,
)
from studies.sobh_tvpsd.fd_baseline import estimate_stationary_psd, fd_dL_posterior

FIG_DIR = Path(__file__).resolve().parents[2] / "notes" / "figures"
RES_DIR = Path(__file__).resolve().parent / "results"


def _optimal_snr(h_td, dt, S):
    n = len(h_td)
    hf = dt * np.fft.rfft(h_td)
    df = 1.0 / (n * dt)
    return float(np.sqrt(np.sum(4.0 * np.abs(hf[1:]) ** 2 / S[1:] * df)))


def run(quick: bool = True) -> dict:
    dt, nt = 50.0, 32
    n = 2**13 if quick else 2**15
    # Band-limit the WDM analysis to the source + confusion band (~0.5-7 mHz),
    # excluding the instrument f^-4 low-frequency divergence.
    cfg = PSplineConfig(trim_low_freq_channels=13, trim_high_freq_channels=77)
    T = n * dt

    # Heavy source: merger ~4 mHz, in the Galactic-confusion band.
    params = SOBHParams(m1=5.0e5, m2=5.0e5)

    # Place the merger at the confusion-power maximum over the observation.
    u = (np.arange(n) + 0.5) / n
    r = digman_cornish_power_modulation(u, channel="A", n_year_cycles=1.0)
    params.tc = float(u[np.argmax(r)] * T)

    # Distance for a target SNR against instrument noise.
    freq = np.fft.rfftfreq(n, d=dt)
    fe = freq.copy(); fe[0] = fe[1]
    S_inst = lisa_instrument_psd(fe)
    target_snr = 40.0
    params.distance *= _optimal_snr(sobh_strain_td(n, dt, params), dt, S_inst) / target_snr
    dl_true = d_ref = params.distance

    # Numerical normalisation (PSD -> O(1) for the WDM estimator); the signal is
    # scaled by the same constant so dL is unchanged.
    norm_cfg = LISANoiseConfig(normalize=True)
    nref = normalization_constant(n, dt, norm_cfg)
    h_ref = sobh_strain_td(n, dt, params) / np.sqrt(nref)  # responded, normalised
    print(f"[setup] M={params.m1+params.m2:.0e} Msun  tc={params.tc/T:.2f}T (conf max)  "
          f"dL_true={dl_true:.4g} Mpc  SNR~{target_snr}  n={n} dt={dt}")

    # Precompute the WDM signal template at d_ref (normalised units) once.
    grid = build_sobh_wdm_grid(n, dt, nt, params, cfg)
    signal_fn = make_sobh_wdm_signal_fn(grid)
    template = np.asarray(signal_fn(jnp.asarray(
        [params.chirp_mass, params.tc, np.log(dl_true)]))) / np.sqrt(nref)

    configs = {
        "A_stationary": LISANoiseConfig(normalize=True, modulation_model="raised_cosine",
                                        modulation_depth=0.0),
        "B_nonstationary": LISANoiseConfig(normalize=True, modulation_model="digman_cornish",
                                           dc_channel="A", n_year_cycles=1.0),
    }
    nw_fd, ns_fd = (400, 800) if quick else (800, 1500)
    nw_wd, ns_wd = (400, 600) if quick else (700, 1000)

    out = {"dl_true": dl_true, "cases": {}, "time_grid": grid.time_grid,
           "freq_grid": grid.freq_grid}
    for case, noise_cfg in configs.items():
        rng = np.random.default_rng(0)
        noise, _ = simulate_tv_lisa_noise(n, dt=dt, rng=rng, config=noise_cfg)
        data = h_ref + noise

        # --- Stationary analysis (frequency domain + log_psplines PSD) ---
        est = estimate_stationary_psd(data, dt, n_knots=24,
                                      n_warmup=nw_fd // 2, n_samples=ns_fd // 2)
        dL_stat = fd_dL_posterior(data, h_ref, dt, S_stat=est["S_stat"], d_ref=d_ref,
                                  dl_ref=dl_true, dl_scale=0.6,
                                  n_warmup=nw_fd, n_samples=ns_fd)["dL"]

        # --- TV-PSD analysis (WDM joint fit, distance only) ---
        coeffs, tg, fg = wdm_analysis_coefficients(data, dt, nt, cfg)
        res_tv = run_joint_dL_wdm_mcmc(
            coeffs, tg, fg, template, d_ref, config=cfg, dl_ref=dl_true,
            dl_scale=0.6, n_warmup=nw_wd, n_samples=ns_wd, random_seed=2)
        dL_tv = res_tv["dL"]

        out["cases"][case] = {
            "dL_stat": dL_stat, "dL_tv": dL_tv,
            "psd_tv_mean": res_tv["psd_mean"], "S_stat": est["S_stat"],
            "divergences": res_tv["divergences"],
        }
        for nm, s in (("stationary", dL_stat), ("TV-PSD", dL_tv)):
            z = (s.mean() - dl_true) / s.std()
            print(f"[{case}] {nm:10s} dL={s.mean():.5g} +/- {s.std():.4g}  "
                  f"z={z:+.2f}  width/true={s.std()/dl_true*100:.2f}%")
    return out


def make_figure(out: dict) -> None:
    import matplotlib.pyplot as plt
    from tv_pspline_psd import save_figure, set_paper_style
    set_paper_style()
    dl = out["dl_true"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    titles = {"A_stationary": "Case A: stationary noise",
              "B_nonstationary": "Case B: non-stationary noise"}
    for ax, case in zip(axes, ("A_stationary", "B_nonstationary")):
        c = out["cases"][case]
        for s, color, label in ((c["dL_stat"], "tab:red", "Stationary PSD (freq-domain)"),
                                (c["dL_tv"], "tab:green", "WDM time-varying PSD")):
            ax.hist(s / dl, bins=40, density=True, histtype="step", color=color,
                    lw=2, label=label)
        ax.axvline(1.0, color="0.3", ls="--", lw=1)
        ax.set_title(titles[case]); ax.set_xlabel(r"$d_L / d_L^{\rm true}$")
    axes[0].set_ylabel("posterior density"); axes[0].legend(frameon=False, fontsize=9)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, FIG_DIR / "sobh_tvpsd_dL_comparison.png")
    print(f"[figure] wrote {FIG_DIR / 'sobh_tvpsd_dL_comparison.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="production scale")
    args = ap.parse_args()
    out = run(quick=not args.full)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RES_DIR / "comparison.npz",
             dl_true=out["dl_true"],
             **{f"{c}_{k}": v for c, d in out["cases"].items()
                for k, v in d.items() if isinstance(v, np.ndarray) or np.isscalar(v)})
    make_figure(out)


if __name__ == "__main__":
    main()
