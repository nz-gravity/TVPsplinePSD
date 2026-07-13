"""Drift experiments: TV-PSD noise estimation on Mojito segments of growing length.

Runs the windowed WDM log-P-spline fit at a ladder of segment lengths -- 1 week,
1 month, 6 months, 1 year, and the full (taper-included) series -- and, for each,
saves into its own subdirectory:

  * ``spectrum.png``      time-averaged S(f) vs Welch,
  * ``null_zoom.png``     linear zoom on the first Michelson null with the
                          1/(2<L(t)>) prediction overlaid (the "wandering"),
  * ``whitening.png``     goodness-of-fit / covariance-consistency test on the
                          standardised coefficients z = w / sqrt(S_hat):
                          hist vs N(0,1), mean z^2 per time bin, per channel,
  * ``surface.png``       raw WDM log-power vs recovered log S(t,f),
  * ``fit.npz`` / ``diag.json``.

The whitening panels are the core check: if the fitted WDM covariance matches the
noise, z ~ N(0,1) and mean(z^2) == 1 both per time bin and per channel, within
the +/-3 sigma bands (sigma_t = sqrt(2/n_freq), sigma_f = sqrt(2/n_time)).

Run:
    python studies/ollie_tdi/mojito_experiments.py --only 1_week 1_month
    python studies/ollie_tdi/mojito_experiments.py --only 6_month --background-safe
    python studies/ollie_tdi/mojito_experiments.py            # all five (slow!)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from check_XYZ import DATA, DT
from fit_mojito_segment import band_trims, load_segment
from null_zoom import predicted_null
from scipy.signal import welch
from scipy.stats import kstest

from tv_pspline_psd import (
    PSplineConfig,
    run_wdm_psd_mcmc,
    set_paper_style,
    summarize_mcmc_diagnostics,
)
from tv_pspline_psd.datasets import wdm_white_noise_calibration

set_paper_style()

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = REPO / "studies" / "results" / "ollie_tdi" / "experiments"

# Finite experiment windows are nested just past the leading taper.  The full
# cross-validation fold starts at the estimated 18-day taper boundary so two
# equal, disjoint 348-day windows fit inside the 716-day series.
START_DAY = 20.0
FULL_CV_START_DAY = 18.0
FULL_CV_DAYS = 348.0

# Each ``cv`` value is ``((train_start_day, train_days),
# (test_start_day, test_days))``.
EXPERIMENTS: dict[str, dict] = {
    "1_week": dict(days=7.0, nt=32, time_knots=8, warmup=300, samples=300,
                   chains=2, cv=((START_DAY, 7.0), (START_DAY + 7.0, 7.0))),
    "1_month": dict(days=30.0, nt=30, time_knots=10, warmup=300, samples=300,
                    chains=2, cv=((START_DAY, 30.0), (START_DAY + 30.0, 30.0))),
    "6_month": dict(days=182.0, nt=180, time_knots=16, warmup=150, samples=150,
                    chains=1, cv=((START_DAY, 182.0), (START_DAY + 182.0, 182.0))),
    "1_year": dict(days=365.0, nt=360, time_knots=24, warmup=120, samples=120,
                   chains=1, cv=((START_DAY, 365.0), (START_DAY + 365.0, 365.0))),
    "1.5_year": dict(days=548.0, nt=548, time_knots=30, warmup=120, samples=120,
                     chains=1, cv=((START_DAY, 548.0), (START_DAY + 548.0, 548.0))),
    "full": dict(days=None, nt=700, time_knots=36, warmup=120, samples=120,
                 chains=1,
                 cv=((FULL_CV_START_DAY, FULL_CV_DAYS),
                     (FULL_CV_START_DAY + FULL_CV_DAYS, FULL_CV_DAYS))),
}


def plot_spectrum(fg, S_est, S_lo, S_hi, wf, wp, band, out, title):
    fig, ax = plt.subplots(figsize=(3.9, 2.9), constrained_layout=True)
    m = (wf >= band[0]) & (wf <= band[1])
    ax.loglog(wf[m], wp[m], color="0.55", lw=0.7, label="Welch")
    ax.loglog(fg, S_est.mean(0), color="tab:blue", lw=1.1, label="TV fit (time avg)")
    ax.fill_between(fg, S_lo.mean(0), S_hi.mean(0), color="tab:blue", alpha=0.3)
    ax.set_xlabel("f [Hz]"); ax.set_ylabel(r"$S(f)$ [1/Hz]")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_null_zoom(tg_days, fg, power, channel, out, title, fband=(0.045, 0.075)):
    from scipy.ndimage import uniform_filter
    band = (fg >= fband[0]) & (fg <= fband[1])
    if band.sum() < 4:
        return  # band not covered at this resolution
    logp = np.log(power[:, band] + 1e-300)
    smooth_t = max(3, round(len(tg_days) / 120))
    logp = uniform_filter(logp, size=(smooth_t, 3))
    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    vmin, vmax = np.percentile(logp, [2, 99.5])
    mesh = ax.pcolormesh(tg_days, fg[band], logp.T, shading="auto", cmap="viridis",
                         vmin=vmin, vmax=vmax)
    fnull = predicted_null(channel, tg_days)
    if fnull is not None:
        ax.plot(tg_days, fnull, "r-", lw=1.2, label=r"$1/2\langle L\rangle$")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(tg_days.min(), tg_days.max()); ax.set_ylim(*fband)
    ax.set_xlabel("time [days]"); ax.set_ylabel("f [Hz]")
    ax.set_title(title, fontsize=9)
    fig.colorbar(mesh, ax=ax, shrink=0.9, label=r"$\log|w|^2$")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_whitening(z, tg_days, fg, out, title):
    """3-panel covariance-consistency test on z = w / sqrt(S_hat)."""
    z2 = z**2
    z2_time = z2.mean(1)               # per time bin (avg over freq)
    z2_freq = z2.mean(0)               # per channel (avg over time)
    n_time, n_freq = z.shape
    sig_t = np.sqrt(2.0 / n_freq)      # std of mean-z^2 per time bin
    sig_f = np.sqrt(2.0 / n_time)      # std of mean-z^2 per channel
    zf = z.ravel()
    ks = kstest(zf, "norm")
    mean_z2 = float(z2.mean())

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.7), constrained_layout=True)
    # (a) histogram vs standard normal
    ax[0].hist(zf, bins=80, range=(-4.5, 4.5), density=True, color="tab:blue", alpha=0.7)
    xx = np.linspace(-4.5, 4.5, 200)
    ax[0].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.2)
    ax[0].set_xlabel(r"$w/\sqrt{\hat S}$"); ax[0].set_ylabel("density")
    ax[0].set_title(rf"$\overline{{z^2}}$={mean_z2:.3f}, KS={ks.statistic:.3f}", fontsize=8)
    # (b) mean z^2 per time bin
    ax[1].plot(tg_days, z2_time, color="tab:blue", lw=0.8)
    ax[1].axhline(1.0, color="k", ls="--", lw=1.0)
    ax[1].fill_between(tg_days, 1 - 3 * sig_t, 1 + 3 * sig_t, color="0.7", alpha=0.4)
    ax[1].set_xlabel("time [days]"); ax[1].set_ylabel(r"$\overline{z^2}$ per time bin")
    # (c) mean z^2 per channel
    ax[2].plot(fg, z2_freq, color="tab:blue", lw=0.5)
    ax[2].axhline(1.0, color="k", ls="--", lw=1.0)
    ax[2].fill_between(fg, 1 - 3 * sig_f, 1 + 3 * sig_f, color="0.7", alpha=0.4)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("f [Hz]"); ax[2].set_ylabel(r"$\overline{z^2}$ per channel")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return dict(mean_z2=mean_z2, ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue),
                z2_time_std=float(z2_time.std()), z2_freq_std=float(z2_freq.std()))


def plot_surface(tg_days, fg, raw_pow, logS, band, out, title):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True, sharey=True)
    m0 = axes[0].pcolormesh(tg_days, fg, raw_pow.T, shading="auto", cmap="viridis")
    axes[0].set_title("raw WDM log power", fontsize=9)
    axes[1].pcolormesh(tg_days, fg, logS.T, shading="auto", cmap="viridis",
                       vmin=m0.get_clim()[0], vmax=m0.get_clim()[1])
    axes[1].set_title(r"posterior mean $\log \hat S$", fontsize=9)
    for ax in axes:
        ax.set_yscale("log"); ax.set_ylim(*band); ax.set_xlabel("time [days]")
    axes[0].set_ylabel("f [Hz]")
    fig.colorbar(axes[1].collections[0], ax=axes, shrink=0.85, label=r"$\log S$ [1/Hz]")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def run_experiment(name: str, channel: str, fmin: float, fmax: float) -> dict:
    cfg = EXPERIMENTS[name]
    outdir = EXP_DIR / name
    outdir.mkdir(parents=True, exist_ok=True)
    nt = cfg["nt"]

    series, start_used = load_segment(channel, nt, START_DAY, cfg["days"])
    n_total = series.size
    end_used = start_used + n_total * DT / 86400.0
    trim_low, trim_high = band_trims(n_total, nt, fmin, fmax)
    nf = n_total // nt
    df = 1.0 / (2.0 * nf * DT)
    f_lo, f_hi = trim_low * df, (nf - trim_high) * df
    print(f"[{name}] {channel} days {start_used:.1f}-{end_used:.1f} ({n_total*DT/86400:.1f} d), "
          f"N={n_total}, nt={nt}, nf={nf}, band [{f_lo:.2e},{f_hi:.3f}] Hz, {nf-trim_high-trim_low+1} chans")

    config = PSplineConfig(
        n_interior_knots_time=cfg["time_knots"], n_interior_knots_freq=30,
        trim_time_bins=2, trim_low_freq_channels=trim_low, trim_high_freq_channels=trim_high,
        centered=True,
    )
    t0 = time.perf_counter()
    res = run_wdm_psd_mcmc(series, dt=DT, nt=nt, config=config,
                           n_warmup=cfg["warmup"], n_samples=cfg["samples"],
                           num_chains=cfg["chains"], random_seed=0)
    wall = time.perf_counter() - t0
    diag = summarize_mcmc_diagnostics(res)
    print(f"[{name}] fit wall={wall:.0f}s div={diag['divergences']} "
          f"rhat<= {max(diag['phi_time']['r_hat'], diag['phi_freq']['r_hat']):.3f}")

    # Whitened coefficients (coefficient-variance units; no calibration needed).
    coeffs2d = res["coeffs"][0] if res["coeffs"].ndim == 3 else res["coeffs"]
    S_coeff = res["psd_mean"]
    z = coeffs2d / np.sqrt(S_coeff)
    tg_days = start_used + res["time_grid"] * n_total * DT / 86400.0
    fg = res["freq_grid"]

    # Calibrate to 1/Hz for the spectrum (few draws suffice on big grids).
    n_draws = int(np.clip(6_000_000 // max(1, n_total // 100), 24, 200))
    cal = wdm_white_noise_calibration(n_total, DT, nt, config, n_draws=n_draws)
    res["provenance"].update({
        "calibration": {"n_draws": n_draws, "seed": 0},
        "source_data": {
            "path": str(DATA),
            "shape": [int(n_total)],
            "channel": channel,
        },
    })
    to_psd = 2.0 * DT / cal[None, :]
    S_est, S_lo, S_hi = (res[k] * to_psd for k in ("psd_mean", "psd_lower", "psd_upper"))
    wf, wp = welch(series, fs=1.0 / DT, nperseg=min(n_total, 2**16))

    stats = plot_whitening(z, tg_days, fg, outdir / "whitening.png",
                           f"{name}: WDM covariance check ({channel}, days {start_used:.0f}-{end_used:.0f})")
    plot_spectrum(fg, S_est, S_lo, S_hi, wf, wp, (f_lo, f_hi), outdir / "spectrum.png",
                  f"{name}: spectrum ({channel})")
    plot_null_zoom(tg_days, fg, res["power"], channel, outdir / "null_zoom.png",
                   f"{name}: null wander ({channel})")
    plot_surface(tg_days, fg, np.log(res["power"] + 1e-300) + np.log(to_psd),
                 np.log(S_est), (f_lo, f_hi), outdir / "surface.png", f"{name}: {channel}")

    np.savez(outdir / "fit.npz", time_grid_days=tg_days, freq_grid=fg,
             psd_mean=S_est, psd_lower=S_lo, psd_upper=S_hi, z2_time=(z**2).mean(1),
             z2_freq=(z**2).mean(0), welch_f=wf, welch_psd=wp,
             start_day=start_used, end_day=end_used, band=(f_lo, f_hi))
    summary = dict(name=name, channel=channel, start_day=start_used, end_day=end_used,
                   n_total=int(n_total), nt=nt, wall_s=wall, divergences=diag["divergences"],
                   rhat_phi_time=diag["phi_time"]["r_hat"], rhat_phi_freq=diag["phi_freq"]["r_hat"],
                   **stats)
    with open(outdir / "diag.json", "w") as fp:
        json.dump(
            {**summary, "provenance": res["provenance"], "full_diag": diag},
            fp,
            indent=2,
            default=float,
        )
    print(f"[{name}] mean(z^2)={stats['mean_z2']:.3f} KS={stats['ks_stat']:.3f} -> {outdir}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    p.add_argument("--channel", default="X", choices=[*"XYZ", *"AET"])
    p.add_argument("--fmin", type=float, default=1e-4)
    p.add_argument("--fmax", type=float, default=1e-1)
    args = p.parse_args()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for n in args.only:
        try:
            summaries.append(run_experiment(n, args.channel, args.fmin, args.fmax))
        except Exception as exc:  # keep going so one slow/failed fit doesn't sink the rest
            print(f"[{n}] FAILED: {type(exc).__name__}: {exc}")
    with open(EXP_DIR / "summary.json", "w") as fp:
        json.dump(summaries, fp, indent=2, default=float)
    print(f"[done] {len(summaries)} experiment(s); summary -> {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
