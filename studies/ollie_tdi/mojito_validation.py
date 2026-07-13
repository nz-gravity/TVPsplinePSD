"""Extra noise-consistency tests for the Mojito TV-PSD fits.

Implements two checks beyond the in-sample whitening in ``mojito_experiments.py``:

  #8 Cross-validated whitening (out-of-sample). Fit S on window A, then whiten a
     DISJOINT adjacent window B of the same length with the time-averaged S(f).
     Because B is a fresh noise realisation at a later epoch, this tests temporal
     generalisation / stationarity -- exactly the "estimate noise off-source,
     apply on-source" situation in parameter estimation -- rather than merely
     re-explaining the training realisation.

  #5 AET cross-channel correlation. Whiten A, E, T separately and measure the
     residual correlations corr(z_i, z_j) overall and as a function of frequency.
     The orthogonal AET combinations are only exactly uncorrelated for equal arms;
     with the flexing Mojito constellation any leftover correlation means the
     joint (multi-channel) noise covariance used in PE needs off-diagonal terms.

Uses the same windows/config as the ``1_week`` and ``1_month`` experiments, with a
shorter MCMC (the whitening only needs the posterior-mean surface). Writes into
``experiments/<name>/validation/``.

Run:
    python studies/ollie_tdi/mojito_validation.py --only 1_week 1_month
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from check_XYZ import DT
from fit_mojito_segment import band_trims, load_segment
from mojito_experiments import EXP_DIR, EXPERIMENTS, START_DAY
from scipy.stats import norm, pearsonr

from tv_pspline_psd import (
    PSplineConfig,
    run_wdm_psd_mcmc,
    set_paper_style,
    wdm_analysis_coefficients,
)

set_paper_style()

FMIN, FMAX = 1e-4, 1e-1
# Whitening needs only a converged posterior mean, so a short chain suffices.
VAL_MCMC = dict(n_warmup=150, n_samples=150, num_chains=1, random_seed=0)


def make_config(n_total: int, nt: int, time_knots: int) -> PSplineConfig:
    trim_low, trim_high = band_trims(n_total, nt, FMIN, FMAX)
    return PSplineConfig(
        n_interior_knots_time=time_knots, n_interior_knots_freq=30,
        trim_time_bins=2, trim_low_freq_channels=trim_low,
        trim_high_freq_channels=trim_high, centered=True,
    )


def coeffs2d(res_coeffs: np.ndarray) -> np.ndarray:
    return res_coeffs[0] if res_coeffs.ndim == 3 else res_coeffs


def fit_channel(channel: str, nt: int, time_knots: int, start_day: float, days):
    """Fit one channel on a window; return (res, series, start_used)."""
    series, start_used = load_segment(channel, nt, start_day, days)
    config = make_config(series.size, nt, time_knots)
    res = run_wdm_psd_mcmc(series, dt=DT, nt=nt, config=config, **VAL_MCMC)
    return res, config, start_used


# ---------------------------------------------------------------- #8 cross-val
def cross_val_whitening(exp: str, channel: str, outdir: Path) -> dict:
    cfg = EXPERIMENTS[exp]
    nt, tk = cfg["nt"], cfg["time_knots"]
    (train_start, train_days), (test_start, test_days) = cfg["cv"]
    res_a, config, s0a = fit_channel(channel, nt, tk, train_start, train_days)
    fg = res_a["freq_grid"]
    S_full = res_a["psd_mean"]                 # (n_time, n_freq), coeff units
    z_a = coeffs2d(res_a["coeffs"]) / np.sqrt(S_full)
    S_bar = S_full.mean(0)                      # time-averaged predictor S(f)

    # Disjoint next window of equal length, whitened by the window-A PSD.
    series_b, s0b = load_segment(channel, nt, test_start, test_days)
    cb, _, fgb = wdm_analysis_coefficients(series_b, DT, nt, config)
    if not np.array_equal(fg, fgb):
        raise ValueError(
            f"cross-validation frequency grids differ for {exp}: "
            f"train shape {fg.shape}, test shape {fgb.shape}"
        )
    z_b = coeffs2d(cb) / np.sqrt(S_bar[None, :])

    za, zb = z_a.ravel(), z_b.ravel()
    n_time = z_a.shape[0]
    sig_f = np.sqrt(2.0 / n_time)
    stats = dict(in_mean_z2=float((za**2).mean()), out_mean_z2=float((zb**2).mean()),
                 in_kurt=float(((za**4).mean()) - 3.0), out_kurt=float((zb**4).mean() - 3.0))

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.8), constrained_layout=True)
    # (a) in vs out histograms over N(0,1)
    for z, lab, c in ((za, "in-sample (A)", "tab:blue"), (zb, "out-of-sample (B)", "tab:orange")):
        ax[0].hist(z, bins=80, range=(-4.5, 4.5), density=True, histtype="step", lw=1.3,
                   color=c, label=f"{lab}: $\\overline{{z^2}}$={ (z**2).mean():.3f}")
    xx = np.linspace(-4.5, 4.5, 200)
    ax[0].plot(xx, norm.pdf(xx), "k--", lw=1.0)
    ax[0].set_xlabel(r"$w/\sqrt{\hat S}$"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=7)
    # (b) QQ-plot of the out-of-sample residuals (tail-sensitive)
    probs = np.linspace(0.0005, 0.9995, 400)
    ax[1].plot(norm.ppf(probs), np.quantile(zb, probs), color="tab:orange", lw=1.2)
    lim = [-4.5, 4.5]
    ax[1].plot(lim, lim, "k--", lw=1.0); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("normal quantile"); ax[1].set_ylabel("out-of-sample $z$ quantile")
    ax[1].set_title("QQ (out-of-sample)", fontsize=8)
    # (c) per-frequency mean z^2, in vs out
    ax[2].plot(fg, (z_a**2).mean(0), color="tab:blue", lw=0.5, label="in (A)")
    ax[2].plot(fg, (z_b**2).mean(0), color="tab:orange", lw=0.5, label="out (B)")
    ax[2].axhline(1.0, color="k", ls="--", lw=1.0)
    ax[2].fill_between(fg, 1 - 3 * sig_f, 1 + 3 * sig_f, color="0.7", alpha=0.4)
    ax[2].set_xscale("log"); ax[2].set_xlabel("f [Hz]")
    ax[2].set_ylabel(r"$\overline{z^2}$ per channel"); ax[2].legend(fontsize=7)
    fig.suptitle(f"{exp}: cross-validated whitening ({channel}) -- "
                 f"fit d{s0a:.0f}-{s0a + train_days:.0f}, "
                 f"applied to d{s0b:.0f}-{s0b + test_days:.0f}", fontsize=9)
    fig.savefig(outdir / "cross_val_whitening.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{exp}] cross-val: in z2={stats['in_mean_z2']:.3f} out z2={stats['out_mean_z2']:.3f}")
    return stats


# ------------------------------------------------------------- #5 AET cross-corr
def aet_cross_channel(exp: str, outdir: Path, n_fbins: int = 25) -> dict:
    cfg = EXPERIMENTS[exp]
    nt, days, tk = cfg["nt"], cfg["days"], cfg["time_knots"]
    z, fg = {}, None
    for ch in "AET":
        res, _, _ = fit_channel(ch, nt, tk, START_DAY, days)
        fg = res["freq_grid"]
        z[ch] = coeffs2d(res["coeffs"]) / np.sqrt(res["psd_mean"])
    pairs = [("A", "E"), ("A", "T"), ("E", "T")]
    overall = {f"{i}{j}": float(pearsonr(z[i].ravel(), z[j].ravel())[0]) for i, j in pairs}

    # correlation as a function of frequency (log-binned over cells in each bin)
    edges = np.logspace(np.log10(fg[0]), np.log10(fg[-1]), n_fbins + 1)
    idx = np.clip(np.digitize(fg, edges) - 1, 0, n_fbins - 1)
    fc = np.sqrt(edges[:-1] * edges[1:])
    r_of_f = {p: np.full(n_fbins, np.nan) for p in overall}
    nbin = np.zeros(n_fbins)
    for b in range(n_fbins):
        m = idx == b
        nbin[b] = m.sum() * z["A"].shape[0]
        if m.sum() < 2:
            continue
        for i, j in pairs:
            r_of_f[f"{i}{j}"][b] = pearsonr(z[i][:, m].ravel(), z[j][:, m].ravel())[0]

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0), constrained_layout=True)
    colors = {"AE": "tab:blue", "AT": "tab:green", "ET": "tab:red"}
    for p in overall:
        ax[0].plot(fc, r_of_f[p], color=colors[p], lw=1.1, label=f"{p[0]},{p[1]} ({overall[p]:+.3f})")
    band = 3.0 / np.sqrt(np.maximum(nbin, 1))
    ax[0].fill_between(fc, -band, band, color="0.7", alpha=0.4, label=r"$\pm3\sigma$")
    ax[0].axhline(0.0, color="k", ls="--", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("f [Hz]")
    ax[0].set_ylabel(r"corr$(z_i, z_j)$"); ax[0].legend(fontsize=7, title="pair (overall r)")
    ax[0].set_title("AET residual cross-correlation vs f", fontsize=9)
    # overall 3x3 correlation matrix
    M = np.eye(3)
    lab = ["A", "E", "T"]
    for i, j in pairs:
        a, b = lab.index(i), lab.index(j)
        M[a, b] = M[b, a] = overall[f"{i}{j}"]
    im = ax[1].imshow(M, vmin=-0.15, vmax=0.15, cmap="coolwarm")
    ax[1].set_xticks(range(3), lab); ax[1].set_yticks(range(3), lab)
    for a in range(3):
        for b in range(3):
            ax[1].text(b, a, f"{M[a,b]:+.3f}", ha="center", va="center", fontsize=9)
    ax[1].set_title("overall corr matrix", fontsize=9)
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    fig.suptitle(f"{exp}: AET cross-channel covariance check", fontsize=10)
    fig.savefig(outdir / "aet_cross_channel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{exp}] AET overall corr: " + " ".join(f"{k}={v:+.3f}" for k, v in overall.items()))
    return overall


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", default=["1_week", "1_month"],
                   choices=[e for e in EXPERIMENTS])
    p.add_argument("--channel", default="X", choices=[*"XYZ", *"AET"])
    args = p.parse_args()
    out = {}
    for exp in args.only:
        outdir = EXP_DIR / exp / "validation"
        outdir.mkdir(parents=True, exist_ok=True)
        cv = cross_val_whitening(exp, args.channel, outdir)
        aet = aet_cross_channel(exp, outdir)
        out[exp] = dict(cross_val=cv, aet_corr=aet)
        with open(outdir / "validation.json", "w") as fp:
            json.dump(out[exp], fp, indent=2, default=float)
        print(f"[{exp}] validation -> {outdir}")
    print("[done] validation complete")


if __name__ == "__main__":
    main()
