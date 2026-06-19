"""Statistical robustness of the WDM log-P-spline estimator.

Three referee-grade checks:

1. Coverage calibration -- across many LS2 realizations, the empirical frequency
   with which the (calibrated) true PSD falls inside a central credible interval
   is compared to the nominal level. A well-calibrated posterior sits on the
   diagonal.
2. Diagonal-Whittle validity -- the empirical correlation of the trimmed WDM
   coefficients quantifies how well the independent-coefficient likelihood holds.
3. Whitened residuals -- ``w / sqrt(S_hat)`` should be standard normal if the
   fitted surface is an adequate noise model.

Saves ``notes/figures/calibration.png`` and ``notes/figures/whittle_diagnostics.png``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from datasets import simulate_ls2, true_psd_ls2, wdm_white_noise_calibration
from wdm_psd import PSplineConfig, run_wdm_psd_mcmc, wdm_analysis_coefficients

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
DT, NT, N_TOTAL = 0.1, 24, 576
CONFIG = PSplineConfig()
LEVELS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


def coverage_calibration(n_repeats: int):
    """Empirical vs nominal central-interval coverage of the true PSD."""
    calibration = wdm_white_noise_calibration(N_TOTAL, DT, NT, CONFIG)
    cover = np.zeros((n_repeats, LEVELS.size))
    whitened_pool = []
    truth = None
    t0 = time.time()
    for rep in range(n_repeats):
        data = simulate_ls2(N_TOTAL, rng=np.random.default_rng(8000 + rep))
        res = run_wdm_psd_mcmc(data, dt=DT, nt=NT, config=CONFIG,
                               n_warmup=300, n_samples=400, random_seed=8000 + rep)
        log_samples = res["samples"]["log_psd"]  # (n_samp, nt, nf)
        if truth is None:
            truth = np.log(calibration[None, :]
                           * true_psd_ls2(res["time_grid"], res["freq_grid"], DT) + 1e-12)
        for k, lvl in enumerate(LEVELS):
            lo = np.percentile(log_samples, 50 * (1 - lvl), axis=0)
            hi = np.percentile(log_samples, 50 * (1 + lvl), axis=0)
            cover[rep, k] = np.mean((truth >= lo) & (truth <= hi))
        whitened_pool.append(
            (res["coeffs_fit"] / np.sqrt(res["psd_mean"])).reshape(-1)
        )
        if (rep + 1) % 10 == 0:
            print(f"[{rep + 1}/{n_repeats}] ({time.time() - t0:.0f}s)")
    return cover.mean(axis=0), cover.std(axis=0) / np.sqrt(n_repeats), np.concatenate(whitened_pool)


def wdm_coefficient_correlation(n_draws: int):
    """Empirical correlation of vectorised trimmed WDM coefficients (white noise)."""
    vectors = []
    for s in range(n_draws):
        coeffs, _, _ = wdm_analysis_coefficients(
            np.random.default_rng(s).standard_normal(N_TOTAL), DT, NT, CONFIG
        )
        vectors.append(coeffs.reshape(-1))
    corr = np.corrcoef(np.stack(vectors, axis=0), rowvar=False)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    return corr, float(np.max(np.abs(off))), float(np.median(np.abs(off)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--corr-draws", type=int, default=400)
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    emp, err, whitened = coverage_calibration(args.repeats)
    corr, max_off, med_off = wdm_coefficient_correlation(args.corr_draws)

    print("\n=== Coverage calibration ===")
    for lvl, e, s in zip(LEVELS, emp, err):
        print(f"  nominal {lvl:.2f}  ->  empirical {e:.3f} +/- {s:.3f}")
    print(f"=== WDM coefficient correlation: max|off|={max_off:.3f} median|off|={med_off:.3f}")
    print(f"=== Whitened residuals: mean={whitened.mean():.3f} var={whitened.var():.3f}")

    # Calibration figure.
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="ideal")
    ax.errorbar(LEVELS, emp, yerr=err, fmt="o-", color="tab:blue", capsize=3,
                label="WDM estimator")
    ax.set_xlabel("Nominal credible level")
    ax.set_ylabel("Empirical coverage of true PSD")
    ax.set_title(f"Coverage calibration ({args.repeats} LS2 realizations)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "calibration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Diagonal-Whittle diagnostics figure.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    lim = np.percentile(np.abs(corr[~np.eye(corr.shape[0], dtype=bool)]), 99)
    mesh = axes[0].imshow(corr, cmap="coolwarm", vmin=-lim, vmax=lim, origin="lower")
    axes[0].set_title(f"WDM coefficient correlation\nmax|off|={max_off:.3f}, "
                      f"median|off|={med_off:.3f}")
    axes[0].set_xlabel("vectorised pixel")
    axes[0].set_ylabel("vectorised pixel")
    fig.colorbar(mesh, ax=axes[0], label="correlation")

    q = np.linspace(0.01, 0.99, 200)
    axes[1].scatter(stats.norm.ppf(q), np.quantile(whitened, q), s=10,
                    color="tab:blue", alpha=0.6)
    lo, hi = stats.norm.ppf(0.01), stats.norm.ppf(0.99)
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1.0)
    axes[1].set_title(f"Whitened residuals $w/\\sqrt{{\\hat S}}$\n"
                      f"mean={whitened.mean():.2f}, var={whitened.var():.2f}")
    axes[1].set_xlabel("N(0,1) quantiles")
    axes[1].set_ylabel("empirical quantiles")
    fig.savefig(FIG_DIR / "whittle_diagnostics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved robustness figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
