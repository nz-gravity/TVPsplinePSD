"""WDM vs the Tang zigzag moving periodogram on LS2 across increasing durations.

The WDM estimator and the Tang dynamic-Whittle method (faithful thinned zigzag
moving periodogram) share the *same* whitened P-spline prior; they differ in the
time-frequency representation (WDM coefficients vs the moving-periodogram
ordinates) and likelihood (Gaussian coefficient vs exponential dynamic Whittle).
For a sweep of total durations we report the median log-PSD error of each method,
scored on a common grid after rescaling to the analytic PSD scale.

Saves ``notes/figures/duration_comparison.png``. Requires the project installed.

    python notes/scripts/make_duration_comparison.py --repeats 12
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from datasets import simulate_ls2, true_psd_ls2, wdm_white_noise_calibration
from wdm_psd import (
    PSplineConfig,
    run_tang_dynamic_whittle_mcmc,
    run_wdm_psd_mcmc,
    tang_moving_periodogram,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

DT = 0.1
NF = 24                      # WDM frequency tiling (data grows by adding time bins)
NT_VALUES = (24, 48, 96, 192)
TANG_M, TANG_THIN = 16, 2    # Tang moving-periodogram order and thinning factor
U_COMMON = np.linspace(0.05, 0.95, 40)
F_COMMON = np.linspace(0.6, 4.4, 40)

WDM_CONFIG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10)
TANG_CONFIG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=6)


def _to_common(time_grid, freq_grid, est_f0):
    interp = RegularGridInterpolator(
        (time_grid, freq_grid), np.log(est_f0 + 1e-12),
        bounds_error=False, fill_value=None,
    )
    uu, ff = np.meshgrid(U_COMMON, F_COMMON, indexing="ij")
    return interp(np.stack([uu.ravel(), ff.ravel()], axis=-1)).reshape(uu.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log_f0_common = np.log(true_psd_ls2(U_COMMON, F_COMMON, DT) + 1e-12)

    # Tang white-noise level (E[MI] for unit white noise ~ 1/(2 pi), per ordinate).
    cal_tang = float(np.mean(tang_moving_periodogram(
        np.random.default_rng(1).standard_normal(8192), m=TANG_M, thin=TANG_THIN)["mi"]))

    durations, wdm_med, tang_med = [], [], []
    div_total = 0
    for nt in NT_VALUES:
        n_total = nt * NF
        cal_wdm = wdm_white_noise_calibration(n_total, DT, nt, WDM_CONFIG)

        rep_wdm, rep_tang = [], []
        t0 = time.time()
        for rep in range(args.repeats):
            data = simulate_ls2(n_total, rng=np.random.default_rng(6000 + rep))
            rw = run_wdm_psd_mcmc(data, dt=DT, nt=nt, config=WDM_CONFIG,
                                  n_warmup=250, n_samples=250, random_seed=6000 + rep)
            rt = run_tang_dynamic_whittle_mcmc(data, dt=DT, m=TANG_M, thin=TANG_THIN,
                                               config=TANG_CONFIG, n_warmup=250,
                                               n_samples=250, random_seed=6000 + rep)
            div_total += rw["divergences"] + rt["divergences"]
            fw = _to_common(rw["time_grid"], rw["freq_grid"], rw["psd_mean"] / cal_wdm[None, :])
            ft = _to_common(rt["time_grid"], rt["freq_grid"], rt["psd_mean"] / cal_tang)
            rep_wdm.append(float(np.mean((fw - log_f0_common) ** 2)))
            rep_tang.append(float(np.mean((ft - log_f0_common) ** 2)))
        durations.append(n_total)
        wdm_med.append(np.median(rep_wdm))
        tang_med.append(np.median(rep_tang))
        print(f"n={n_total:6d}  WDM={wdm_med[-1]:.3f}  Tang={tang_med[-1]:.3f}  "
              f"({time.time() - t0:.0f}s)")

    print(f"total divergences across all fits: {div_total}")

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.loglog(durations, wdm_med, "o-", color="tab:blue", lw=2.0, label="WDM")
    ax.loglog(durations, tang_med, "s--", color="tab:orange", lw=2.0,
              label="Tang moving periodogram")
    ax.set_xlabel("Total samples $n$")
    ax.set_ylabel(r"median $\mathrm{MSE}_{\log f}$ (common grid)")
    ax.set_title("LS2 recovery vs. duration: WDM vs Tang")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(FIG_DIR / "duration_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    np.savez(FIG_DIR / "duration_comparison_metrics.npz",
             n_total=np.asarray(durations), wdm=np.asarray(wdm_med), tang=np.asarray(tang_med))
    print(f"Saved duration comparison to {FIG_DIR}")


if __name__ == "__main__":
    main()
