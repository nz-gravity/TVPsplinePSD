"""Head-to-head: WDM estimator vs the Tang zigzag moving-periodogram method.

Both methods fit the LS2 time-varying PSD with the *same* whitened P-spline
prior; they differ in the time-frequency representation (squared WDM coefficients
vs the thinned zigzag moving-periodogram ordinates) and the likelihood (Gaussian
coefficient vs exponential dynamic Whittle). Each posterior surface is converted
to the analytic PSD scale (via its white-noise calibration) and interpolated onto
a common grid before scoring.

Saves ``notes/figures/baseline_comparison.png``. Requires the project installed.

    python notes/scripts/make_baseline_comparison.py --repeats 25
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

DT, N_TOTAL = 0.1, 576
NT = 24                    # WDM time bins
TANG_M, TANG_THIN = 16, 2  # Tang moving-periodogram order and thinning
WDM_CONFIG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10)
TANG_CONFIG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=6)

# Common comparison grid (kept inside both native frequency ranges).
U_COMMON = np.linspace(0.05, 0.95, 40)
F_COMMON = np.linspace(0.6, 4.4, 40)


def _to_common_f0(time_grid, freq_grid, est_f0):
    """Interpolate a method's f0-scale estimate onto the common (u, f) grid."""
    interp = RegularGridInterpolator(
        (time_grid, freq_grid), np.log(est_f0 + 1e-12),
        bounds_error=False, fill_value=None,
    )
    uu, ff = np.meshgrid(U_COMMON, F_COMMON, indexing="ij")
    return interp(np.stack([uu.ravel(), ff.ravel()], axis=-1)).reshape(uu.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cal_wdm = wdm_white_noise_calibration(N_TOTAL, DT, NT, WDM_CONFIG)
    cal_tang = float(np.mean(tang_moving_periodogram(
        np.random.default_rng(1).standard_normal(8192), m=TANG_M, thin=TANG_THIN)["mi"]))

    log_f0_common = np.log(true_psd_ls2(U_COMMON, F_COMMON, DT) + 1e-12)

    mse_wdm, mse_tang, div_wdm, div_tang = [], [], 0, 0
    example = {}
    t0 = time.time()
    for rep in range(args.repeats):
        data = simulate_ls2(N_TOTAL, rng=np.random.default_rng(4000 + rep))

        rw = run_wdm_psd_mcmc(data, dt=DT, nt=NT, config=WDM_CONFIG,
                              n_warmup=250, n_samples=250, random_seed=4000 + rep)
        rt = run_tang_dynamic_whittle_mcmc(data, dt=DT, m=TANG_M, thin=TANG_THIN,
                                           config=TANG_CONFIG, n_warmup=250,
                                           n_samples=250, random_seed=4000 + rep)

        f0_wdm = _to_common_f0(rw["time_grid"], rw["freq_grid"],
                               rw["psd_mean"] / cal_wdm[None, :])
        f0_tang = _to_common_f0(rt["time_grid"], rt["freq_grid"], rt["psd_mean"] / cal_tang)
        mse_wdm.append(float(np.mean((f0_wdm - log_f0_common) ** 2)))
        mse_tang.append(float(np.mean((f0_tang - log_f0_common) ** 2)))
        div_wdm += rw["divergences"]
        div_tang += rt["divergences"]

        if rep == 0:
            example = {"wdm": f0_wdm, "tang": f0_tang}
        if (rep + 1) % 5 == 0:
            print(f"[{rep + 1}/{args.repeats}] WDM={mse_wdm[-1]:.3f} "
                  f"Tang={mse_tang[-1]:.3f} ({time.time() - t0:.0f}s)")

    mse_wdm, mse_tang = np.asarray(mse_wdm), np.asarray(mse_tang)
    print("\n=== WDM vs Tang dynamic Whittle (LS2) ===")
    print(f"WDM   MSE_log: median={np.median(mse_wdm):.3f} "
          f"[{np.percentile(mse_wdm, 5):.3f}, {np.percentile(mse_wdm, 95):.3f}]  div={div_wdm}")
    print(f"Tang  MSE_log: median={np.median(mse_tang):.3f} "
          f"[{np.percentile(mse_tang, 5):.3f}, {np.percentile(mse_tang, 95):.3f}]  div={div_tang}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].boxplot([mse_wdm, mse_tang], tick_labels=["WDM", "Tang"])
    axes[0].set_ylabel(r"$\mathrm{MSE}_{\log f}$ on common grid")
    axes[0].set_title(f"LS2 recovery error ({args.repeats} repeats)")

    vmin = min(log_f0_common.min(), example["wdm"].min(), example["tang"].min())
    vmax = max(log_f0_common.max(), example["wdm"].max(), example["tang"].max())
    for ax, field, title in [
        (axes[1], example["wdm"], "WDM posterior (f0 scale)"),
        (axes[2], example["tang"], "Tang dynamic Whittle (f0 scale)"),
    ]:
        mesh = ax.pcolormesh(U_COMMON, F_COMMON, field.T, shading="auto",
                             cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Rescaled time $u$")
        ax.set_ylabel("Frequency [Hz]")
        fig.colorbar(mesh, ax=ax, label="log PSD")
    fig.savefig(FIG_DIR / "baseline_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved baseline comparison to {FIG_DIR}")


if __name__ == "__main__":
    main()
