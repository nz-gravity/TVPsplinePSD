"""Posterior contraction as the data increases (LS2).

Holds the model and frequency resolution fixed and increases the number of WDM
time bins (longer observation at fixed dt), so each fit observes the same smooth
``S(u, f)`` with progressively more data. Reports the median log-PSD error and
the mean posterior 90%-interval width versus the total sample size, which should
both decrease -- a direct demonstration of Bayesian posterior contraction.

Saves ``notes/figures/convergence.png``. Requires the project to be installed.

    python notes/scripts/make_convergence_figure.py
    python notes/scripts/make_convergence_figure.py --repeats 8   # faster
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wdm_transform import TimeSeries

from datasets import (
    monte_carlo_reference,
    simulate_ls2,
    true_psd_ls2,
    trimmed_keep_indices,
    wdm_white_noise_calibration,
)
from wdm_psd import (
    PSplineConfig,
    mse_log_psd,
    run_wdm_psd_mcmc,
    save_figure,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

DT = 0.1
NF = 32  # fixed frequency tiling; data grows by adding WDM time bins
NT_VALUES = (24, 48, 96, 192)
CONFIG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10)


def _trimmed_grids(n_total: int, nt: int):
    """Rescaled time grid and Hz frequency grid on the trimmed WDM tiling."""
    probe = TimeSeries(np.zeros(n_total), dt=DT).to_wdm(nt=nt)
    keep_time, keep_freq = trimmed_keep_indices(n_total, DT, nt, CONFIG)
    time_grid = np.asarray(probe.time_grid)[keep_time] / probe.duration
    freq_grid = np.asarray(probe.freq_grid)[keep_freq]
    return time_grid, freq_grid


def run_sweep(n_repeats: int) -> dict[str, np.ndarray]:
    n_totals, mse_ref, mse_true, widths = [], [], [], []
    for nt in NT_VALUES:
        n_total = nt * NF
        time_grid, freq_grid = _trimmed_grids(n_total, nt)
        reference = monte_carlo_reference(
            lambda rng: simulate_ls2(n_total, rng=rng),
            n_draws=80, n_total=n_total, dt=DT, nt=nt, config=CONFIG, seed=7,
        )
        calibration = wdm_white_noise_calibration(n_total, DT, nt, CONFIG)
        true = calibration[None, :] * true_psd_ls2(time_grid, freq_grid, DT)

        rep_mse_ref, rep_mse_true, rep_width = [], [], []
        t0 = time.time()
        for rep in range(n_repeats):
            data = simulate_ls2(n_total, rng=np.random.default_rng(2000 + rep))
            res = run_wdm_psd_mcmc(
                data, dt=DT, nt=nt, config=CONFIG,
                n_warmup=250, n_samples=250, num_chains=1, random_seed=2000 + rep,
            )
            rep_mse_ref.append(mse_log_psd(reference, res["psd_mean"]))
            rep_mse_true.append(mse_log_psd(true, res["psd_mean"]))
            rep_width.append(float(np.mean(
                np.log(res["psd_upper"] + 1e-12) - np.log(res["psd_lower"] + 1e-12)
            )))
        n_totals.append(n_total)
        mse_ref.append(np.median(rep_mse_ref))
        mse_true.append(np.median(rep_mse_true))
        widths.append(np.median(rep_width))
        print(f"nt={nt:4d}  n={n_total:6d}  "
              f"MSE_log(ref)={mse_ref[-1]:.3f}  MSE_log(true)={mse_true[-1]:.3f}  "
              f"width={widths[-1]:.3f}  ({time.time() - t0:.0f}s)")

    return {
        "n_total": np.asarray(n_totals),
        "mse_log_reference": np.asarray(mse_ref),
        "mse_log_true": np.asarray(mse_true),
        "interval_width": np.asarray(widths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    out = run_sweep(args.repeats)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].loglog(out["n_total"], out["mse_log_reference"], "o-", color="tab:blue",
                   lw=2.0, label=r"vs $E[w^2]$ reference")
    axes[0].loglog(out["n_total"], out["mse_log_true"], "s--", color="tab:green",
                   lw=1.8, label="vs analytic PSD")
    axes[0].set_xlabel("Total samples $n$")
    axes[0].set_ylabel(r"median $\mathrm{MSE}_{\log f}$")
    axes[0].set_title("Error vs. data size")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(out["n_total"], out["interval_width"], "o-", color="tab:purple", lw=2.0)
    axes[1].set_xlabel("Total samples $n$")
    axes[1].set_ylabel("median 90% interval width (log PSD)")
    axes[1].set_title("Posterior contraction")
    axes[1].grid(True, which="both", alpha=0.3)

    save_figure(fig, FIG_DIR / "convergence.png")
    np.savez(FIG_DIR / "convergence_metrics.npz", **out)
    print(f"Saved convergence figure to {FIG_DIR}")


if __name__ == "__main__":
    main()
