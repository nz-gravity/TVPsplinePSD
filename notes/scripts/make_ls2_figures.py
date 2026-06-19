"""Generate the LS2 figures used in the manuscript.

Saves into ``notes/figures/``. Requires the project to be installed
(``uv pip install -e .`` / ``uv sync``), so no path manipulation is needed.

    python notes/scripts/make_ls2_figures.py               # 100 repeats
    python notes/scripts/make_ls2_figures.py --repeats 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import (
    monte_carlo_reference,
    simulate_ls2,
    true_psd_ls2,
    wdm_white_noise_calibration,
)
from tv_pspline_psd import (
    PSplineConfig,
    interval_coverage,
    mse_log_psd,
    plot_surface_comparison,
    relative_surface_error,
    run_wdm_psd_mcmc,
    save_figure,
    summarize_mcmc_diagnostics,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

DT, NT, N_TOTAL = 0.1, 24, 576
CONFIG = PSplineConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    reference = monte_carlo_reference(
        lambda rng: simulate_ls2(N_TOTAL, rng=rng),
        n_draws=200, n_total=N_TOTAL, dt=DT, nt=NT, config=CONFIG, seed=7,
    )
    calibration = wdm_white_noise_calibration(N_TOTAL, DT, NT, CONFIG)

    # Representative single fit -> surface example figure.
    data = simulate_ls2(N_TOTAL, rng=np.random.default_rng(3))
    res = run_wdm_psd_mcmc(
        data, dt=DT, nt=NT, config=CONFIG, n_warmup=300, n_samples=300,
        num_chains=1, random_seed=3,
    )
    plot_surface_comparison(
        res, reference, freq_label="Frequency [Hz]",
        path=FIG_DIR / "ls2_surface_example.png",
    )
    true = calibration[None, :] * true_psd_ls2(res["time_grid"], res["freq_grid"], DT)

    # Repeated estimation -> error distribution figure.
    mse_ref, mse_true, coverage, divergences = [], [], [], []
    t0 = time.time()
    for rep in range(args.repeats):
        rng = np.random.default_rng(1000 + rep)
        d = simulate_ls2(N_TOTAL, rng=rng)
        r = run_wdm_psd_mcmc(
            d, dt=DT, nt=NT, config=CONFIG, n_warmup=300, n_samples=300,
            num_chains=1, random_seed=1000 + rep,
        )
        mse_ref.append(mse_log_psd(reference, r["psd_mean"]))
        mse_true.append(mse_log_psd(true, r["psd_mean"]))
        coverage.append(interval_coverage(reference, r["psd_lower"], r["psd_upper"]))
        divergences.append(summarize_mcmc_diagnostics(r)["divergences"])
        if (rep + 1) % 20 == 0:
            print(f"[{rep + 1}/{args.repeats}] ({time.time() - t0:.0f}s)")

    mse_ref = np.asarray(mse_ref)
    print(f"LS2 {args.repeats} repeats: MSE_log(ref) median={np.median(mse_ref):.3f}  "
          f"coverage median={np.median(coverage):.2f}  divergences={int(np.sum(divergences))}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].boxplot([mse_ref, np.asarray(mse_true)],
                    tick_labels=["vs $E[w^2]$ ref", "vs analytic PSD"])
    axes[0].set_ylabel(r"$\mathrm{MSE}_{\log f}$")
    axes[0].set_title(f"LS2 log-PSD error ({args.repeats} repeats)")
    axes[1].hist(coverage, bins=15, color="tab:blue", alpha=0.8)
    axes[1].axvline(0.9, color="black", ls="--", lw=1.2, label="nominal 90%")
    axes[1].set_xlabel("90% interval coverage")
    axes[1].set_ylabel("count")
    axes[1].legend()
    save_figure(fig, FIG_DIR / "ls2_error_distribution.png")
    print(f"Saved LS2 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
