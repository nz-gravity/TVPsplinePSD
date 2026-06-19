"""Minimal example: simulate LS2, estimate the time-varying PSD, plot it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets import monte_carlo_reference, simulate_ls2
from tv_pspline_psd import (
    PSplineConfig,
    plot_surface_comparison,
    relative_surface_error,
    run_wdm_psd_mcmc,
    summarize_mcmc_diagnostics,
)


def main() -> None:
    dt, nt, n_total = 0.1, 24, 576
    config = PSplineConfig()

    data = simulate_ls2(n_total, rng=np.random.default_rng(0))
    results = run_wdm_psd_mcmc(
        data, dt=dt, nt=nt, config=config, n_warmup=250, n_samples=250, num_chains=1
    )
    reference = monte_carlo_reference(
        lambda rng: simulate_ls2(n_total, rng=rng),
        n_draws=80, n_total=n_total, dt=dt, nt=nt, config=config, seed=7,
    )

    diag = summarize_mcmc_diagnostics(results)
    print(f"divergences: {diag['divergences']}")
    print(f"relative error vs E[w^2]: {relative_surface_error(reference, results['psd_mean']):.3f}")

    out = Path(__file__).resolve().parent / "quickstart_surface.png"
    plot_surface_comparison(results, reference, freq_label="Frequency [Hz]", path=out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
