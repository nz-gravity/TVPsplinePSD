"""Simulation study: LS2 locally stationary process, repeated estimation.

Fits the WDM log-P-spline estimator to many independent LS2 realizations and
reports the distribution of the log-PSD error. The primary target is the WDM
local power ``E[w_nm^2]`` (estimated once by Monte Carlo), which is what the
estimator actually infers. The analytic pointwise PSD is also reported; it is
converted to WDM-coefficient units via a per-channel white-noise calibration
(``E[w^2] = C_m * S_dig``), after which the two targets agree closely (residual
differences come only from atom-averaging where the PSD varies fast across a
WDM atom).

Run:
    python studies/ls2_simulation_study.py            # 100 repeats
    python studies/ls2_simulation_study.py --quick    # fast smoke run
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
from wdm_psd import (
    PSplineConfig,
    interval_coverage,
    mse_log_psd,
    relative_surface_error,
    run_wdm_psd_mcmc,
    summarize_mcmc_diagnostics,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "studies" / "results" / "ls2"


def run_study(
    *,
    n_repeats: int,
    n_total: int = 576,
    dt: float = 0.1,
    nt: int = 24,
    n_warmup: int = 300,
    n_samples: int = 300,
    n_reference_draws: int = 200,
    seed0: int = 1000,
) -> dict[str, np.ndarray]:
    """Run ``n_repeats`` LS2 fits and collect per-repeat error metrics."""
    config = PSplineConfig()

    # The WDM target E[w^2] and the analytic pointwise PSD live on the fixed,
    # deterministic trimmed grid, so they are computed once.
    reference_psd = monte_carlo_reference(
        lambda rng: simulate_ls2(n_total, rng=rng),
        n_draws=n_reference_draws,
        n_total=n_total,
        dt=dt,
        nt=nt,
        config=config,
        seed=7,
    )
    # Per-channel calibration to express the analytic digital-convention PSD in
    # the WDM-coefficient units the estimator infers (E[w^2] = C_m * S_dig).
    calibration = wdm_white_noise_calibration(n_total, dt, nt, config)

    mse_ref, mse_true, rel_ref, coverage, divergences = [], [], [], [], []
    true_psd = None
    t0 = time.time()
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed0 + rep)
        data = simulate_ls2(n_total, rng=rng)
        res = run_wdm_psd_mcmc(
            data, dt=dt, nt=nt, config=config,
            n_warmup=n_warmup, n_samples=n_samples,
            num_chains=1, random_seed=seed0 + rep,
        )
        if true_psd is None:
            true_psd = calibration[None, :] * true_psd_ls2(
                res["time_grid"], res["freq_grid"], dt
            )

        mse_ref.append(mse_log_psd(reference_psd, res["psd_mean"]))
        mse_true.append(mse_log_psd(true_psd, res["psd_mean"]))
        rel_ref.append(relative_surface_error(reference_psd, res["psd_mean"]))
        coverage.append(
            interval_coverage(reference_psd, res["psd_lower"], res["psd_upper"])
        )
        divergences.append(summarize_mcmc_diagnostics(res)["divergences"])

        if (rep + 1) % 10 == 0 or rep == 0:
            elapsed = time.time() - t0
            print(
                f"[{rep + 1:3d}/{n_repeats}] "
                f"mse_log(ref)={mse_ref[-1]:.3f}  rel_err={rel_ref[-1]:.3f}  "
                f"cov={coverage[-1]:.2f}  div={divergences[-1]}  "
                f"({elapsed:.0f}s)"
            )

    return {
        "mse_log_reference": np.asarray(mse_ref),
        "mse_log_true": np.asarray(mse_true),
        "relative_error_reference": np.asarray(rel_ref),
        "coverage": np.asarray(coverage),
        "divergences": np.asarray(divergences),
        "reference_psd": reference_psd,
        "true_psd": true_psd,
    }


def summarize_and_plot(results: dict[str, np.ndarray]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS_DIR / "ls2_metrics.npz", **results)

    def _q(x: np.ndarray) -> str:
        p = np.percentile(x, [5, 50, 95])
        return f"median={p[1]:.3f}  [p5={p[0]:.3f}, p95={p[2]:.3f}]"

    print("\n=== LS2 simulation study ===")
    print(f"repeats: {results['mse_log_reference'].size}")
    print(f"MSE_log vs E[w^2] reference : {_q(results['mse_log_reference'])}")
    print(f"MSE_log vs pointwise true   : {_q(results['mse_log_true'])}")
    print(f"relative error vs reference : {_q(results['relative_error_reference'])}")
    print(f"90% interval coverage       : {_q(results['coverage'])}")
    print(f"total divergences           : {int(results['divergences'].sum())}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].boxplot(
        [results["mse_log_reference"], results["mse_log_true"]],
        tick_labels=["vs E[w^2] ref", "vs pointwise true"],
    )
    axes[0].set_ylabel(r"$\mathrm{MSE}_{\log f}$")
    axes[0].set_title("LS2 log-PSD error across repeats")
    axes[1].hist(results["relative_error_reference"], bins=20, color="tab:blue", alpha=0.8)
    axes[1].set_xlabel("Relative surface error vs E[w^2]")
    axes[1].set_ylabel("count")
    axes[1].set_title("LS2 relative error distribution")
    fig.savefig(RESULTS_DIR / "ls2_error_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved results to {RESULTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    args = parser.parse_args()
    if args.quick:
        results = run_study(n_repeats=5, n_warmup=80, n_samples=80, n_reference_draws=40)
    else:
        results = run_study(n_repeats=args.repeats)
    summarize_and_plot(results)


if __name__ == "__main__":
    main()
