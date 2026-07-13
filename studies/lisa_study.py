"""Simulation study: non-stationary LISA noise with modulated confusion.

Drives the WDM log-P-spline estimator with instrument-plus-confusion noise where
the Galactic foreground amplitude is seasonally modulated. Produces surface and
channel-recovery figures and, optionally, a repeat loop for an error
distribution analogous to the LS2 study.

Run:
    python studies/lisa_study.py             # single rich fit + figures
    python studies/lisa_study.py --repeats 30
    python studies/lisa_study.py --quick
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tv_pspline_psd import (
    PSplineConfig,
    interval_coverage,
    mse_log_psd,
    plot_channel_slice,
    plot_surface_comparison,
    relative_surface_error,
    run_wdm_psd_mcmc,
    summarize_mcmc_diagnostics,
)
from tv_pspline_psd.datasets import (
    LISANoiseConfig,
    monte_carlo_reference,
    normalization_constant,
    simulate_tv_lisa_noise,
    true_psd_lisa,
    wdm_white_noise_calibration,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "studies" / "results" / "lisa"

# dt = 167 s places the WDM band on the confusion bump (Nyquist ~ 3 mHz).
DT = 167.0
NT = 24
N_TOTAL = 768
LISA_CONFIG = LISANoiseConfig(tobs_key="1yr", n_modulation_cycles=3.0)
PSPLINE_CONFIG = PSplineConfig(
    n_interior_knots_time=8, n_interior_knots_freq=10, trim_low_freq_channels=2
)


def _reference_and_truth(time_grid: np.ndarray, freq_grid: np.ndarray, n_draws: int):
    reference = monte_carlo_reference(
        lambda rng: simulate_tv_lisa_noise(N_TOTAL, dt=DT, rng=rng, config=LISA_CONFIG)[0],
        n_draws=n_draws, n_total=N_TOTAL, dt=DT, nt=NT, config=PSPLINE_CONFIG, seed=321,
    )
    norm_ref = normalization_constant(N_TOTAL, DT, LISA_CONFIG)
    calibration = wdm_white_noise_calibration(N_TOTAL, DT, NT, PSPLINE_CONFIG)
    true = calibration[None, :] * true_psd_lisa(
        time_grid, freq_grid, LISA_CONFIG, norm_ref=norm_ref
    )
    return reference, true


def single_fit(n_warmup: int, n_samples: int, n_reference_draws: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    data, _ = simulate_tv_lisa_noise(N_TOTAL, dt=DT, rng=rng, config=LISA_CONFIG)

    res = run_wdm_psd_mcmc(
        data, dt=DT, nt=NT, config=PSPLINE_CONFIG,
        n_warmup=n_warmup, n_samples=n_samples, num_chains=2, random_seed=21,
    )
    reference, true = _reference_and_truth(res["time_grid"], res["freq_grid"], n_reference_draws)
    diag = summarize_mcmc_diagnostics(res)

    print("=== LISA single fit ===")
    print(f"WDM grid (trimmed): {res['power'].shape}")
    print(f"divergences: {diag['divergences']}")
    print(f"rel err vs E[w^2]      : raw={relative_surface_error(reference, res['power']):.3f} "
          f"post={relative_surface_error(reference, res['psd_mean']):.3f}")
    print(f"MSE_log vs E[w^2]      : raw={mse_log_psd(reference, res['power']):.3f} "
          f"post={mse_log_psd(reference, res['psd_mean']):.3f}")
    print(f"MSE_log vs analytic S  : post={mse_log_psd(true, res['psd_mean']):.3f}")
    print(f"90% coverage vs E[w^2] : {interval_coverage(reference, res['psd_lower'], res['psd_upper']):.2f}")

    plot_surface_comparison(
        res, reference, freq_scale=1e3, freq_label="Frequency [mHz]",
        path=RESULTS_DIR / "lisa_surface_comparison.png",
    )
    # Pick the channel with the largest temporal coefficient of variation: the
    # confusion-dominated band where the seasonal modulation is most prominent
    # (absolute variance is largest at the steep low-f instrument channels).
    cv = reference.std(axis=0) / np.maximum(reference.mean(axis=0), 1e-30)
    channel = int(np.argmax(cv))
    plot_channel_slice(
        res, reference, channel, true_psd=true,
        freq_scale=1e3, freq_label="Confusion", path=RESULTS_DIR / "lisa_modulation_channel.png",
    )
    print(f"Saved figures to {RESULTS_DIR}")


def repeat_study(n_repeats: int, n_warmup: int, n_samples: int, n_reference_draws: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reference = true = None
    mse_ref, mse_true, rel_ref, coverage, divergences = [], [], [], [], []
    t0 = time.time()
    for rep in range(n_repeats):
        rng = np.random.default_rng(5000 + rep)
        data, _ = simulate_tv_lisa_noise(N_TOTAL, dt=DT, rng=rng, config=LISA_CONFIG)
        res = run_wdm_psd_mcmc(
            data, dt=DT, nt=NT, config=PSPLINE_CONFIG,
            n_warmup=n_warmup, n_samples=n_samples, num_chains=1, random_seed=5000 + rep,
        )
        if reference is None:
            reference, true = _reference_and_truth(res["time_grid"], res["freq_grid"], n_reference_draws)
        mse_ref.append(mse_log_psd(reference, res["psd_mean"]))
        mse_true.append(mse_log_psd(true, res["psd_mean"]))
        rel_ref.append(relative_surface_error(reference, res["psd_mean"]))
        coverage.append(interval_coverage(reference, res["psd_lower"], res["psd_upper"]))
        divergences.append(summarize_mcmc_diagnostics(res)["divergences"])
        if (rep + 1) % 5 == 0 or rep == 0:
            print(f"[{rep + 1:3d}/{n_repeats}] mse_log(ref)={mse_ref[-1]:.3f} "
                  f"rel_err={rel_ref[-1]:.3f} cov={coverage[-1]:.2f} ({time.time() - t0:.0f}s)")

    out = {
        "mse_log_reference": np.asarray(mse_ref),
        "mse_log_true": np.asarray(mse_true),
        "relative_error_reference": np.asarray(rel_ref),
        "coverage": np.asarray(coverage),
        "divergences": np.asarray(divergences),
    }
    np.savez(RESULTS_DIR / "lisa_metrics.npz", **out)

    def _q(x):
        p = np.percentile(x, [5, 50, 95])
        return f"median={p[1]:.3f} [p5={p[0]:.3f}, p95={p[2]:.3f}]"

    print("\n=== LISA simulation study ===")
    print(f"repeats: {n_repeats}")
    print(f"MSE_log vs E[w^2] reference : {_q(out['mse_log_reference'])}")
    print(f"MSE_log vs analytic S(u,f)  : {_q(out['mse_log_true'])}")
    print(f"relative error vs reference : {_q(out['relative_error_reference'])}")
    print(f"90% interval coverage       : {_q(out['coverage'])}")
    print(f"total divergences           : {int(out['divergences'].sum())}")

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.boxplot([out["mse_log_reference"], out["mse_log_true"]],
               tick_labels=["vs E[w^2] ref", "vs analytic S"])
    ax.set_ylabel(r"$\mathrm{MSE}_{\log f}$")
    ax.set_title(f"LISA confusion noise, {n_repeats} repeats")
    fig.savefig(RESULTS_DIR / "lisa_error_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved results to {RESULTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=0, help="0 => single rich fit only")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        single_fit(n_warmup=80, n_samples=80, n_reference_draws=40)
    elif args.repeats > 0:
        repeat_study(args.repeats, n_warmup=250, n_samples=250, n_reference_draws=120)
    else:
        single_fit(n_warmup=300, n_samples=300, n_reference_draws=200)


if __name__ == "__main__":
    main()
