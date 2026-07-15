"""Centered-NUTS confirmation of frequency-knot allocation on 30-day A2.

The MAP benchmark is the cheap strategy screen.  This script performs the
smaller confirmatory comparison: the current adaptive-time/uniform-frequency
grid versus MAP-selected explicit frequency knots.  Both fits use exactly the
same current time knots and the same blocked held-out WDM time rows, isolating
frequency allocation.  Held-out scores integrate over posterior surface draws.

The production fitter must accept physical ``interior_knots_time`` and
``interior_knots_freq`` keyword arrays.  Until that integration lands, this
script exits before starting NUTS with a precise compatibility error.

Run (after ``knot_map_benchmark.py`` has populated the coefficient cache):

    uv run python studies/ollie_tdi/knot_nuts_confirmation.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import kurtosis

from tv_pspline_psd import (
    PSplineConfig,
    fit_log_pspline_surface,
    summarize_mcmc_diagnostics,
)
from tv_pspline_psd.adaptive_knots import (
    fit_adaptive_knots,
    fit_running_median_chi2_knots,
)
from tv_pspline_psd.inference import reconstruct_eig_coeff_samples
from tv_pspline_psd.model import power_floor
from tv_pspline_psd.splines import create_adaptive_time_knots, evaluate_bspline_basis

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "studies" / "results" / "ollie_tdi" / "knot_map_benchmark"
CACHE = RESULTS / "wdm_coeffs.npz"
OUTDIR = RESULTS / "nuts_confirmation"
LOG_2PI = float(np.log(2.0 * np.pi))


def holdout_rows(n_time: int, offset: int = 4) -> np.ndarray:
    """The same deterministic two-row-in-ten blocks as the MAP screen."""
    index = np.arange(n_time)
    mask = ((index % 10) == offset) | ((index % 10) == ((offset + 1) % 10))
    mask[[0, -1]] = False
    return mask


def explicit_knot_api_check() -> None:
    parameters = inspect.signature(fit_log_pspline_surface).parameters
    needed = {"interior_knots_time", "interior_knots_freq"}
    missing = sorted(needed - parameters.keys())
    if missing:
        raise RuntimeError(
            "Explicit-knot inference integration has not landed: "
            f"fit_log_pspline_surface is missing {missing}. Do not launch NUTS yet."
        )


def posterior_predictive(
    results: dict[str, object],
    coeffs_test: np.ndarray,
    time_test: np.ndarray,
    *,
    freq_chunk: int = 256,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Held-out posterior predictive score and predictive variance surface."""
    config: PSplineConfig = results["config"]  # type: ignore[assignment]
    eig = reconstruct_eig_coeff_samples(
        results["samples"], results["whitened"], config  # type: ignore[arg-type]
    )
    Bt = evaluate_bspline_basis(
        time_test, np.asarray(results["knots_time"]), degree=config.degree_time
    )
    Bt_eig = Bt @ results["whitened"]["U_time"]  # type: ignore[index]
    Bf_eig = (
        np.asarray(results["B_freq"])
        @ results["whitened"]["U_freq"]  # type: ignore[index]
    )
    n_time, n_freq = coeffs_test.shape
    predictive_variance = np.empty((n_time, n_freq))
    log_predictive = np.empty((n_time, n_freq))
    for j0 in range(0, n_freq, freq_chunk):
        j1 = min(j0 + freq_chunk, n_freq)
        eta = np.einsum(
            "ta,nab,jb->ntj", Bt_eig, eig, Bf_eig[j0:j1], optimize=True
        )
        predictive_variance[:, j0:j1] = np.mean(np.exp(eta), axis=0)
        power = coeffs_test[:, j0:j1] ** 2
        log_pdf = -0.5 * (
            LOG_2PI + eta + power[None] * np.exp(np.clip(-eta, -100.0, 100.0))
        )
        log_predictive[:, j0:j1] = logsumexp(log_pdf, axis=0) - np.log(eta.shape[0])

    ratio = coeffs_test**2 / predictive_variance
    z = coeffs_test / np.sqrt(predictive_variance)
    freq = np.asarray(results["freq_grid"])
    nulls = np.arange(0.03, freq[-1] + 0.015, 0.03)
    null_region = np.min(np.abs(freq[:, None] - nulls[None, :]), axis=1) < 0.002
    log_s = np.log(predictive_variance)
    saturated = np.log(coeffs_test**2 + power_floor(coeffs_test**2)) + 1.0
    excess = 0.5 * (log_s + ratio - saturated)
    metrics = {
        "mean_negative_log_predictive_density": float(-np.mean(log_predictive)),
        "mean_excess_whittle_deviance": float(np.mean(excess)),
        "null_excess_whittle_deviance": float(np.mean(excess[:, null_region])),
        "non_null_excess_whittle_deviance": float(np.mean(excess[:, ~null_region])),
        "mean_z2": float(np.mean(ratio)),
        "time_z2_rmse": float(np.sqrt(np.mean((ratio.mean(axis=1) - 1.0) ** 2))),
        "freq_z2_rmse": float(np.sqrt(np.mean((ratio.mean(axis=0) - 1.0) ** 2))),
        "z_excess_kurtosis": float(kurtosis(z.ravel(), fisher=True, bias=False)),
    }
    return metrics, predictive_variance, log_predictive


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-knots", type=int, default=16)
    parser.add_argument("--freq-knots", type=int, default=94)
    parser.add_argument("--pilot-freq-knots", type=int, default=120)
    parser.add_argument("--adaptive-method", choices=("curvature", "deviance", "hybrid"), default="curvature")
    parser.add_argument("--pilot-penalty", type=float, default=2.0)
    parser.add_argument("--pilot-coarsen-freq", type=int, default=1)
    parser.add_argument("--chi2-window-hz", type=float, default=5e-4)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--holdout-offset", type=int, default=4, choices=range(10))
    parser.add_argument(
        "--strategies", nargs="+", choices=("current", "adaptive_map", "chi2_freq"),
        default=("current", "adaptive_map"),
    )
    parser.add_argument("--n-warmup", type=int, default=250)
    parser.add_argument("--n-samples", type=int, default=250)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--prepare-only", action="store_true",
                        help="select and save knots, but do not launch NUTS")
    args = parser.parse_args()

    explicit_knot_api_check()
    if not CACHE.exists():
        raise FileNotFoundError(
            f"Missing {CACHE}; run studies/ollie_tdi/knot_map_benchmark.py first."
        )
    cached = np.load(CACHE)
    coeffs = np.asarray(cached["coeffs"])
    time_grid = np.asarray(cached["time_grid"])
    freq_grid = np.asarray(cached["freq_grid"])
    if args.fmax is not None:
        keep = freq_grid <= args.fmax
        if np.count_nonzero(keep) < args.freq_knots + 8:
            raise ValueError("--fmax leaves too few channels for the requested knot count")
        coeffs = coeffs[:, keep]
        freq_grid = freq_grid[keep]
    test = holdout_rows(time_grid.size, args.holdout_offset)
    train = ~test
    power_train = coeffs[train] ** 2

    current_time_knots = create_adaptive_time_knots(
        time_grid[train],
        np.mean(np.log(power_train + power_floor(power_train)), axis=1),
        n_interior_knots=args.time_knots,
        smoothing_sigma=1.0,
        variation_floor=0.25,
    )
    uniform_freq_knots = np.linspace(
        freq_grid[0], freq_grid[-1], args.freq_knots + 2
    )[1:-1]
    pilot_started = time.perf_counter()
    allocation = fit_adaptive_knots(
        power_train, time_grid[train], freq_grid,
        n_pilot_knots_time=args.time_knots,
        n_pilot_knots_freq=args.pilot_freq_knots,
        n_knots_time=args.time_knots,
        n_knots_freq=args.freq_knots,
        method=args.adaptive_method,
        coarsen_freq=args.pilot_coarsen_freq,
        penalty_time=args.pilot_penalty,
        penalty_freq=args.pilot_penalty,
    )
    pilot_runtime = time.perf_counter() - pilot_started
    chi2_allocation = fit_running_median_chi2_knots(
        power_train,
        freq_grid,
        args.freq_knots,
        median_window_hz=args.chi2_window_hz,
    )

    config = PSplineConfig(
        n_interior_knots_time=args.time_knots,
        n_interior_knots_freq=args.freq_knots,
        freq_knot_strategy="linear",
        centered=True,
        trim_time_bins=4,
        trim_low_freq_channels=4,
        trim_high_freq_channels=1,
    )
    available_strategies = {
        "current": uniform_freq_knots,
        "adaptive_map": allocation.freq_knots,
        "chi2_freq": chi2_allocation.knots,
    }
    strategies = {name: available_strategies[name] for name in args.strategies}
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "cache": str(CACHE),
        "grid_shape": list(coeffs.shape),
        "frequency_band_hz": [float(freq_grid[0]), float(freq_grid[-1])],
        "holdout_rows": np.flatnonzero(test),
        "shared_time_knots": current_time_knots,
        "pilot": {
            "method": allocation.method,
            "runtime_s": pilot_runtime,
            "success": allocation.pilot.success,
            "objective": allocation.pilot.penalized_objective,
            "freq_knots": allocation.freq_knots,
        },
        "chi2": {
            "threshold": chi2_allocation.threshold,
            "residual_scale": chi2_allocation.residual_scale,
            "window_bins": chi2_allocation.window_bins,
            "freq_knots": chi2_allocation.knots,
        },
        "arguments": vars(args),
        "strategies": {},
    }
    if args.prepare_only:
        args.outdir.mkdir(parents=True, exist_ok=True)
        plan = args.outdir / "prepared_knots.json"
        plan.write_text(json.dumps(summary, indent=2, default=json_default))
        print(f"[prepared] {plan}; NUTS not launched")
        return
    for i, (name, freq_knots) in enumerate(strategies.items()):
        print(f"[nuts] {name}: {args.n_warmup} warmup + {args.n_samples} samples x {args.num_chains}")
        result = fit_log_pspline_surface(
            coeffs[train][None], time_grid[train], freq_grid,
            config=config,
            interior_knots_time=current_time_knots,
            interior_knots_freq=freq_knots,
            n_warmup=args.n_warmup,
            n_samples=args.n_samples,
            num_chains=args.num_chains,
            random_seed=args.seed + i,
            target_accept_prob=args.target_accept,
            max_tree_depth=10,
        )
        metrics, pred_var, log_pred = posterior_predictive(
            result, coeffs[test], time_grid[test]
        )
        diagnostics = summarize_mcmc_diagnostics(result)
        summary["strategies"][name] = {  # type: ignore[index]
            "frequency_knots": freq_knots,
            "nuts_runtime_s": result["nuts_runtime_s"],
            "diagnostics": diagnostics,
            "heldout": metrics,
        }
        np.savez_compressed(
            args.outdir / f"{name}_predictive.npz",
            time_grid=time_grid[test], freq_grid=freq_grid,
            predictive_variance=pred_var, log_predictive=log_pred,
            frequency_knots=freq_knots, time_knots=current_time_knots,
        )
        print(f"[score] {name}: {json.dumps(metrics, sort_keys=True)}")

    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default)
    )
    print(f"[out] {args.outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
