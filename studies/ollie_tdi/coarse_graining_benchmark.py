"""Benchmark exact, uniform, and adaptive coarse-grained WDM likelihoods.

The benchmark uses the cached 30-day A2 WDM coefficients produced by
``knot_map_benchmark.py``. Posterior fitting is performed on reduced likelihood
grids, but every accuracy and whitening comparison is evaluated on the original
120 x 5396 grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface
from tv_pspline_psd.inference import adaptive_frequency_bin_starts
from tv_pspline_psd.model import power_floor

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = Path(
    "/Users/avi/Documents/projects/wdm_psd/studies/results/ollie_tdi/"
    "knot_map_benchmark/wdm_coeffs.npz"
)
DEFAULT_OUTPUT = REPO / "studies/results/ollie_tdi/coarse_graining_benchmark.json"
NULLS = np.arange(0.03, 0.121, 0.03)


def _posterior_sd(result: dict[str, object]) -> np.ndarray:
    return (
        np.asarray(result["log_psd_upper"])
        - np.asarray(result["log_psd_lower"])
    ) / (2.0 * 1.6448536269514722)


def _whitening_metrics(
    coeffs: np.ndarray, log_psd: np.ndarray, null_mask: np.ndarray
) -> dict[str, float]:
    z2 = coeffs**2 * np.exp(-log_psd)
    return {
        "mean_z2": float(np.mean(z2)),
        "time_z2_rmse": float(np.sqrt(np.mean((np.mean(z2, axis=1) - 1.0) ** 2))),
        "freq_z2_rmse": float(np.sqrt(np.mean((np.mean(z2, axis=0) - 1.0) ** 2))),
        "null_mean_z2": float(np.mean(z2[:, null_mask])),
        "non_null_mean_z2": float(np.mean(z2[:, ~null_mask])),
    }


def _comparison(
    result: dict[str, object],
    exact: dict[str, object],
    coeffs: np.ndarray,
    null_mask: np.ndarray,
) -> dict[str, object]:
    estimate = np.asarray(result["log_psd_mean"])
    reference = np.asarray(exact["log_psd_mean"])
    delta = estimate - reference
    pooled_sd = np.sqrt(
        0.5 * (_posterior_sd(result) ** 2 + _posterior_sd(exact) ** 2)
    )
    normalized = delta / np.maximum(pooled_sd, 1e-12)
    samples = result["samples"]
    exact_samples = exact["samples"]
    phi_shift: dict[str, float] = {}
    for name in ("phi_time", "phi_freq"):
        x = np.asarray(samples[name], dtype=float)
        y = np.asarray(exact_samples[name], dtype=float)
        pooled = np.sqrt(0.5 * (np.var(x) + np.var(y)))
        phi_shift[name] = float(abs(np.mean(x) - np.mean(y)) / max(pooled, 1e-12))
    extra = result["mcmc"].get_extra_fields()
    return {
        "likelihood_grid_shape": list(result["likelihood_grid_shape"]),
        "cell_reduction": float(coeffs.size / np.prod(result["likelihood_grid_shape"])),
        "nuts_runtime_s": float(result["nuts_runtime_s"]),
        "mean_num_steps": float(np.mean(np.asarray(extra["num_steps"]))),
        "divergences": int(result["divergences"]),
        "log_surface_rms": float(np.sqrt(np.mean(delta**2))),
        "normalized_log_surface_rms": float(np.sqrt(np.mean(normalized**2))),
        "fraction_beyond_one_pooled_sd": float(np.mean(np.abs(normalized) > 1.0)),
        "null_log_surface_rms": float(np.sqrt(np.mean(delta[:, null_mask] ** 2))),
        "non_null_log_surface_rms": float(np.sqrt(np.mean(delta[:, ~null_mask] ** 2))),
        "log_phi_shift_pooled_sd": phi_shift,
        "whitening": _whitening_metrics(coeffs, estimate, null_mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--uniform-freq-bin", type=int, default=12)
    parser.add_argument("--adaptive-log-range", type=float, default=0.3)
    parser.add_argument("--adaptive-max-bin", type=int, default=64)
    args = parser.parse_args()

    cached = np.load(args.cache)
    coeffs = np.asarray(cached["coeffs"], dtype=float)
    time_grid = np.asarray(cached["time_grid"], dtype=float)
    freq_grid = np.asarray(cached["freq_grid"], dtype=float)
    power = coeffs**2
    null_mask = np.min(np.abs(freq_grid[:, None] - NULLS[None, :]), axis=1) < 0.002

    # This pilot is deliberately cheap and much smoother than the chi-square-1
    # powers. It is used only to allocate bins; NUTS still sees summed raw power.
    pilot = gaussian_filter(
        np.log(power + power_floor(power)), sigma=(3.0, 12.0), mode="nearest"
    )
    adaptive_starts = adaptive_frequency_bin_starts(
        pilot,
        max_log_range=args.adaptive_log_range,
        max_bin=args.adaptive_max_bin,
    )
    adaptive_widths = np.diff(np.r_[adaptive_starts, freq_grid.size])
    channel_to_bin = np.searchsorted(
        adaptive_starts, np.arange(freq_grid.size), side="right"
    ) - 1

    config = PSplineConfig(
        n_interior_knots_time=16,
        n_interior_knots_freq=94,
        adaptive_time_knots=False,
        centered=True,
    )
    common = dict(
        coeffs=coeffs[None],
        time_grid=time_grid,
        freq_grid=freq_grid,
        config=config,
        n_warmup=args.warmup,
        n_samples=args.samples,
        num_chains=1,
        random_seed=17,
        progress_bar=True,
    )
    specifications = {
        "exact": {},
        "uniform_frequency": {"freq_bin": args.uniform_freq_bin},
        "adaptive_frequency": {"freq_bin_starts": adaptive_starts},
        "adaptive_frequency_time2": {
            "freq_bin_starts": adaptive_starts,
            "time_bin": 2,
        },
    }

    fits: dict[str, dict[str, object]] = {}
    for name, options in specifications.items():
        option_summary = {
            key: (f"{value.size} variable bins" if isinstance(value, np.ndarray) else value)
            for key, value in options.items()
        }
        print(f"\n[{name}] {option_summary}", flush=True)
        fits[name] = fit_log_pspline_surface(**common, **options)
        print(
            f"[{name}] grid={fits[name]['likelihood_grid_shape']} "
            f"runtime={fits[name]['nuts_runtime_s']:.2f}s",
            flush=True,
        )

    exact = fits["exact"]
    report: dict[str, object] = {
        "source_cache": str(args.cache),
        "fine_grid_shape": list(coeffs.shape),
        "configuration": {
            "warmup": args.warmup,
            "samples": args.samples,
            "uniform_freq_bin": args.uniform_freq_bin,
            "adaptive_log_range": args.adaptive_log_range,
            "adaptive_max_bin": args.adaptive_max_bin,
        },
        "adaptive_bins": {
            "count": int(adaptive_starts.size),
            "mean_width": float(np.mean(adaptive_widths)),
            "median_width": float(np.median(adaptive_widths)),
            "max_width": int(np.max(adaptive_widths)),
            "mean_width_null_corridors": float(
                np.mean(adaptive_widths[channel_to_bin[null_mask]])
            ),
            "mean_width_elsewhere": float(
                np.mean(adaptive_widths[channel_to_bin[~null_mask]])
            ),
        },
        "fits": {},
    }
    for name, result in fits.items():
        comparison = _comparison(result, exact, coeffs, null_mask)
        comparison["speedup_vs_exact"] = float(
            exact["nuts_runtime_s"] / result["nuts_runtime_s"]
        )
        report["fits"][name] = comparison

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {args.output}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
