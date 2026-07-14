"""Known-truth coverage benchmark for adaptive frequency knots.

This is a deliberately small, representation-level experiment: it generates
independent real coefficients directly from the likelihood assumed by
``fit_log_pspline_surface``, with a smooth LISA-like spectrum containing four
narrow, drifting transfer-function notches.  The current uniform-frequency
basis and a pilot-adaptive frequency basis use the same knot count and are fit
to the same coefficients in every replicate.

Knot allocation uses a separate pilot realization.  Thus posterior coverage
is not helped by reusing the inference data during empirical-Bayes selection;
the comparison isolates basis allocation at a fixed parameter budget.

Run the short integration check or the intended compact benchmark with:

    uv run python studies/knot_coverage_simulation.py --quick
    uv run python studies/knot_coverage_simulation.py

The script requires the explicit-knot inference interface:
``fit_log_pspline_surface(..., interior_knots_freq=...)``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tv_pspline_psd import PSplineConfig, fit_log_pspline_surface
from tv_pspline_psd.adaptive_knots import fit_adaptive_knots

RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "studies" / "results" / "knot_coverage"
)

F_MIN = 1.0e-4
F_MAX = 0.133
NULL_FREQUENCIES = np.array([0.03, 0.06, 0.09, 0.12])
NULL_WIDTH_HZ = 1.8e-3
NULL_DRIFT_FRACTION = 8.0e-3
NULL_LOG_DEPTH = 1.8


@dataclass(frozen=True)
class StrategyMetrics:
    """Scalar metrics from one posterior fit."""

    coverage: float
    notch_coverage: float
    off_notch_coverage: float
    log_mse: float
    notch_log_mse: float
    interval_width: float
    notch_interval_width: float
    divergences: int
    runtime_s: float


def true_log_psd(time_grid: np.ndarray, freq_grid: np.ndarray) -> np.ndarray:
    """Smooth broadband surface with four narrow, time-drifting notches."""
    u = np.asarray(time_grid, dtype=float)[:, None]
    f = np.asarray(freq_grid, dtype=float)[None, :]
    x = (f - F_MIN) / (F_MAX - F_MIN)

    # A gently sloped broadband spectrum plus mild time modulation.  Overall
    # scale is arbitrary because the estimator and allocator are scale-free.
    eta = -0.2 + 0.8 * (x - 0.5) + 0.16 * np.cos(2.0 * np.pi * u) * (1.2 - x)
    for index, f0 in enumerate(NULL_FREQUENCIES):
        phase = 0.35 * index
        center = f0 * (1.0 + NULL_DRIFT_FRACTION * np.sin(2.0 * np.pi * u + phase))
        eta -= NULL_LOG_DEPTH * np.exp(
            -0.5 * ((f - center) / NULL_WIDTH_HZ) ** 2
        )
    return eta


def notch_mask(time_grid: np.ndarray, freq_grid: np.ndarray) -> np.ndarray:
    """Cells within two Gaussian widths of any instantaneous notch center."""
    u = np.asarray(time_grid, dtype=float)[:, None]
    f = np.asarray(freq_grid, dtype=float)[None, :]
    mask = np.zeros((u.size, f.size), dtype=bool)
    for index, f0 in enumerate(NULL_FREQUENCIES):
        center = f0 * (
            1.0 + NULL_DRIFT_FRACTION * np.sin(2.0 * np.pi * u + 0.35 * index)
        )
        mask |= np.abs(f - center) <= 2.0 * NULL_WIDTH_HZ
    return mask


def simulate_coefficients(
    log_psd: np.ndarray, rng: np.random.Generator, n_components: int
) -> np.ndarray:
    """Draw independent real Gaussian coefficients on the supplied grid."""
    scale = np.exp(0.5 * log_psd)
    return rng.standard_normal((n_components, *log_psd.shape)) * scale[None, :, :]


def posterior_metrics(
    result: dict[str, object], truth_log: np.ndarray, in_notch: np.ndarray
) -> StrategyMetrics:
    """Compute pointwise 90% coverage, width, and bias summaries."""
    lower = np.asarray(result["log_psd_lower"])
    mean = np.asarray(result["log_psd_mean"])
    upper = np.asarray(result["log_psd_upper"])
    covered = (truth_log >= lower) & (truth_log <= upper)
    squared_error = (mean - truth_log) ** 2
    width = upper - lower
    return StrategyMetrics(
        coverage=float(np.mean(covered)),
        notch_coverage=float(np.mean(covered[in_notch])),
        off_notch_coverage=float(np.mean(covered[~in_notch])),
        log_mse=float(np.mean(squared_error)),
        notch_log_mse=float(np.mean(squared_error[in_notch])),
        interval_width=float(np.mean(width)),
        notch_interval_width=float(np.mean(width[in_notch])),
        divergences=int(result["divergences"]),
        runtime_s=float(result["nuts_runtime_s"]),
    )


def summarize(records: list[dict[str, object]], strategy: str) -> dict[str, float]:
    """Summarize replicate metrics without hiding the paired raw values."""
    names = tuple(StrategyMetrics.__dataclass_fields__)
    values = {
        name: np.asarray([record[strategy][name] for record in records], dtype=float)
        for name in names
    }
    return {
        **{f"mean_{name}": float(np.mean(value)) for name, value in values.items()},
        **{
            f"sd_{name}": float(np.std(value, ddof=1)) if value.size > 1 else 0.0
            for name, value in values.items()
        },
        "total_divergences": int(np.sum(values["divergences"])),
    }


def paired_improvement(
    records: list[dict[str, object]], metric: str, *, larger_is_better: bool
) -> dict[str, float]:
    """Adaptive minus current paired change, sign-adjusted as improvement."""
    current = np.asarray([r["current"][metric] for r in records], dtype=float)
    adaptive = np.asarray([r["adaptive_freq"][metric] for r in records], dtype=float)
    delta = adaptive - current if larger_is_better else current - adaptive
    return {
        "mean": float(np.mean(delta)),
        "median": float(np.median(delta)),
        "fraction_positive": float(np.mean(delta > 0.0)),
    }


def run_benchmark(
    *,
    repeats: int,
    n_time: int,
    n_freq: int,
    n_time_knots: int,
    n_freq_knots: int,
    n_warmup: int,
    n_samples: int,
    pilot_components: int,
    seed: int,
) -> dict[str, object]:
    """Run paired current/adaptive fits over deterministic seeded replicates."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    time_grid = (np.arange(n_time) + 0.5) / n_time
    freq_grid = np.linspace(F_MIN, F_MAX, n_freq)
    truth_log = true_log_psd(time_grid, freq_grid)
    in_notch = notch_mask(time_grid, freq_grid)
    config = PSplineConfig(
        n_interior_knots_time=n_time_knots,
        n_interior_knots_freq=n_freq_knots,
        adaptive_time_knots=False,
        centered=True,
    )

    records: list[dict[str, object]] = []
    adaptive_knots: list[np.ndarray] = []
    started = time.perf_counter()
    for repeat in range(repeats):
        # Independent pilot and inference coefficients are intentional.  Both
        # strategies below then see exactly the same inference realization.
        pilot_rng = np.random.default_rng(seed + 10 * repeat)
        data_rng = np.random.default_rng(seed + 10 * repeat + 1)
        pilot_coeffs = simulate_coefficients(truth_log, pilot_rng, pilot_components)
        coeffs = simulate_coefficients(truth_log, data_rng, 1)
        allocation = fit_adaptive_knots(
            np.sum(pilot_coeffs**2, axis=0),
            time_grid,
            freq_grid,
            counts=float(pilot_components),
            n_pilot_knots_time=max(n_time_knots, 10),
            n_pilot_knots_freq=max(n_freq_knots + 12, 40),
            n_knots_time=n_time_knots,
            n_knots_freq=n_freq_knots,
            method="curvature",
            density_floor=0.2,
            mixed_weight=0.25,
            min_spacing_fraction=0.15,
            penalty_time=0.02,
            penalty_freq=0.02,
        )
        adaptive_knots.append(allocation.freq_knots)

        fits: dict[str, StrategyMetrics] = {}
        for offset, (strategy, freq_knots) in enumerate(
            (("current", None), ("adaptive_freq", allocation.freq_knots))
        ):
            fit = fit_log_pspline_surface(
                coeffs,
                time_grid,
                freq_grid,
                config=config,
                interior_knots_freq=freq_knots,
                n_warmup=n_warmup,
                n_samples=n_samples,
                num_chains=1,
                random_seed=seed + 10 * repeat + 2 + offset,
                target_accept_prob=0.9,
                progress_bar=False,
            )
            fits[strategy] = posterior_metrics(fit, truth_log, in_notch)

        record: dict[str, object] = {
            "repeat": repeat,
            "pilot_success": bool(allocation.pilot.success),
            "current": asdict(fits["current"]),
            "adaptive_freq": asdict(fits["adaptive_freq"]),
        }
        records.append(record)
        print(
            f"[{repeat + 1:02d}/{repeats}] "
            f"coverage current/adaptive="
            f"{fits['current'].coverage:.3f}/{fits['adaptive_freq'].coverage:.3f}  "
            f"notch={fits['current'].notch_coverage:.3f}/"
            f"{fits['adaptive_freq'].notch_coverage:.3f}  "
            f"notch MSE={fits['current'].notch_log_mse:.3f}/"
            f"{fits['adaptive_freq'].notch_log_mse:.3f}"
        )

    arguments = {
        "repeats": repeats,
        "n_time": n_time,
        "n_freq": n_freq,
        "n_time_knots": n_time_knots,
        "n_freq_knots": n_freq_knots,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "pilot_components": pilot_components,
        "seed": seed,
        "nominal_coverage": 0.9,
    }
    return {
        "arguments": arguments,
        "truth": {
            "frequency_range_hz": [F_MIN, F_MAX],
            "null_frequencies_hz": NULL_FREQUENCIES.tolist(),
            "null_width_hz": NULL_WIDTH_HZ,
            "null_drift_fraction": NULL_DRIFT_FRACTION,
            "null_log_depth": NULL_LOG_DEPTH,
            "notch_cell_fraction": float(np.mean(in_notch)),
        },
        "summary": {
            "current": summarize(records, "current"),
            "adaptive_freq": summarize(records, "adaptive_freq"),
            "paired_improvement": {
                "coverage": paired_improvement(
                    records, "coverage", larger_is_better=True
                ),
                "notch_coverage": paired_improvement(
                    records, "notch_coverage", larger_is_better=True
                ),
                "log_mse": paired_improvement(
                    records, "log_mse", larger_is_better=False
                ),
                "notch_log_mse": paired_improvement(
                    records, "notch_log_mse", larger_is_better=False
                ),
            },
        },
        "records": records,
        "adaptive_frequency_knots_hz": np.asarray(adaptive_knots).tolist(),
        "wall_time_s": float(time.perf_counter() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=7310)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    settings = {
        "repeats": args.repeats,
        "n_time": 24,
        "n_freq": 128,
        "n_time_knots": 6,
        "n_freq_knots": 24,
        "n_warmup": 200,
        "n_samples": 250,
        "pilot_components": 4,
        "seed": args.seed,
    }
    if args.quick:
        settings.update(
            repeats=min(args.repeats, 2),
            n_warmup=60,
            n_samples=80,
            pilot_components=3,
        )

    output = run_benchmark(**settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "quick" if args.quick else "full"
    json_path = args.output_dir / f"knot_coverage_{suffix}.json"
    with json_path.open("w") as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
