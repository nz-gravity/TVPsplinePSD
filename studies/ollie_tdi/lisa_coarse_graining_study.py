"""Paired LISA coarse-graining study, including a gap stress test.

The cached 30-day A-channel WDM coefficients are fitted repeatedly with the
same spline model and different likelihood partitions.  Each coarse fit is
compared with an exact fit of the *same retained coefficients*.  The gapped
condition removes the WDM rows that would be discarded after tapered gap
gating; time pooling is restarted on both sides of every missing interval.

Every fit is saved independently, so rerunning this command resumes after the
last completed setting rather than repeating the whole study.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter

from tv_pspline_psd import (
    PSplineConfig,
    fit_log_pspline_surface,
    gap_aware_time_bin_starts,
    set_paper_style,
    summarize_mcmc_diagnostics,
)
from tv_pspline_psd.inference import (
    adaptive_frequency_bin_starts,
    bin_power_rectangular,
)
from tv_pspline_psd.model import power_floor

from fit_aet_fullband import lisa_like_gaps


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "studies/results/ollie_tdi/knot_map_benchmark/wdm_coeffs.npz"
DEFAULT_OUTPUT = REPO / "studies/results/ollie_tdi/coarse_graining_lisa"
PAPER_FIGURE = REPO / "overleaf/figures/coarse_graining_cells.png"
OBSERVATION_DAYS = 30.0
NULLS = np.arange(0.03, 0.101, 0.03)


def _gap_mask(time_grid: np.ndarray, seed: int) -> tuple[np.ndarray, list[list[float]]]:
    """Mimic the rows retained after the existing tapered-gap pipeline."""
    t_obs_s = OBSERVATION_DAYS * 86400.0
    gaps = lisa_like_gaps(t_obs_s, seed=seed)
    centers_s = time_grid * t_obs_s
    nominal_bin_s = float(np.median(np.diff(centers_s)))
    taper_s = 3600.0
    keep = np.ones(time_grid.size, dtype=bool)
    for start, stop in gaps:
        keep &= (
            (centers_s < start - taper_s - nominal_bin_s)
            | (centers_s > stop + taper_s + nominal_bin_s)
        )
    return keep, [[float(start), float(stop)] for start, stop in gaps]


def _contiguous_runs(time_grid: np.ndarray) -> list[tuple[int, int]]:
    if time_grid.size == 1:
        return [(0, 1)]
    steps = np.diff(time_grid)
    breaks = np.flatnonzero(steps > 1.5 * np.median(steps)) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, time_grid.size]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _pilot_log_psd(power: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    """Smooth each contiguous time run independently, never across a gap."""
    log_power = np.log(power + power_floor(power))
    pilot = np.empty_like(log_power)
    for start, stop in _contiguous_runs(time_grid):
        pilot[start:stop] = gaussian_filter(
            log_power[start:stop], sigma=(3.0, 12.0), mode="nearest"
        )
    return pilot


def _binning_metadata(tolerance: float, max_bin: int) -> dict[str, Any]:
    return {
        "frequency": {
            "method": "greedy_pilot_log_psd_range",
            "max_log_range": tolerance,
            "max_bin": max_bin,
            "pilot": {
                "quantity": "log(power + power_floor(power))",
                "smoother": "scipy.ndimage.gaussian_filter",
                "sigma": [3.0, 12.0],
                "mode": "nearest",
                "gap_handling": "smooth_each_contiguous_time_run_independently",
            },
        }
    }


def _specifications(
    power: np.ndarray,
    time_grid: np.ndarray,
    uniform_freq_bin: int,
    adaptive_max_bin: int,
) -> dict[str, dict[str, Any]]:
    pilot = _pilot_log_psd(power, time_grid)
    starts_02 = adaptive_frequency_bin_starts(
        pilot, max_log_range=0.2, max_bin=adaptive_max_bin
    )
    starts_03 = adaptive_frequency_bin_starts(
        pilot, max_log_range=0.3, max_bin=adaptive_max_bin
    )
    time2_starts = gap_aware_time_bin_starts(time_grid, 2)
    combined_metadata = _binning_metadata(0.3, adaptive_max_bin)
    combined_metadata["time"] = {
        "method": "gap_aware_uniform",
        "nominal_width": 2,
        "gap_threshold": "1.5 * median retained-grid cadence",
        "rule": "restart partition at every missing interval",
    }
    return {
        "exact": {},
        "uniform_frequency_12": {"freq_bin": uniform_freq_bin},
        "adaptive_frequency_02": {
            "freq_bin_starts": starts_02,
            "binning_metadata": _binning_metadata(0.2, adaptive_max_bin),
        },
        "adaptive_frequency_03": {
            "freq_bin_starts": starts_03,
            "binning_metadata": _binning_metadata(0.3, adaptive_max_bin),
        },
        "adaptive_frequency_03_time2": {
            "freq_bin_starts": starts_03,
            "time_bin_starts": time2_starts,
            "binning_metadata": combined_metadata,
        },
    }


def _save_fit(path: Path, result: dict[str, Any], seed: int) -> None:
    samples = result["samples"]
    try:
        diagnostics = summarize_mcmc_diagnostics(result)
    except AssertionError:
        # NumPyro's split-Rhat requires at least four draws; permit deliberately
        # tiny smoke runs while production runs retain the full diagnostics.
        diagnostics = {
            "num_chains": int(result["mcmc"].num_chains),
            "divergences": int(result["divergences"]),
            "note": "too few draws for split-Rhat (smoke run)",
        }
    extra = result["mcmc"].get_extra_fields()
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        log_psd_mean=np.asarray(result["log_psd_mean"]),
        log_psd_lower=np.asarray(result["log_psd_lower"]),
        log_psd_upper=np.asarray(result["log_psd_upper"]),
        phi_time=np.asarray(samples["phi_time"]),
        phi_freq=np.asarray(samples["phi_freq"]),
        likelihood_grid_shape=np.asarray(result["likelihood_grid_shape"], dtype=int),
        nuts_runtime_s=np.asarray(result["nuts_runtime_s"]),
        divergences=np.asarray(result["divergences"]),
        mean_num_steps=np.asarray(np.mean(np.asarray(extra["num_steps"]))),
        random_seed=np.asarray(seed),
        provenance_json=np.asarray(json.dumps(result["provenance"])),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    temporary.replace(path)


def _load_fit(path: Path) -> dict[str, Any]:
    with np.load(path) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def _posterior_sd(saved: dict[str, Any]) -> np.ndarray:
    return (saved["log_psd_upper"] - saved["log_psd_lower"]) / (
        2.0 * 1.6448536269514722
    )


def _metrics(
    saved: dict[str, Any],
    exact: dict[str, Any],
    coeffs: np.ndarray,
    null_mask: np.ndarray,
) -> dict[str, Any]:
    estimate = saved["log_psd_mean"]
    reference = exact["log_psd_mean"]
    delta = estimate - reference
    pooled_sd = np.sqrt(0.5 * (_posterior_sd(saved) ** 2 + _posterior_sd(exact) ** 2))
    normalized = delta / np.maximum(pooled_sd, 1e-12)
    z2 = coeffs**2 * np.exp(-estimate)
    exact_runtime = float(exact["nuts_runtime_s"])
    runtime = float(saved["nuts_runtime_s"])
    return {
        "likelihood_grid_shape": saved["likelihood_grid_shape"].astype(int).tolist(),
        "cell_reduction": float(coeffs.size / np.prod(saved["likelihood_grid_shape"])),
        "nuts_runtime_s": runtime,
        "speedup_vs_exact": exact_runtime / runtime,
        "divergences": int(saved["divergences"]),
        "mean_num_steps": float(saved["mean_num_steps"]),
        "log_surface_rms": float(np.sqrt(np.mean(delta**2))),
        "normalized_log_surface_rms": float(np.sqrt(np.mean(normalized**2))),
        "fraction_beyond_one_pooled_sd": float(np.mean(np.abs(normalized) > 1.0)),
        "null_log_surface_rms": float(np.sqrt(np.mean(delta[:, null_mask] ** 2))),
        "mean_z2": float(np.mean(z2)),
        "null_mean_z2": float(np.mean(z2[:, null_mask])),
        "time_z2_rmse": float(np.sqrt(np.mean((np.mean(z2, axis=1) - 1.0) ** 2))),
        "freq_z2_rmse": float(np.sqrt(np.mean((np.mean(z2, axis=0) - 1.0) ** 2))),
        "random_seed": int(saved["random_seed"]),
    }


def _write_summary(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "condition", "setting", "cell_reduction", "nuts_runtime_s",
        "speedup_vs_exact", "normalized_log_surface_rms",
        "fraction_beyond_one_pooled_sd", "null_log_surface_rms", "mean_z2",
        "null_mean_z2", "divergences", "random_seed",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition, condition_result in report["conditions"].items():
            for setting, metrics in condition_result.get("fits", {}).items():
                writer.writerow({
                    "condition": condition,
                    "setting": setting,
                    **{key: metrics[key] for key in fields[2:]},
                })


def _plot_pooling_cells(
    coeffs: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    keep: np.ndarray,
    output_dir: Path,
    paper_figure: Path,
    adaptive_max_bin: int,
) -> None:
    """Render an actual WDM crop before and after gap-aware pooling."""
    retained = np.flatnonzero(keep)
    retained_coeffs = coeffs[keep]
    retained_time = time_grid[keep]
    power = retained_coeffs**2
    pilot = _pilot_log_psd(power, retained_time)
    freq_starts = adaptive_frequency_bin_starts(
        pilot, max_log_range=0.3, max_bin=adaptive_max_bin
    )
    time_starts = gap_aware_time_bin_starts(retained_time, 2)
    pooled, _, _, counts = bin_power_rectangular(
        power,
        retained_time,
        freq_grid,
        1,
        time_bin_starts=time_starts,
        freq_bin_starts=freq_starts,
    )

    missing = np.flatnonzero(~keep)
    center_row = int(missing[len(missing) // 2]) if missing.size else time_grid.size // 2
    row_lo = max(0, center_row - 7)
    row_hi = min(time_grid.size, center_row + 8)

    freq_ends = np.r_[freq_starts[1:], freq_grid.size]
    freq_centers = np.array([
        np.mean(freq_grid[start:stop]) for start, stop in zip(freq_starts, freq_ends)
    ])
    center_bin = int(np.argmin(np.abs(freq_centers - 0.06)))
    bin_lo = max(0, center_bin - 10)
    bin_hi = min(freq_starts.size, center_bin + 11)
    freq_lo = int(freq_starts[bin_lo])
    freq_hi = int(freq_ends[bin_hi - 1])

    fine = np.full((row_hi - row_lo, freq_hi - freq_lo), np.nan)
    for retained_row, original_row in enumerate(retained):
        if row_lo <= original_row < row_hi:
            fine[original_row - row_lo] = np.log(
                power[retained_row, freq_lo:freq_hi] + power_floor(power)
            )
    coarse = np.full_like(fine, np.nan)
    rectangles: list[tuple[float, float, float, float]] = []
    time_ends = np.r_[time_starts[1:], retained_time.size]
    for ti, (t_start, t_stop) in enumerate(zip(time_starts, time_ends)):
        original_rows = retained[t_start:t_stop]
        if original_rows.size == 0 or original_rows[-1] < row_lo or original_rows[0] >= row_hi:
            continue
        for fi in range(bin_lo, bin_hi):
            f_start, f_stop = int(freq_starts[fi]), int(freq_ends[fi])
            value = np.log(pooled[ti, fi] / counts[ti, fi])
            rows = original_rows[(original_rows >= row_lo) & (original_rows < row_hi)]
            coarse[rows - row_lo, f_start - freq_lo : f_stop - freq_lo] = value
            rectangles.append((
                f_start - freq_lo,
                original_rows[0] - row_lo,
                f_stop - f_start,
                original_rows[-1] - original_rows[0] + 1,
            ))

    valid = fine[np.isfinite(fine)]
    vmin, vmax = np.percentile(valid, [5, 95])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.82")
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), sharex=True, sharey=True)
    images = []
    for axis, image, title in zip(
        axes,
        (fine, coarse),
        (r"Fine WDM cells $w_{nm}^2$", r"Pooled sufficient statistics $P_B/\nu_B$"),
    ):
        images.append(axis.imshow(
            image, origin="lower", aspect="auto", interpolation="nearest",
            cmap=cmap, vmin=vmin, vmax=vmax,
        ))
        axis.set_title(title, fontsize=10)
        ticks = np.linspace(0.07, 0.93, 3) * (freq_hi - freq_lo - 1)
        channels = np.rint(freq_lo + np.linspace(0.07, 0.93, 3) * (freq_hi - freq_lo - 1)).astype(int)
        axis.set_xticks(ticks, [f"{1e3 * freq_grid[channel]:.2f}" for channel in channels])
        axis.set_xlabel("Frequency [mHz]", fontsize=9)
        axis.tick_params(labelsize=8)
    axes[0].set_ylabel("Time [days]", fontsize=9)
    y_ticks = np.array([0, (row_hi - row_lo - 1) / 2, row_hi - row_lo - 1])
    row_ticks = np.rint(y_ticks + row_lo).astype(int)
    axes[0].set_yticks(y_ticks, [f"{OBSERVATION_DAYS * time_grid[row]:.1f}" for row in row_ticks])
    for x, y, width, height in rectangles:
        axes[1].add_patch(Rectangle((x - 0.5, y - 0.5), width, height,
                                    fill=False, edgecolor="white", linewidth=0.65))
    axes[1].legend(
        handles=[Patch(facecolor="0.82", edgecolor="0.5", label="gap rows")],
        loc="upper right", frameon=True, fontsize=7,
    )
    colorbar = fig.colorbar(images[-1], ax=axes, pad=0.02, fraction=0.035)
    colorbar.set_label("log WDM power", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Gap-aware time pooling restarts at each missing interval",
        y=0.98,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.19, top=0.83, wspace=0.08)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "pooling_cells.png"
    pdf = output_dir / "pooling_cells.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    paper_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_figure, dpi=240)
    plt.close(fig)
    print(f"[figure] wrote {png}, {pdf}, and {paper_figure}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paper-figure", type=Path, default=PAPER_FIGURE)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--gap-seed", type=int, default=1)
    parser.add_argument("--uniform-freq-bin", type=int, default=12)
    parser.add_argument("--adaptive-max-bin", type=int, default=64)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with np.load(args.cache) as cached:
        coeffs_all = np.asarray(cached["coeffs"], dtype=float)
        time_grid_all = np.asarray(cached["time_grid"], dtype=float)
        freq_grid_all = np.asarray(cached["freq_grid"], dtype=float)
    band = (freq_grid_all >= 1e-4) & (freq_grid_all <= 0.1)
    coeffs_all = coeffs_all[:, band]
    freq_grid = freq_grid_all[band]
    keep_gapped, gap_schedule = _gap_mask(time_grid_all, args.gap_seed)
    if not np.any(~keep_gapped):
        raise RuntimeError("The gap schedule removed no WDM rows; check the grid cadence.")

    _plot_pooling_cells(
        coeffs_all, time_grid_all, freq_grid, keep_gapped, args.output_dir,
        args.paper_figure, args.adaptive_max_bin,
    )
    if args.plot_only:
        return

    config = PSplineConfig(
        n_interior_knots_time=16,
        n_interior_knots_freq=94,
        freq_knot_strategy="linear",
        centered=True,
    )
    report: dict[str, Any] = {
        "source_cache": str(args.cache),
        "observation_days": OBSERVATION_DAYS,
        "configuration": {
            "warmup": args.warmup,
            "samples": args.samples,
            "chains": args.chains,
            "base_seed": args.seed,
            "gap_seed": args.gap_seed,
            "uniform_freq_bin": args.uniform_freq_bin,
            "adaptive_max_bin": args.adaptive_max_bin,
            "frequency_band_hz": [float(freq_grid[0]), float(freq_grid[-1])],
        },
        "gap_schedule_seconds": gap_schedule,
        "conditions": {},
    }
    conditions = {"intact": np.ones(time_grid_all.size, dtype=bool), "gapped": keep_gapped}
    null_mask = np.min(np.abs(freq_grid[:, None] - NULLS[None, :]), axis=1) < 0.002

    for condition_index, (condition, keep) in enumerate(conditions.items()):
        condition_dir = args.output_dir / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        coeffs = coeffs_all[keep]
        time_grid = time_grid_all[keep]
        specs = _specifications(
            coeffs**2, time_grid, args.uniform_freq_bin, args.adaptive_max_bin
        )
        report["conditions"][condition] = {
            "fine_grid_shape": list(coeffs.shape),
            "removed_time_rows": int(np.count_nonzero(~keep)),
            "fits": {},
        }
        saved_fits: dict[str, dict[str, Any]] = {}
        for setting_index, (setting, options) in enumerate(specs.items()):
            seed = args.seed + 100 * condition_index + setting_index
            path = condition_dir / f"{setting}.npz"
            if path.exists() and not args.force:
                print(f"[{condition}/{setting}] resume from {path}", flush=True)
            else:
                print(f"[{condition}/{setting}] seed={seed} options={list(options)}", flush=True)
                result = fit_log_pspline_surface(
                    coeffs=coeffs[None],
                    time_grid=time_grid,
                    freq_grid=freq_grid,
                    config=config,
                    n_warmup=args.warmup,
                    n_samples=args.samples,
                    num_chains=args.chains,
                    random_seed=seed,
                    progress_bar=True,
                    **options,
                )
                _save_fit(path, result, seed)
                print(
                    f"[{condition}/{setting}] grid={result['likelihood_grid_shape']} "
                    f"runtime={result['nuts_runtime_s']:.1f}s div={result['divergences']}",
                    flush=True,
                )
            saved_fits[setting] = _load_fit(path)

            exact = saved_fits.get("exact")
            if exact is not None:
                for available, saved in saved_fits.items():
                    report["conditions"][condition]["fits"][available] = _metrics(
                        saved, exact, coeffs, null_mask
                    )
                _write_summary(args.output_dir, report)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
