"""Paired LS2 validation of time- and frequency-likelihood coarse-graining.

Every setting fits the same simulated data realization, so differences from the
exact likelihood are paired rather than obscured by realization-to-realization
variation. Results are written after each realization and can be resumed safely.

Production run (about 1--2 hours on a modern local CPU):

    .venv/bin/python studies/ls2_coarse_graining_study.py --repeats 100

Quick smoke run:

    .venv/bin/python studies/ls2_coarse_graining_study.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tv_pspline_psd import (
    PSplineConfig,
    interval_coverage,
    mse_log_psd,
    run_wdm_psd_mcmc,
    summarize_mcmc_diagnostics,
)
from tv_pspline_psd.datasets import (
    simulate_ls2,
    true_psd_ls2,
    wdm_white_noise_calibration,
)

N_TOTAL = 16_384
NT = 128
DT = 0.1
N_FREQ_KNOTS = 10
SETTINGS = (
    ("exact", 1, 1),
    ("time_x4", 4, 1),
    ("frequency_x4", 1, 4),
    ("balanced_x2_x2", 2, 2),
    ("combined_x4_x4", 4, 4),
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "ls2" / "coarse_graining_16384"
FIELDS = (
    "repeat", "data_seed", "sampler_seed", "setting", "time_bin", "freq_bin", "mse_log", "coverage",
    "ci_width_log", "nuts_runtime_s", "speedup_vs_exact",
    "normalized_surface_rms", "fraction_beyond_one_pooled_sd", "divergences",
    "max_rhat", "min_neff", "likelihood_time_cells", "likelihood_freq_cells",
)


@dataclass(frozen=True)
class Setting:
    name: str
    time_bin: int
    freq_bin: int


def _config() -> PSplineConfig:
    # Match the paper's growing-knot rule at nt=128 and retain its WDM
    # frequency-basis size. The x4 grid still has more likelihood locations
    # than basis functions along each axis after trimming.
    n_time_knots = round(8 * (NT / 24) ** 0.4)
    return PSplineConfig(
        n_interior_knots_time=n_time_knots,
        n_interior_knots_freq=N_FREQ_KNOTS,
    )


def _posterior_shift(result: dict[str, object], exact: dict[str, object]) -> tuple[float, float]:
    if result is exact:
        return 0.0, 0.0
    z90 = 1.6448536269514722
    delta = np.asarray(result["log_psd_mean"]) - np.asarray(exact["log_psd_mean"])
    sd_result = (np.asarray(result["log_psd_upper"]) - np.asarray(result["log_psd_lower"])) / (2 * z90)
    sd_exact = (np.asarray(exact["log_psd_upper"]) - np.asarray(exact["log_psd_lower"])) / (2 * z90)
    normalized = delta / np.maximum(np.sqrt(0.5 * (sd_result**2 + sd_exact**2)), 1e-12)
    return float(np.sqrt(np.mean(normalized**2))), float(np.mean(np.abs(normalized) > 1.0))


def _diagnostics(result: dict[str, object]) -> tuple[int, float, float]:
    diag = summarize_mcmc_diagnostics(result)
    rhat = np.r_[
        np.atleast_1d(diag["phi_time"]["r_hat"]),
        np.atleast_1d(diag["phi_freq"]["r_hat"]),
    ]
    neff = np.r_[
        np.atleast_1d(diag["phi_time"]["n_eff"]),
        np.atleast_1d(diag["phi_freq"]["n_eff"]),
    ]
    return int(diag["divergences"]), float(np.nanmax(rhat)), float(np.nanmin(neff))


def _completed_repeats(path: Path) -> set[int]:
    """Retain only complete paired repeats, so an interrupted run is restartable."""
    if not path.exists():
        return set()
    rows: list[dict[str, str]] = []
    seen: dict[int, set[str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            seen.setdefault(int(row["repeat"]), set()).add(row["setting"])
    expected = {setting[0] for setting in SETTINGS}
    complete = {repeat for repeat, settings in seen.items() if settings == expected}
    retained = [row for row in rows if int(row["repeat"]) in complete]
    if len(retained) != len(rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(retained)
    return complete


def _merge_shards(output_dir: Path) -> None:
    """Merge disjoint Slurm-array shards into one result file, rejecting overlap."""
    shard_paths = sorted((output_dir / "shards").glob("*/per_repeat.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard results found under {output_dir / 'shards'}")
    rows: list[dict[str, str]] = []
    repeats_seen: set[int] = set()
    for path in shard_paths:
        with path.open(newline="") as handle:
            shard_rows = list(csv.DictReader(handle))
        shard_repeats = {int(row["repeat"]) for row in shard_rows}
        if repeats_seen.intersection(shard_repeats):
            raise ValueError(f"Duplicate repeat indices across shards, including {path}.")
        repeats_seen.update(shard_repeats)
        rows.extend(shard_rows)
    rows.sort(key=lambda row: (int(row["repeat"]), row["setting"]))
    destination = output_dir / "per_repeat.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(shard_paths)} shards and {len(repeats_seen)} paired repeats into {destination}")


def _write_protocol(path: Path, config: PSplineConfig, args: argparse.Namespace) -> None:
    path.write_text(json.dumps({
        "n_total": N_TOTAL,
        "nt": NT,
        "nf": N_TOTAL // NT,
        "dt": DT,
        "config": asdict(config),
        "settings": [asdict(Setting(*setting)) for setting in SETTINGS],
        "repeats": args.repeats,
        "repeat_range": [args.repeat_start, args.repeat_stop],
        "warmup": args.warmup,
        "samples": args.samples,
        "chains": args.chains,
        "seed0": args.seed0,
    }, indent=2) + "\n")


def _render(output_dir: Path) -> None:
    path = output_dir / "per_repeat.csv"
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}")
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("No completed repeats to render.")

    summary: list[dict[str, object]] = []
    for name, time_bin, freq_bin in SETTINGS:
        block = [row for row in rows if row["setting"] == name]

        def metric(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in block])

        runtime = metric("nuts_runtime_s")
        summary.append({
            "setting": name,
            "time_bin": time_bin,
            "freq_bin": freq_bin,
            "repeats": len(block),
            "mse_median": float(np.median(metric("mse_log"))),
            "coverage_mean": float(np.mean(metric("coverage"))),
            "ci_width_median": float(np.median(metric("ci_width_log"))),
            "surface_rms_median": float(np.median(metric("normalized_surface_rms"))),
            "beyond_one_sd_mean": float(np.mean(metric("fraction_beyond_one_pooled_sd"))),
            "runtime_median_s": float(np.median(runtime)),
            "speedup_median": float(np.median(metric("speedup_vs_exact"))),
            "divergences_total": int(np.sum(metric("divergences"))),
            "max_rhat": float(np.nanmax(metric("max_rhat"))),
            "min_neff": float(np.nanmin(metric("min_neff"))),
        })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    header = "| setting | bins (time, freq) | MSE log | 90% coverage | CI width | shift / pooled SD | >1 SD | runtime | speedup | div |"
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    table_rows = [header, divider]
    tex_rows = []
    for row in summary:
        table_rows.append(
            f"| {row['setting']} | ({row['time_bin']}, {row['freq_bin']}) | "
            f"{row['mse_median']:.4f} | {row['coverage_mean']:.3f} | "
            f"{row['ci_width_median']:.3f} | {row['surface_rms_median']:.3f} | "
            f"{100 * row['beyond_one_sd_mean']:.2f}% | {row['runtime_median_s']:.2f} s | "
            f"{row['speedup_median']:.2f}x | {row['divergences_total']} |"
        )
        tex_rows.append(
            f"{row['setting'].replace('_', r'\_')} & ({row['time_bin']}, {row['freq_bin']}) & "
            f"{row['mse_median']:.4f} & {row['coverage_mean']:.3f} & "
            f"{row['ci_width_median']:.3f} & {row['surface_rms_median']:.3f} & "
            f"{100 * row['beyond_one_sd_mean']:.2f}\\% & {row['runtime_median_s']:.2f} & "
            f"{row['speedup_median']:.2f} & {row['divergences_total']} \\\\"
        )
    (output_dir / "summary.md").write_text("\n".join(table_rows) + "\n")
    (output_dir / "summary_rows.tex").write_text("\n".join(tex_rows) + "\n")
    print("\n".join(table_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed0", type=int, default=8_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick", action="store_true", help="Run 2 repeats with 80 warmup/draws and one chain.")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--merge-shards", action="store_true", help="Merge Slurm-array result shards, then render the summary table.")
    parser.add_argument("--repeat-start", type=int, default=0, help="Inclusive repeat index for this shard.")
    parser.add_argument("--repeat-stop", type=int, default=None, help="Exclusive repeat index for this shard (default: --repeats).")
    args = parser.parse_args()
    if args.quick:
        args.repeats, args.warmup, args.samples, args.chains = 2, 80, 80, 1
    if args.repeat_stop is None:
        args.repeat_stop = args.repeats
    if not 0 <= args.repeat_start <= args.repeat_stop <= args.repeats:
        parser.error("require 0 <= --repeat-start <= --repeat-stop <= --repeats")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_shards:
        _merge_shards(output_dir)
        _render(output_dir)
        return
    if args.render_only:
        _render(output_dir)
        return

    config = _config()
    _write_protocol(output_dir / "protocol.json", config, args)
    rows_path = output_dir / "per_repeat.csv"
    completed = _completed_repeats(rows_path)
    calibration = wdm_white_noise_calibration(N_TOTAL, DT, NT, config)
    settings = [Setting(*setting) for setting in SETTINGS]

    new_file = not rows_path.exists()
    with rows_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for repeat in range(args.repeat_start, args.repeat_stop):
            if repeat in completed:
                continue
            data_seed = args.seed0 + repeat
            data = simulate_ls2(N_TOTAL, rng=np.random.default_rng(data_seed))
            results: dict[str, dict[str, object]] = {}
            sampler_seeds: dict[str, int] = {}
            for setting_index, setting in enumerate(settings):
                sampler_seed = args.seed0 + 100_000 * (setting_index + 1) + repeat
                sampler_seeds[setting.name] = sampler_seed
                results[setting.name] = run_wdm_psd_mcmc(
                    data,
                    dt=DT,
                    nt=NT,
                    config=config,
                    n_warmup=args.warmup,
                    n_samples=args.samples,
                    num_chains=args.chains,
                    random_seed=sampler_seed,
                    progress_bar=False,
                    time_bin=setting.time_bin,
                    freq_bin=setting.freq_bin,
                    binning_metadata={
                        "study": "ls2_paired_time_frequency_coarse_graining",
                        "setting": setting.name,
                    },
                )
            exact = results["exact"]
            true_psd = calibration[None, :] * true_psd_ls2(
                np.asarray(exact["time_grid"]), np.asarray(exact["freq_grid"]), DT
            )
            for setting in settings:
                result = results[setting.name]
                divergences, max_rhat, min_neff = _diagnostics(result)
                shift, beyond_one_sd = _posterior_shift(result, exact)
                shape = result["likelihood_grid_shape"]
                writer.writerow({
                    "repeat": repeat,
                    "data_seed": data_seed,
                    "sampler_seed": sampler_seeds[setting.name],
                    "setting": setting.name,
                    "time_bin": setting.time_bin,
                    "freq_bin": setting.freq_bin,
                    "mse_log": mse_log_psd(true_psd, np.asarray(result["psd_mean"])),
                    "coverage": interval_coverage(true_psd, np.asarray(result["psd_lower"]), np.asarray(result["psd_upper"])),
                    "ci_width_log": float(np.mean(np.asarray(result["log_psd_upper"]) - np.asarray(result["log_psd_lower"]))),
                    "nuts_runtime_s": result["nuts_runtime_s"],
                    "speedup_vs_exact": float(exact["nuts_runtime_s"]) / float(result["nuts_runtime_s"]),
                    "normalized_surface_rms": shift,
                    "fraction_beyond_one_pooled_sd": beyond_one_sd,
                    "divergences": divergences,
                    "max_rhat": max_rhat,
                    "min_neff": min_neff,
                    "likelihood_time_cells": shape[0],
                    "likelihood_freq_cells": shape[1],
                })
            handle.flush()
            print(
                f"completed repeat {repeat} in shard [{args.repeat_start}, {args.repeat_stop})",
                flush=True,
            )
    _render(output_dir)


if __name__ == "__main__":
    main()
