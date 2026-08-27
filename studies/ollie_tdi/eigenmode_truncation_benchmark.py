"""Benchmark roughness-ranked eigenmode truncation on the Ollie WDM grid.

This is an approximation study, not a production model option.  It retains the
smoothest tensor roughness modes according to the prior-mean precision
``E[phi_t] * lambda_t + E[phi_f] * lambda_f`` and compares every truncated fit
with the full centered model on the original likelihood grid.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_value

from tv_pspline_psd import PSplineConfig
from tv_pspline_psd.inference import _prepare_spline_bases, surface_summaries
from tv_pspline_psd.model import (
    _sample_log_gamma,
    eigen_prior_scale,
    initialize_with_penalized_least_squares,
    power_whittle_log_likelihood,
    tensor_product_surface,
    whiten_penalty_pair,
    whitened_init_values,
)
from tv_pspline_psd.splines import create_bspline_roughness_penalty

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "studies/results/ollie_tdi/knot_map_benchmark/wdm_coeffs.npz"
DEFAULT_OUTPUT = REPO / "studies/results/ollie_tdi/performance/eigenmode_truncation.json"
NULLS_HZ = np.arange(0.03, 0.121, 0.03)


def _truncated_model(
    summed_power,
    counts,
    basis_time,
    basis_freq,
    lam_time,
    lam_freq,
    joint_null,
    retained_indices,
    config,
):
    phi_time = _sample_log_gamma(
        "phi_time", config.alpha_phi, config.beta_phi, config.phi_log_base_scale
    )
    phi_freq = _sample_log_gamma(
        "phi_freq", config.alpha_phi, config.beta_phi, config.phi_log_base_scale
    )
    scale = eigen_prior_scale(
        phi_time, phi_freq, lam_time, lam_freq, joint_null, config
    ).reshape(-1)[retained_indices]
    with numpyro.plate("s_plate", retained_indices.size):
        retained = numpyro.sample("s", dist.Normal(0.0, scale))
    full = jnp.zeros(lam_time.size * lam_freq.size).at[retained_indices].set(retained)
    log_psd = tensor_product_surface(
        basis_time, full.reshape((lam_time.size, lam_freq.size)), basis_freq
    )
    numpyro.factor(
        "whittle", power_whittle_log_likelihood(summed_power, counts, log_psd)
    )


def _positive_summary(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.median(values)), float(np.min(values))


def _run_fit(problem, retained_indices, *, warmup, samples, chains, seed):
    init_values = {
        "s": problem["init_values"]["s"][retained_indices],
        "phi_time": problem["init_values"]["phi_time"],
        "phi_freq": problem["init_values"]["phi_freq"],
    }
    args = (
        problem["power"],
        1,
        problem["basis_time"],
        problem["basis_freq"],
        problem["lam_time"],
        problem["lam_freq"],
        problem["joint_null"],
        jnp.asarray(retained_indices),
        problem["config"],
    )

    def make_mcmc():
        kernel = NUTS(
            _truncated_model,
            init_strategy=init_to_value(values=init_values),
            target_accept_prob=0.85,
            max_tree_depth=10,
        )
        return MCMC(
            kernel,
            num_warmup=warmup,
            num_samples=samples,
            num_chains=chains,
            chain_method="sequential",
            progress_bar=False,
        )

    # First identical warmup pays compilation.  A fresh second warmup uses the
    # cached executable and provides the compilation-excluded timing.
    compile_mcmc = make_mcmc()
    t0 = time.perf_counter()
    compile_mcmc.warmup(random.PRNGKey(seed), *args)
    first_warmup_s = time.perf_counter() - t0

    mcmc = make_mcmc()
    t0 = time.perf_counter()
    mcmc.warmup(random.PRNGKey(seed + 1), *args)
    warmup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    mcmc.run(
        random.PRNGKey(seed + 2),
        *args,
        extra_fields=("diverging", "num_steps", "accept_prob"),
    )
    sample_s = time.perf_counter() - t0

    grouped = mcmc.get_samples(group_by_chain=True)
    diag = summary(grouped, group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    steps = np.asarray(extra["num_steps"])
    s_ess_median, s_ess_min = _positive_summary(diag["s"]["n_eff"])
    n_total = problem["n_time_basis"] * problem["n_freq_basis"]
    eig_samples = np.zeros((chains * samples, n_total))
    eig_samples[:, retained_indices] = np.asarray(grouped["s"]).reshape(
        chains * samples, -1
    )
    eig_samples = eig_samples.reshape(
        chains * samples, problem["n_time_basis"], problem["n_freq_basis"]
    )
    mean, lower, upper = surface_summaries(
        eig_samples, np.asarray(problem["basis_time"]), np.asarray(problem["basis_freq"])
    )
    runtime_s = warmup_s + sample_s
    metrics = {
        "retained_modes": int(retained_indices.size),
        "sampled_parameters": int(retained_indices.size + 2),
        "first_warmup_including_compile_s": first_warmup_s,
        "cached_warmup_s": warmup_s,
        "cached_sampling_s": sample_s,
        "cached_nuts_runtime_s": runtime_s,
        "estimated_compile_overhead_s": max(0.0, first_warmup_s - warmup_s),
        "mean_num_steps": float(np.mean(steps)),
        "median_num_steps": float(np.median(steps)),
        "max_tree_depth_hit_fraction": float(np.mean(steps >= 1023)),
        "divergences": int(np.sum(np.asarray(extra["diverging"]))),
        "mean_accept_prob": float(np.mean(np.asarray(extra["accept_prob"]))),
        "s_ess_median": s_ess_median,
        "s_ess_min": s_ess_min,
        "s_ess_per_s": s_ess_median / runtime_s,
        "s_rhat_max": float(np.nanmax(diag["s"]["r_hat"])),
        "phi_time_ess": float(diag["phi_time"]["n_eff"]),
        "phi_time_ess_per_s": float(diag["phi_time"]["n_eff"]) / runtime_s,
        "phi_time_rhat": float(diag["phi_time"]["r_hat"]),
        "phi_freq_ess": float(diag["phi_freq"]["n_eff"]),
        "phi_freq_ess_per_s": float(diag["phi_freq"]["n_eff"]) / runtime_s,
        "phi_freq_rhat": float(diag["phi_freq"]["r_hat"]),
    }
    return metrics, {"mean": mean, "lower": lower, "upper": upper}


def _compare(surface, reference, null_mask):
    delta = surface["mean"] - reference["mean"]
    baseline_sd = (reference["upper"] - reference["lower"]) / (2 * 1.6448536269514722)
    normalized = delta / np.maximum(baseline_sd, 1e-12)
    width = surface["upper"] - surface["lower"]
    reference_width = reference["upper"] - reference["lower"]
    relative_psd = np.abs(np.expm1(delta))
    time_gradient_delta = np.diff(surface["mean"], axis=0) - np.diff(
        reference["mean"], axis=0
    )
    return {
        "mean_absolute_log_psd_difference": float(np.mean(np.abs(delta))),
        "log_psd_rms_difference": float(np.sqrt(np.mean(delta**2))),
        "max_relative_psd_difference": float(np.max(relative_psd)),
        "p99_relative_psd_difference": float(np.percentile(relative_psd, 99)),
        "normalized_mean_difference_rms": float(np.sqrt(np.mean(normalized**2))),
        "fraction_mean_beyond_one_baseline_sd": float(np.mean(np.abs(normalized) > 1)),
        "mean_absolute_interval_width_change": float(
            np.mean(np.abs(width - reference_width))
        ),
        "mean_interval_width_ratio": float(np.mean(width) / np.mean(reference_width)),
        "null_corridor_log_rms_difference": float(
            np.sqrt(np.mean(delta[:, null_mask] ** 2))
        ),
        "outside_null_log_rms_difference": float(
            np.sqrt(np.mean(delta[:, ~null_mask] ** 2))
        ),
        "time_gradient_log_rms_difference": float(
            np.sqrt(np.mean(time_gradient_delta**2))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()

    cached = np.load(args.cache)
    coeffs = np.asarray(cached["coeffs"], dtype=float)
    time_grid = np.asarray(cached["time_grid"], dtype=float)
    freq_grid = np.asarray(cached["freq_grid"], dtype=float)
    power = coeffs**2
    config = PSplineConfig(
        n_interior_knots_time=16,
        n_interior_knots_freq=94,
        freq_knot_strategy="linear",
        centered=True,
    )
    spline = _prepare_spline_bases(power, time_grid, freq_grid, config, n_components=1)
    penalty_time = create_bspline_roughness_penalty(
        spline["knots_time"], degree=config.degree_time,
        derivative_order=config.diff_order_time,
    )
    penalty_freq = create_bspline_roughness_penalty(
        spline["knots_freq_unit"], degree=config.degree_freq,
        derivative_order=config.diff_order_freq,
    )
    whitened = whiten_penalty_pair(penalty_time, penalty_freq)
    basis_time = spline["B_time"] @ whitened["U_time"]
    basis_freq = spline["B_freq"] @ whitened["U_freq"]
    pls = initialize_with_penalized_least_squares(
        power, spline["B_time"], spline["B_freq"], penalty_time, penalty_freq, config
    )
    init_values = whitened_init_values(pls, whitened, config)
    n_time_basis, n_freq_basis = basis_time.shape[1], basis_freq.shape[1]
    score = (
        (config.alpha_phi / config.beta_phi) * whitened["lam_time"][:, None]
        + (config.alpha_phi / config.beta_phi) * whitened["lam_freq"][None, :]
    ).reshape(-1)
    ranking = np.argsort(score, kind="stable")
    problem = {
        "power": jnp.asarray(power),
        "basis_time": jnp.asarray(basis_time),
        "basis_freq": jnp.asarray(basis_freq),
        "lam_time": jnp.asarray(whitened["lam_time"]),
        "lam_freq": jnp.asarray(whitened["lam_freq"]),
        "joint_null": jnp.asarray(whitened["joint_null"]),
        "config": config,
        "init_values": init_values,
        "n_time_basis": n_time_basis,
        "n_freq_basis": n_freq_basis,
    }
    null_mask = np.min(
        np.abs(freq_grid[:, None] - NULLS_HZ[None, :]), axis=1
    ) < 0.002
    report = {
        "source_cache": str(args.cache),
        "likelihood_grid_shape": list(power.shape),
        "basis_shape": [n_time_basis, n_freq_basis],
        "backend": "jax-cpu-float64",
        "ranking": "ascending prior-mean roughness precision",
        "protocol": {
            "warmup_per_chain": args.warmup,
            "draws_per_chain": args.samples,
            "chains": args.chains,
            "target_accept_prob": 0.85,
            "max_tree_depth": 10,
        },
        "fits": {},
    }
    surfaces = {}
    n_total = n_time_basis * n_freq_basis
    for fraction in (1.0, 0.75, 0.5, 0.25):
        n_keep = int(np.ceil(fraction * n_total))
        retained = np.sort(ranking[:n_keep])
        label = f"{int(100 * fraction)}pct"
        print(f"[{label}] retaining {n_keep}/{n_total} modes", flush=True)
        metrics, surfaces[label] = _run_fit(
            problem,
            retained,
            warmup=args.warmup,
            samples=args.samples,
            chains=args.chains,
            seed=20260825 + n_keep,
        )
        report["fits"][label] = metrics
        print(json.dumps({label: metrics}, sort_keys=True), flush=True)

    reference = surfaces["100pct"]
    full_runtime = report["fits"]["100pct"]["cached_nuts_runtime_s"]
    for label, surface in surfaces.items():
        report["fits"][label]["speedup_vs_full"] = (
            full_runtime / report["fits"][label]["cached_nuts_runtime_s"]
        )
        report["fits"][label]["surface_comparison_vs_full"] = _compare(
            surface, reference, null_mask
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
