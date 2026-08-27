"""Prototype native-coefficient sparse B-spline likelihood evaluation.

The native and whitened coordinates describe the same full tensor spline.  The
native basis exposes cubic B-spline local support, but its roughness prior is
correlated.  This script measures the likelihood kernel and runs a deliberately
short NUTS geometry pilot; it does not add a production inference option.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_value

from tv_pspline_psd import PSplineConfig
from tv_pspline_psd.inference import _prepare_spline_bases
from tv_pspline_psd.model import (
    _sample_log_gamma,
    initialize_with_penalized_least_squares,
    power_whittle_log_likelihood,
    tensor_product_surface,
    whiten_penalty_pair,
)
from tv_pspline_psd.splines import create_bspline_roughness_penalty

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "studies/results/ollie_tdi/knot_map_benchmark/wdm_coeffs.npz"
DEFAULT_OUTPUT = REPO / "studies/results/ollie_tdi/performance/sparse_basis_prototype.json"


def _active_support(basis: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.zeros((basis.shape[0], width), dtype=np.int32)
    values = np.zeros((basis.shape[0], width), dtype=basis.dtype)
    for row_index, row in enumerate(basis):
        active = np.flatnonzero(np.abs(row) > 1e-14)
        if active.size > width:
            raise ValueError("basis row has more active entries than expected")
        indices[row_index, : active.size] = active
        values[row_index, : active.size] = row[active]
    return indices, values


def sparse_tensor_surface(
    coefficients, time_indices, time_values, freq_indices, freq_values
):
    """Evaluate a cubic tensor spline using the four active entries per axis."""
    frequency_projection = jnp.sum(
        coefficients[:, freq_indices] * freq_values[None, :, :], axis=-1
    )
    return jnp.sum(
        frequency_projection[time_indices, :] * time_values[:, :, None], axis=1
    )


def _native_model(
    power,
    counts,
    time_indices,
    time_values,
    freq_indices,
    freq_values,
    penalty_time,
    penalty_freq,
    eigenvectors_time,
    eigenvectors_freq,
    lam_time,
    lam_freq,
    joint_null,
    config,
):
    phi_time = _sample_log_gamma(
        "phi_time", config.alpha_phi, config.beta_phi, config.phi_log_base_scale
    )
    phi_freq = _sample_log_gamma(
        "phi_freq", config.alpha_phi, config.beta_phi, config.phi_log_base_scale
    )
    n_weights = penalty_time.shape[0] * penalty_freq.shape[0]
    native = numpyro.sample(
        "w", dist.Normal(0.0, 1.0).expand([n_weights]).to_event(1)
    ).reshape((penalty_time.shape[0], penalty_freq.shape[0]))

    # Replace the independent standard-Normal reference density with the exact
    # normalized structured Gaussian roughness prior.  Evaluating its quadratic
    # in the eigenbasis is cheap, but the sampled native coordinates remain
    # strongly correlated -- precisely the geometry tradeoff under study.
    eigen = eigenvectors_time.T @ native @ eigenvectors_freq
    precision = phi_time * lam_time[:, None] + phi_freq * lam_freq[None, :]
    precision = jnp.where(joint_null, config.null_precision, precision + config.ridge_eps)
    numpyro.factor(
        "structured_prior",
        -0.5 * jnp.sum(precision * eigen**2)
        + 0.5 * jnp.sum(native**2)
        + 0.5 * jnp.sum(jnp.log(precision)),
    )
    log_psd = sparse_tensor_surface(
        native, time_indices, time_values, freq_indices, freq_values
    )
    numpyro.factor("whittle", power_whittle_log_likelihood(power, counts, log_psd))


def _bench(fn, argument, repetitions: int = 100) -> dict[str, float]:
    compiled = jax.jit(fn)
    t0 = time.perf_counter()
    jax.block_until_ready(compiled(argument))
    compile_and_first_s = time.perf_counter() - t0
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter_ns()
        jax.block_until_ready(compiled(argument))
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "compile_and_first_s": compile_and_first_s,
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "p90_ms": float(np.percentile(times, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()

    cached = np.load(args.cache)
    coeffs = np.asarray(cached["coeffs"], dtype=float)
    power_np = coeffs**2
    config = PSplineConfig(
        n_interior_knots_time=16,
        n_interior_knots_freq=94,
        freq_knot_strategy="linear",
        centered=True,
    )
    spline = _prepare_spline_bases(
        power_np, cached["time_grid"], cached["freq_grid"], config, n_components=1
    )
    penalty_time = create_bspline_roughness_penalty(
        spline["knots_time"], degree=config.degree_time,
        derivative_order=config.diff_order_time,
    )
    penalty_freq = create_bspline_roughness_penalty(
        spline["knots_freq_unit"], degree=config.degree_freq,
        derivative_order=config.diff_order_freq,
    )
    whitened = whiten_penalty_pair(penalty_time, penalty_freq)
    basis_eig_time = jnp.asarray(spline["B_time"] @ whitened["U_time"])
    basis_eig_freq = jnp.asarray(spline["B_freq"] @ whitened["U_freq"])
    time_indices, time_values = _active_support(spline["B_time"], config.degree_time + 1)
    freq_indices, freq_values = _active_support(spline["B_freq"], config.degree_freq + 1)
    ti, tv = jnp.asarray(time_indices), jnp.asarray(time_values)
    fi, fv = jnp.asarray(freq_indices), jnp.asarray(freq_values)
    power = jnp.asarray(power_np)
    rng = np.random.default_rng(20260825)
    native = jnp.asarray(rng.normal(scale=0.1, size=(20, 98)))
    eigen = jnp.asarray(whitened["U_time"].T) @ native @ jnp.asarray(whitened["U_freq"])

    def dense_surface(value):
        return tensor_product_surface(basis_eig_time, value, basis_eig_freq)

    def sparse_surface(value):
        return sparse_tensor_surface(value, ti, tv, fi, fv)

    def dense_ll(value):
        return power_whittle_log_likelihood(power, 1, dense_surface(value))

    def sparse_ll(value):
        return power_whittle_log_likelihood(power, 1, sparse_surface(value))
    report = {
        "likelihood_grid_shape": list(power_np.shape),
        "basis_shape": [20, 98],
        "active_entries_per_axis": [4, 4],
        "basis_storage_bytes": {
            "dense_whitened": int(basis_eig_time.nbytes + basis_eig_freq.nbytes),
            "native_active_indices_values": int(
                time_indices.nbytes + time_values.nbytes
                + freq_indices.nbytes + freq_values.nbytes
            ),
        },
        "microbenchmarks": {
            "dense_whitened_forward": _bench(dense_surface, eigen),
            "native_sparse_forward": _bench(sparse_surface, native),
            "dense_whitened_value_and_grad": _bench(jax.value_and_grad(dense_ll), eigen, 50),
            "native_sparse_value_and_grad": _bench(jax.value_and_grad(sparse_ll), native, 50),
        },
    }
    np.testing.assert_allclose(
        np.asarray(dense_surface(eigen)), np.asarray(sparse_surface(native)),
        rtol=2e-12, atol=2e-12,
    )

    pls = initialize_with_penalized_least_squares(
        power_np, spline["B_time"], spline["B_freq"], penalty_time, penalty_freq, config
    )
    init_values = {
        "w": np.asarray(pls["W"]).reshape(-1),
        "phi_time": float(np.log(pls["phi_time"])),
        "phi_freq": float(np.log(pls["phi_freq"])),
    }
    model_args = (
        power, 1, ti, tv, fi, fv,
        jnp.asarray(penalty_time), jnp.asarray(penalty_freq),
        jnp.asarray(whitened["U_time"]), jnp.asarray(whitened["U_freq"]),
        jnp.asarray(whitened["lam_time"]), jnp.asarray(whitened["lam_freq"]),
        jnp.asarray(whitened["joint_null"]), config,
    )
    kernel = NUTS(
        _native_model,
        init_strategy=init_to_value(values=init_values),
        target_accept_prob=0.85,
        max_tree_depth=10,
    )
    mcmc = MCMC(
        kernel, num_warmup=args.warmup, num_samples=args.samples,
        num_chains=args.chains, chain_method="sequential", progress_bar=False,
    )
    t0 = time.perf_counter()
    mcmc.run(
        random.PRNGKey(20260825), *model_args,
        extra_fields=("diverging", "num_steps", "accept_prob"),
    )
    runtime_s = time.perf_counter() - t0
    grouped = mcmc.get_samples(group_by_chain=True)
    diag = summary(grouped, group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    steps = np.asarray(extra["num_steps"])
    valid_ess = np.asarray(diag["w"]["n_eff"], dtype=float)
    valid_ess = valid_ess[np.isfinite(valid_ess) & (valid_ess > 0)]
    report["native_sparse_nuts_pilot"] = {
        "warmup_per_chain": args.warmup,
        "draws_per_chain": args.samples,
        "chains": args.chains,
        "runtime_including_compile_s": runtime_s,
        "mean_num_steps": float(np.mean(steps)),
        "median_num_steps": float(np.median(steps)),
        "max_tree_depth_hit_fraction": float(np.mean(steps >= 1023)),
        "divergences": int(np.sum(np.asarray(extra["diverging"]))),
        "w_ess_median": float(np.median(valid_ess)),
        "w_ess_min": float(np.min(valid_ess)),
        "w_rhat_max": float(np.nanmax(diag["w"]["r_hat"])),
        "phi_time_ess": float(diag["phi_time"]["n_eff"]),
        "phi_time_rhat": float(diag["phi_time"]["r_hat"]),
        "phi_freq_ess": float(diag["phi_freq"]["n_eff"]),
        "phi_freq_rhat": float(diag["phi_freq"]["r_hat"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
