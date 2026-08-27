"""JAX hot-path benchmark for the production-like Ollie WDM spline grid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import potential_energy

from tv_pspline_psd import PSplineConfig
from tv_pspline_psd.inference import _prepare_spline_bases
from tv_pspline_psd.model import (
    initialize_with_penalized_least_squares,
    power_whittle_log_likelihood,
    pspline_surface_model,
    tensor_product_surface,
    whiten_penalty_pair,
    whitened_init_values,
)
from tv_pspline_psd.splines import create_bspline_roughness_penalty

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "studies/results/ollie_tdi/knot_map_benchmark/wdm_coeffs.npz"
DEFAULT_OUTPUT = REPO / "studies/results/ollie_tdi/performance/hot_path.json"


def _bench(fn, argument, repetitions=100):
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
    args = parser.parse_args()

    cached = np.load(args.cache)
    power_np = np.asarray(cached["coeffs"], dtype=float) ** 2
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
    basis_time = jnp.asarray(spline["B_time"] @ whitened["U_time"])
    basis_freq = jnp.asarray(spline["B_freq"] @ whitened["U_freq"])
    power = jnp.asarray(power_np)
    counts = jnp.ones_like(power)
    rng = np.random.default_rng(20260825)
    coefficients = jnp.asarray(
        rng.normal(scale=0.1, size=(basis_time.shape[1], basis_freq.shape[1]))
    )
    def left(value):
        return (basis_time @ value) @ basis_freq.T

    def right(value):
        return basis_time @ (value @ basis_freq.T)

    def einsum(value):
        return tensor_product_surface(basis_time, value, basis_freq)

    def log_like(form):
        def evaluate(value):
            return power_whittle_log_likelihood(power, counts, form(value))

        return evaluate

    surface = einsum(coefficients)

    def first_left(value):
        return basis_time @ value

    def first_right(value):
        return value @ basis_freq.T

    def counts_term(value):
        return counts * value

    def power_term(value):
        return power * jnp.exp(-value)

    def reduction(value):
        return -0.5 * jnp.sum(value)

    pls = initialize_with_penalized_least_squares(
        power_np, spline["B_time"], spline["B_freq"],
        penalty_time, penalty_freq, config,
    )
    init_values_np = whitened_init_values(pls, whitened, config)
    init_values = {name: jnp.asarray(value) for name, value in init_values_np.items()}
    model_args = (
        power, 1, basis_time, basis_freq,
        jnp.asarray(whitened["lam_time"]), jnp.asarray(whitened["lam_freq"]),
        jnp.asarray(whitened["joint_null"]), config, False, 1.0,
        jnp.zeros_like(power),
    )
    def potential(z):
        return potential_energy(pspline_surface_model, model_args, {}, z)
    n_time, n_freq = power.shape
    k_time, k_freq = coefficients.shape
    left_flops = n_time * k_time * k_freq + n_time * k_freq * n_freq
    right_flops = k_time * k_freq * n_freq + n_time * k_time * n_freq
    report = {
        "backend": jax.default_backend(),
        "device": jax.devices()[0].device_kind,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "likelihood_grid_shape": [n_time, n_freq],
        "basis_shape": [k_time, k_freq],
        "sampled_parameters": int(k_time * k_freq + 2),
        "theoretical_contraction_multiply_add_counts": {
            "left_associated": int(left_flops),
            "right_associated": int(right_flops),
            "left_over_right": float(left_flops / right_flops),
        },
        "components": {
            "A_Bt_matmul_W": _bench(first_left, coefficients),
            "A_alternative_W_matmul_BfT": _bench(first_right, coefficients),
            "B_full_surface_optimized": _bench(einsum, coefficients),
            "C1_counts_times_log_psd": _bench(counts_term, surface),
            "C2_power_times_exp_minus_log_psd": _bench(power_term, surface),
            "C3_reduction": _bench(reduction, counts_term(surface) + power_term(surface)),
            "C_all_whittle_elementwise_and_reduction": _bench(
                lambda value: power_whittle_log_likelihood(power, counts, value), surface
            ),
            "D_complete_log_likelihood": _bench(log_like(einsum), coefficients, 50),
            "E_value_and_grad_log_likelihood": _bench(
                jax.value_and_grad(log_like(einsum)), coefficients, 50
            ),
            "F_complete_potential_energy": _bench(potential, init_values, 50),
            "F_value_and_grad_potential_energy": _bench(
                jax.value_and_grad(potential), init_values, 50
            ),
        },
        "contraction_strategies": {},
    }
    for name, form in (("left", left), ("right", right), ("einsum", einsum)):
        report["contraction_strategies"][name] = {
            "forward": _bench(form, coefficients),
            "complete_log_likelihood": _bench(log_like(form), coefficients, 50),
            "value_and_grad_log_likelihood": _bench(
                jax.value_and_grad(log_like(form)), coefficients, 50
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
