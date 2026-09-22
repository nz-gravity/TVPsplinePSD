"""Initialization and NUTS execution for time-frequency surface models."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value

from ._surface_setup import LikelihoodGrid, SurfaceBasis
from .binning import _mean_power_for_masked_initialization
from .config import PSplineConfig
from .model import (
    initialize_with_penalized_least_squares,
    nested_residual_surface_model,
    power_floor,
    pspline_surface_model,
    whitened_init_values,
)


def run_surface_nuts(
    model: Callable,
    model_args: tuple,
    init_sites: dict[str, Any],
    *,
    n_warmup: int,
    n_samples: int,
    num_chains: int,
    random_seed: int,
    max_tree_depth: int,
    target_accept_prob: float,
    progress_bar: bool,
    initial_state: Any | None,
) -> tuple[MCMC, float]:
    """Run NUTS or advance an adapted chain under the current residual data."""
    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=init_sites),
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress_bar,
    )
    if initial_state is not None:
        # Advancing a persistent chain: reuse the position and the adapted step
        # size and mass matrix, and skip warmup entirely. The stored potential
        # energy and gradient belong to the *previous* data (residual), so they
        # must be recomputed under the new model args or every proposal is
        # rejected through a stale energy difference.
        from jax import value_and_grad
        from numpyro.infer.util import potential_energy as _potential_energy

        refreshed_pe, refreshed_grad = value_and_grad(
            lambda z: _potential_energy(model, model_args, {}, z)
        )(initial_state.z)
        mcmc.post_warmup_state = initial_state._replace(
            potential_energy=refreshed_pe, z_grad=refreshed_grad
        )
    nuts_t0 = time.perf_counter()
    mcmc.run(
        random.PRNGKey(random_seed),
        *model_args,
        extra_fields=(
            "diverging",
            "accept_prob",
            "num_steps",
            "potential_energy",
            "energy",
        ),
    )
    nuts_runtime_s = time.perf_counter() - nuts_t0

    return mcmc, nuts_runtime_s


def initialize_nested_surface(
    init_power: np.ndarray,
    counts_fit: np.ndarray | int,
    basis_nested_freq_fit: np.ndarray,
    basis_interaction_time_fit: np.ndarray,
    B_time_interaction_fit: np.ndarray,
    B_freq_fit: np.ndarray,
    P_time_interaction: np.ndarray,
    P_freq: np.ndarray,
    whitened_interaction: dict[str, np.ndarray],
    config: PSplineConfig,
) -> dict[str, Any]:
    """Initialize the stationary correction and centered interaction from power."""
    target = np.log(init_power + power_floor(init_power))
    count_weights = np.broadcast_to(np.asarray(counts_fit, dtype=float), target.shape)
    stationary_target = np.divide(
        np.sum(target * count_weights, axis=0),
        count_weights.sum(axis=0),
        out=np.median(target, axis=0),
        where=count_weights.sum(axis=0) > 0,
    )
    lam_f = whitened_interaction["lam_freq"]
    g_system = (
        basis_nested_freq_fit.T @ basis_nested_freq_fit
        + config.init_penalty_freq * np.diag(lam_f)
        + config.ridge_eps * np.eye(lam_f.size)
    )
    g_init = np.linalg.solve(g_system, basis_nested_freq_fit.T @ stationary_target)
    phi_stationary_init = max(
        1e-2,
        g_init.size / (float(np.sum(lam_f * g_init**2)) + 1e-6),
    )
    interaction_target = target - (basis_nested_freq_fit @ g_init)[None, :]
    interaction_power = np.exp(interaction_target)
    interaction_pls = initialize_with_penalized_least_squares(
        interaction_power,
        B_time_interaction_fit,
        B_freq_fit,
        P_time_interaction,
        P_freq,
        config,
    )
    h_eig_init = (
        whitened_interaction["U_time"].T
        @ np.asarray(interaction_pls["W"])
        @ whitened_interaction["U_freq"]
    )
    h_surface_init = basis_interaction_time_fit @ h_eig_init @ basis_nested_freq_fit.T
    sigma_init = float(np.clip(np.std(h_surface_init), 0.05, 1.0))
    init_sites = {
        "g": g_init,
        "h": h_eig_init.reshape(-1),
        "phi_stationary": float(np.log(phi_stationary_init)),
        "sigma_interaction": sigma_init,
    }
    return init_sites


def prepare_surface_model(
    grid: LikelihoodGrid,
    basis: SurfaceBasis,
    config: PSplineConfig,
    *,
    mask_applied: bool,
    interaction_scale_prior: float,
    likelihood_beta: float,
) -> tuple[Callable, tuple, dict[str, Any]]:
    """Select the residual model and prepare its arguments and initial sites."""
    nested = basis.interaction is not None
    # The warm start fits log S to the per-component mean power, matching the
    # likelihood mode (S = mean of squared components), on the fit grid.
    init_power = (
        _mean_power_for_masked_initialization(grid.power, grid.counts)
        if mask_applied
        else grid.power / grid.counts
    )
    init_power = init_power * np.exp(-grid.log_offset)
    if nested:
        init_sites = initialize_nested_surface(
            init_power,
            grid.counts,
            grid.basis_nested_freq,
            grid.basis_interaction_time,
            grid.B_time_interaction,
            grid.B_freq,
            basis.interaction.P_time,
            basis.P_freq,
            basis.interaction.whitened,
            config,
        )
        model = nested_residual_surface_model
        model_args = (
            jnp.asarray(grid.power),
            jnp.asarray(grid.counts),
            jnp.asarray(grid.basis_interaction_time),
            jnp.asarray(grid.basis_nested_freq),
            jnp.asarray(basis.interaction.whitened["lam_time"]),
            jnp.asarray(basis.interaction.whitened["lam_freq"]),
            jnp.asarray(basis.interaction.whitened["joint_null"]),
            jnp.asarray(basis.interaction.null_freq),
            config,
            interaction_scale_prior,
            False,
            likelihood_beta,
            jnp.asarray(grid.log_offset),
        )
    else:
        pls_init = initialize_with_penalized_least_squares(
            init_power, grid.B_time, grid.B_freq, basis.P_time, basis.P_freq, config
        )
        init_sites = whitened_init_values(pls_init, basis.whitened, config)
        model = pspline_surface_model
        model_args = (
            jnp.asarray(grid.power),
            jnp.asarray(grid.counts),
            jnp.asarray(grid.basis_time),
            jnp.asarray(grid.basis_freq),
            jnp.asarray(basis.whitened["lam_time"]),
            jnp.asarray(basis.whitened["lam_freq"]),
            jnp.asarray(basis.whitened["joint_null"]),
            config,
            False,  # never store the per-sample log_psd surface; reconstruct instead
            likelihood_beta,
            jnp.asarray(grid.log_offset),
        )
    return model, model_args, init_sites
