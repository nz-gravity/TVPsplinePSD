"""Regression tests for centered and non-centered model invariants."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from jax import random
from numpyro import handlers

from tv_pspline_psd import PSplineConfig
from tv_pspline_psd.inference import reconstruct_eig_coeff_samples
from tv_pspline_psd.joint import (
    _multichannel_joint_model,
    run_gibbs_signal_noise_mcmc,
)
from tv_pspline_psd.model import (
    eigen_prior_scale,
    pspline_surface_model,
    whitened_init_values,
)
from tv_pspline_psd.moving_periodogram import _dynamic_whittle_model
from tv_pspline_psd.stationary import _stationary_model


def _trace(model, values, *args):
    seeded = handlers.seed(model, random.PRNGKey(0))
    substituted = handlers.substitute(seeded, data=values)
    return handlers.trace(substituted).get_trace(*args)


def _tensor_problem():
    basis_time = jnp.asarray([[1.0, 0.2], [0.3, 1.0]])
    basis_freq = jnp.asarray([[1.0, -0.1], [0.4, 1.0]])
    lam_time = jnp.asarray([0.0, 2.0])
    lam_freq = jnp.asarray([0.0, 3.0])
    joint_null = jnp.asarray([[True, False], [False, False]])
    eig_coeffs = jnp.asarray([[0.2, -0.4], [0.7, 0.1]])
    return basis_time, basis_freq, lam_time, lam_freq, joint_null, eig_coeffs


def _tensor_values(config, eig_coeffs, lam_time, lam_freq, joint_null, *, suffix=""):
    phi_time, phi_freq = 1.7, 0.8
    scale = eigen_prior_scale(
        phi_time, phi_freq, lam_time, lam_freq, joint_null, config
    )
    s = eig_coeffs if config.centered else eig_coeffs / scale
    return {
        f"s{suffix}": s.reshape(-1),
        f"phi_time{suffix}": np.log(phi_time),
        f"phi_freq{suffix}": np.log(phi_freq),
    }


def test_core_centered_and_noncentered_surfaces_match() -> None:
    bt, bf, lt, lf, null, eig = _tensor_problem()
    surfaces = []
    for centered in (False, True):
        config = PSplineConfig(centered=centered)
        trace = _trace(
            pspline_surface_model,
            _tensor_values(config, eig, lt, lf, null),
            jnp.zeros((2, 2)), 1, bt, bf, lt, lf, null, config, True,
        )
        surfaces.append(np.asarray(trace["log_psd"]["value"]))
    np.testing.assert_allclose(surfaces[0], surfaces[1], rtol=1e-6, atol=1e-6)


def test_multichannel_centered_and_noncentered_surfaces_match() -> None:
    bt, bf, lt, lf, null, eig = _tensor_problem()
    surfaces = []
    for centered in (False, True):
        config = PSplineConfig(centered=centered)
        values = _tensor_values(config, eig, lt, lf, null, suffix="_0")
        values["beta"] = jnp.zeros(1)
        trace = _trace(
            _multichannel_joint_model,
            values,
            jnp.zeros((1, 2, 2)), jnp.zeros((1, 1, 2, 2)),
            bt, bf, lt, lf, null, 1.0, config,
        )
        surfaces.append(np.asarray(trace["log_psd_0"]["value"]))
    np.testing.assert_allclose(surfaces[0], surfaces[1], rtol=1e-6, atol=1e-6)


def test_tang_centered_and_noncentered_coefficients_match() -> None:
    bt, bf, lt, lf, null, eig = _tensor_problem()
    # Two frequency rungs repeated for each time block.
    basis_time = jnp.repeat(bt, 2, axis=0)
    coefficients = []
    for centered in (False, True):
        config = PSplineConfig(centered=centered)
        trace = _trace(
            _dynamic_whittle_model,
            _tensor_values(config, eig, lt, lf, null),
            jnp.ones(4), basis_time, bf, lt, lf, null, config,
        )
        coefficients.append(np.asarray(trace["eig_coeffs"]["value"]))
    np.testing.assert_allclose(coefficients[0], coefficients[1], rtol=1e-6, atol=1e-6)


def test_stationary_centered_and_noncentered_surfaces_match() -> None:
    basis_freq = jnp.asarray([[1.0, -0.1], [0.4, 1.0]])
    lam_freq = jnp.asarray([0.0, 3.0])
    null_freq = jnp.asarray([True, False])
    eig_coeffs = jnp.asarray([0.2, -0.4])
    phi = 0.8
    surfaces = []
    for centered in (False, True):
        config = PSplineConfig(centered=centered)
        scale = eigen_prior_scale(
            0.0, phi, jnp.zeros(1), lam_freq, null_freq[None, :], config
        )[0]
        s = eig_coeffs if centered else eig_coeffs / scale
        trace = _trace(
            _stationary_model,
            {"s": s, "phi_freq": np.log(phi)},
            jnp.ones(2), 3.0, basis_freq, lam_freq, null_freq, config,
        )
        surfaces.append(np.asarray(trace["log_psd"]["value"]))
    np.testing.assert_allclose(surfaces[0], surfaces[1], rtol=1e-6, atol=1e-6)


def test_init_reconstruction_round_trip_for_both_parameterizations() -> None:
    whitened = {
        "U_time": np.asarray([[0.8, -0.6], [0.6, 0.8]]),
        "U_freq": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "lam_time": np.asarray([0.0, 2.0]),
        "lam_freq": np.asarray([0.0, 3.0]),
        "joint_null": np.asarray([[True, False], [False, False]]),
    }
    pls = {
        "W": np.asarray([[0.2, -0.4], [0.7, 0.1]]),
        "phi_time": 1.7,
        "phi_freq": 0.8,
    }
    expected = whitened["U_time"].T @ pls["W"] @ whitened["U_freq"]
    for centered in (False, True):
        config = PSplineConfig(centered=centered)
        init = whitened_init_values(pls, whitened, config)
        samples = {key: np.asarray(value)[None] for key, value in init.items()}
        reconstructed = reconstruct_eig_coeff_samples(samples, whitened, config)
        np.testing.assert_allclose(reconstructed[0], expected, rtol=1e-12, atol=1e-12)


def test_gibbs_joint_smoke_for_both_parameterizations() -> None:
    rng = np.random.default_rng(5)
    template = rng.normal(size=(1, 8, 8))
    coeffs = 0.3 * template[0] + rng.normal(scale=0.7, size=(8, 8))
    base_config = PSplineConfig(
        n_interior_knots_time=1,
        n_interior_knots_freq=1,
        degree_time=1,
        degree_freq=1,
        diff_order_time=1,
        diff_order_freq=1,
        adaptive_time_knots=False,
    )
    beta_means = []
    for centered in (False, True):
        result = run_gibbs_signal_noise_mcmc(
            coeffs,
            template,
            np.linspace(0.0, 1.0, 8),
            np.linspace(0.1, 1.0, 8),
            config=replace(base_config, centered=centered),
            n_sweeps=2,
            n_burn_sweeps=1,
            block_warmup=2,
            block_samples=2,
            random_seed=9,
            max_tree_depth=3,
        )
        assert np.isfinite(result["psd_geometric_mean"]).all()
        assert result["psd_mean"] is result["psd_geometric_mean"]
        beta_means.append(np.asarray(result["beta_mean"]))
    np.testing.assert_allclose(beta_means[0], beta_means[1], rtol=1.0, atol=1.0)
