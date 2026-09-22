"""Check exact prior densities, including the log-precision Jacobian."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro.infer.util import log_density
from scipy.stats import gamma, halfnorm

from tv_pspline_psd import PSplineConfig
from tv_pspline_psd.model import _sample_smoothing_precision

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("scale", [0.5, 1.0, 10.0])
@pytest.mark.parametrize("base_scale", [0.5, 3.0])
def test_half_normal_log_precision_density(scale, base_scale):
    config = PSplineConfig(
        smoothing_prior="half_normal_sigma",
        roughness_scale=scale,
        phi_log_base_scale=base_scale,
    )
    for x in np.linspace(-6, 20, 17):
        actual, _ = log_density(
            lambda: _sample_smoothing_precision("phi", config),
            (),
            {},
            {"phi": jnp.asarray(x)},
        )
        expected = halfnorm.logpdf(np.exp(-x / 2), scale=scale) - x / 2 - np.log(2)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_legacy_gamma_is_unchanged():
    config = PSplineConfig()
    for x in [-4.0, 0.0, 2.0]:
        actual, _ = log_density(
            lambda: _sample_smoothing_precision("phi", config),
            (),
            {},
            {"phi": jnp.asarray(x)},
        )
        np.testing.assert_allclose(actual, gamma.logpdf(np.exp(x), 2) + x)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"roughness_scale": 0},
        {"roughness_scale": float("nan")},
        {"roughness_scale": float("inf")},
        {"smoothing_prior": "half_normal_phi"},
    ],
)
def test_invalid_prior_rejected(kwargs):
    with pytest.raises(ValueError):
        PSplineConfig(**kwargs)


def test_stationary_uses_configured_roughness_prior():
    from tv_pspline_psd.stationary import _stationary_model

    config = PSplineConfig(
        smoothing_prior="half_normal_sigma", roughness_scale=10, centered=True
    )
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(_stationary_model, 7)
    ).get_trace(
        jnp.ones(3),
        jnp.ones(3),
        jnp.eye(3),
        jnp.ones(3),
        jnp.array([True, False, False]),
        config,
    )
    x = trace["phi_freq"]["value"]
    actual = (
        trace["phi_freq"]["fn"].log_prob(x) + trace["phi_freq_prior"]["fn"].log_factor
    )
    expected = (
        halfnorm.logpdf(np.exp(-float(x) / 2), scale=10) - float(x) / 2 - np.log(2)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)
