"""Variational warm start for the P-spline surface sampler.

A staged initialisation between the analytic penalized-least-squares (PLS) point
and NUTS: a diagonal-guide SVI run refines the PLS ``init_sites`` for a few
thousand cheap steps, and its posterior medians seed ``init_to_value`` for NUTS.
Because every latent in :func:`tv_pspline_psd.model.pspline_surface_model`
(``s``, ``phi_time``, ``phi_freq``) is an unconstrained ``Normal`` site -- the
priors are imposed via ``numpyro.factor`` -- a mean-field diagonal guide fits the
geometry directly, with no constraint transforms to manage.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import optax
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer.util import init_to_value


def vi_warmstart(
    model: Callable[..., Any],
    model_args: tuple,
    init_sites: dict[str, jnp.ndarray],
    *,
    rng_key,
    steps: int = 2000,
    lr: float = 1e-2,
    progress_bar: bool = True,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Refine analytic ``init_sites`` with diagonal-guide VI.

    Args:
        model: NumPyro model (e.g. ``pspline_surface_model``).
        model_args: Positional model arguments, exactly as passed to ``mcmc.run``.
        init_sites: PLS warm-start values used to initialise both the guide and
            the returned dict (so any site VI does not report falls back to PLS).
        rng_key: JAX PRNG key.
        steps: Number of SVI iterations.
        lr: Adam learning rate.
        progress_bar: Show the SVI iteration progress bar.

    Returns:
        ``(refined_init_sites, losses)`` -- the VI posterior medians merged over
        ``init_sites``, ready for ``init_to_value``, and the ELBO loss trace.
    """
    guide = AutoDiagonalNormal(
        model, init_loc_fn=init_to_value(values=init_sites)
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    result = svi.run(rng_key, steps, *model_args, progress_bar=progress_bar)
    median = guide.median(result.params)
    refined = {**init_sites, **{k: v for k, v in median.items() if k in init_sites}}
    return refined, result.losses


__all__ = ["vi_warmstart"]
