"""Configuration for the WDM tensor-product log-P-spline PSD estimator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PSplineConfig:
    """Configuration for the smooth WDM log-power surface.

    The model fits ``log S(t, f) = B_t W B_f^T`` with a non-centered (whitened)
    tensor-product P-spline prior. Smoothness is controlled by two precisions,
    ``phi_time`` and ``phi_freq``, each given a ``Gamma(alpha_phi, beta_phi)``
    hyperprior and sampled on the log scale for a well-behaved geometry.
    """

    # Basis.
    n_interior_knots_time: int = 8
    n_interior_knots_freq: int = 10
    degree_time: int = 3
    degree_freq: int = 3

    # Roughness penalty (derivative order in each direction).
    diff_order_time: int = 2
    diff_order_freq: int = 2

    # Smoothing-precision hyperprior: phi ~ Gamma(alpha_phi, beta_phi).
    alpha_phi: float = 2.0
    beta_phi: float = 1.0

    # Whitened-prior numerics.
    null_precision: float = 1e-4  # weak prior on the penalty null space (bilinear trend)
    ridge_eps: float = 1e-6  # guards the eigen-scale against division by zero
    phi_log_base_scale: float = 1.0  # reference scale for the log-phi sampling site

    # Penalized-least-squares warm start.
    init_penalty_time: float = 5e-2
    init_penalty_freq: float = 5e-2

    # WDM-grid trimming (drop edge bins/channels with strong boundary effects).
    trim_time_bins: int = 1
    trim_low_freq_channels: int = 1
    trim_high_freq_channels: int = 1

    # Adaptive time-knot placement from a pilot time profile.
    adaptive_time_knots: bool = True
    adaptive_time_knot_smoothing: float = 1.0
    adaptive_time_knot_floor: float = 0.25

    # Parameterization of the eigen-coefficients. The non-centered (whitened)
    # default suits weak-data problems. On large grids the likelihood pins the
    # coefficients, so in non-centered form any move of phi must rescale every
    # coefficient coherently -- phi freezes (n_eff ~ 1, saturated tree depth).
    # The centered form samples the coefficients directly and decouples phi.
    centered: bool = False
