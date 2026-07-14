"""Configuration for the WDM tensor-product log-P-spline PSD estimator."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal


@dataclass
class PSplineConfig:
    """Configuration for the smooth WDM log-power surface.

    The model fits ``log S(t, f) = B_t W B_f^T`` with a centered or
    non-centered tensor-product P-spline prior in a whitened eigenbasis.
    Smoothness is controlled by two precisions, ``phi_time`` and ``phi_freq``,
    each given a ``Gamma(alpha_phi, beta_phi)`` hyperprior and sampled on the
    log scale for a well-behaved geometry.
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

    # Time knots are always uniform. Frequency knots can instead be placed by
    # a cheap Whittle-MAP pilot or on a logarithmic frequency coordinate.
    freq_knot_strategy: Literal["adaptive", "linear", "log"] = "adaptive"

    # Parameterization of the eigen-coefficients. The non-centered (whitened)
    # default suits weak-data problems. On large grids the likelihood pins the
    # coefficients, so in non-centered form any move of phi must rescale every
    # coefficient coherently -- phi freezes (n_eff ~ 1, saturated tree depth).
    # The centered form samples the coefficients directly and decouples phi.
    centered: bool = False

    def __post_init__(self) -> None:
        """Reject invalid spline, prior, and trimming settings early."""
        integer_fields = (
            ("n_interior_knots_time", 0),
            ("n_interior_knots_freq", 0),
            ("degree_time", 0),
            ("degree_freq", 0),
            ("diff_order_time", 1),
            ("diff_order_freq", 1),
            ("trim_time_bins", 0),
            ("trim_low_freq_channels", 0),
            ("trim_high_freq_channels", 0),
        )
        for name, minimum in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                qualifier = "positive" if minimum == 1 else "non-negative"
                raise ValueError(f"{name} must be a {qualifier} integer.")

        axis_settings = (
            ("time", self.n_interior_knots_time, self.degree_time, self.diff_order_time),
            ("freq", self.n_interior_knots_freq, self.degree_freq, self.diff_order_freq),
        )
        for axis, n_knots, degree, diff_order in axis_settings:
            if diff_order > degree:
                raise ValueError(f"diff_order_{axis} must not exceed degree_{axis}.")
            n_basis = n_knots + degree + 1
            if diff_order >= n_basis:
                raise ValueError(
                    f"diff_order_{axis} must be smaller than the {n_basis} "
                    f"{axis}-basis functions."
                )

        for name in (
            "alpha_phi",
            "beta_phi",
            "null_precision",
            "ridge_eps",
            "phi_log_base_scale",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be strictly positive.")
        for name in (
            "init_penalty_time",
            "init_penalty_freq",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.freq_knot_strategy not in {"adaptive", "linear", "log"}:
            raise ValueError(
                "freq_knot_strategy must be one of 'adaptive', 'linear', or 'log'."
            )
