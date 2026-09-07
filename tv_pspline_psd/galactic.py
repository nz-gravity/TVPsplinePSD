"""Karnesis et al. (2021), arXiv:2103.14598v1, Eqs. 6--7/Table II.

This is the residual unresolved-binary *strain* spectrum, before applying
the sky and TDI response. Frequencies are Hz and subtraction duration is years.
The five spectral parameters are independent in inference; Eq. 7 selects
simulation injection values, not an inference constraint.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import log_expit


@dataclass(frozen=True)
class GalacticParameters:
    amplitude: float
    f1_hz: float
    f2_hz: float
    f_knee_hz: float
    alpha: float

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("All five Galactic parameters must be finite and positive")

    def to_dict(self):
        return asdict(self)


# A, alpha, f2, a1, b1, ak, bk; Table II columns, not rounded notebook values.
TABLE_II = {
    "mean_snr5": (1.15e-44, 1.66, .00059, -.16, -2.78, -.34, -2.53),
    "mean_snr7": (1.14e-44, 1.80, .00031, -.25, -2.70, -.27, -2.47),
    "median_snr5": (1.14e-44, 1.66, .00059, -.15, -2.78, -.34, -2.55),
    "median_snr7": (1.15e-44, 1.56, .00067, -.15, -2.72, -.37, -2.49),
}


def simulation_parameters(tobs_years=1.0, preset="mean_snr7"):
    """Choose injection parameters; the paper checked 0.25--10 years."""
    if not np.isfinite(tobs_years) or not .25 <= tobs_years <= 10:
        raise ValueError("subtraction duration must be between 0.25 and 10 years")
    A, alpha, f2, a1, b1, ak, bk = TABLE_II[preset]
    return GalacticParameters(A, 10**b1 * tobs_years**a1, f2,
                              10**bk * tobs_years**ak, alpha)


def log_galactic_psd(frequency_hz, parameters):
    """Stable Eq. 6, including 1/2 through (1+tanh(x))/2 = sigmoid(2x)."""
    f = np.asarray(frequency_hz, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("evaluate the strain PSD only at finite positive frequencies")
    p = parameters
    return (np.log(p.amplitude) - (7/3)*np.log(f) - (f/p.f1_hz)**p.alpha
            + log_expit(2*(p.f_knee_hz-f)/p.f2_hz))


def galactic_psd(frequency_hz, parameters):
    return np.exp(log_galactic_psd(frequency_hz, parameters))


def log_galactic_psd_jax(f, log_amplitude, log_f1, log_f2, log_f_knee, log_alpha):
    """Differentiable Eq. 6; all five arguments are natural-log physical values."""
    import jax
    import jax.numpy as jnp
    return (log_amplitude - (7/3)*jnp.log(f)
            - jnp.exp(jnp.exp(log_alpha)*(jnp.log(f)-log_f1))
            + jax.nn.log_sigmoid(2*(jnp.exp(log_f_knee)-f)/jnp.exp(log_f2)))
