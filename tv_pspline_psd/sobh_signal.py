"""Differentiable SOBH signal -> WDM coefficients (jax), for the joint fit.

The chirping-binary signal of :mod:`datasets.sobh` enters the joint
signal+noise fit as a *nonlinear* mean. To keep the joint sampler on NUTS, the
whole map

    theta = (Mc, tc, ln dL)  ->  ripple h+(f), hx(f)  ->  h(t)  ->  WDM coeffs

must be differentiable. It is: ``ripple`` is jax-native, the FFT embedding and
the (fixed) long-wavelength antenna modulation are jax ops, and
``wdm_transform`` ships a jax backend whose forward transform is
differentiable. This module precomputes the static pieces (frequency grid,
antenna patterns, WDM trim indices) and returns a jitted ``theta -> coeffs``
closure that is grid-consistent with :func:`wdm_analysis_coefficients`.

The sampled parameters are the three that drive the demonstration: the chirp
mass ``Mc`` (chirp track), the coalescence time ``tc`` (places the track against
the cyclostationary confusion), and ``ln dL`` (distance/SNR). The remaining
source parameters are held fixed at their injected values.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from wdm_transform import TimeSeries, get_backend
from wdm_transform.transforms import from_time_to_wdm

from .config import PSplineConfig


@dataclass(frozen=True)
class SOBHWDMSignalGrid:
    """Static, theta-independent pieces of the SOBH -> WDM signal map."""

    n: int
    dt: float
    nt: int
    nf: int
    a: float
    d: float
    f_ref: float
    freqs_pos: np.ndarray       # (n//2,) positive rfft frequencies
    f_plus: np.ndarray          # (n,) long-wavelength antenna patterns
    f_cross: np.ndarray
    keep_time: np.ndarray       # trim indices into the WDM grid
    keep_freq: np.ndarray
    time_grid: np.ndarray       # trimmed, rescaled to [0, 1]
    freq_grid: np.ndarray       # trimmed (Hz)
    fixed: dict[str, float]     # eta, chi1, chi2, phic, inc


def build_sobh_wdm_grid(
    n: int, dt: float, nt: int, params, config: PSplineConfig
) -> SOBHWDMSignalGrid:
    """Precompute the static grid for the differentiable SOBH -> WDM map.

    ``params`` is a :class:`datasets.sobh.SOBHParams`. The trim indices and grids
    reproduce those of :func:`wdm_analysis_coefficients` for the same ``(n, dt,
    nt, config)``, so signal and data WDM coefficients share one grid.
    """
    from datasets.sobh import lisa_lw_antenna

    a, d = 1.0 / 3.0, 1.0
    freqs = np.fft.rfftfreq(n, d=dt)
    ref = TimeSeries(np.zeros(n), dt=dt).to_wdm(nt=nt, a=a, d=d)
    nf = int(ref.nf)
    keep_time = np.arange(config.trim_time_bins, ref.nt - config.trim_time_bins)
    keep_freq = np.arange(
        config.trim_low_freq_channels, ref.nf + 1 - config.trim_high_freq_channels
    )
    time_grid = np.asarray(ref.time_grid)[keep_time] / ref.duration
    freq_grid = np.asarray(ref.freq_grid)[keep_freq]

    t = np.arange(n, dtype=float) * dt
    f_plus, f_cross = lisa_lw_antenna(t, params)

    return SOBHWDMSignalGrid(
        n=n, dt=dt, nt=int(ref.nt), nf=nf, a=a, d=d, f_ref=float(params.f_ref),
        freqs_pos=freqs[1:], f_plus=f_plus, f_cross=f_cross,
        keep_time=keep_time, keep_freq=keep_freq,
        time_grid=time_grid, freq_grid=freq_grid,
        fixed={"eta": params.eta, "chi1": params.chi1, "chi2": params.chi2,
               "phic": params.phic, "inc": params.inc},
    )


def make_sobh_wdm_signal_fn(grid: SOBHWDMSignalGrid):
    """Return a jax closure ``theta=(Mc, tc, ln_dL) -> WDM signal coeffs``.

    The output has shape ``(len(keep_time), len(keep_freq))`` -- the trimmed
    analysis grid -- matching the data coefficients from
    :func:`wdm_analysis_coefficients`.
    """
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc

    backend = get_backend("jax")
    freqs = jnp.asarray(grid.freqs_pos)
    fp = jnp.asarray(grid.f_plus)
    fc = jnp.asarray(grid.f_cross)
    n, dt = grid.n, grid.dt
    eta = grid.fixed["eta"]
    chi1, chi2 = grid.fixed["chi1"], grid.fixed["chi2"]
    phic, inc = grid.fixed["phic"], grid.fixed["inc"]
    keep_time = jnp.asarray(grid.keep_time)
    keep_freq = jnp.asarray(grid.keep_freq)

    def _embed(h_fd):
        spectrum = jnp.zeros(n // 2 + 1, dtype=jnp.complex128)
        spectrum = spectrum.at[1:].set(jnp.nan_to_num(h_fd) / dt)
        return jnp.fft.irfft(spectrum, n=n)

    def signal_coeffs(theta):
        mc, tc, ln_dl = theta[0], theta[1], theta[2]
        ripple_theta = jnp.array([mc, eta, chi1, chi2, jnp.exp(ln_dl), tc, phic, inc])
        hp, hc = gen_IMRPhenomD_hphc(freqs, ripple_theta, grid.f_ref)
        h = fp * _embed(hp) + fc * _embed(hc)
        coeffs = from_time_to_wdm(
            h, nt=grid.nt, nf=grid.nf, a=grid.a, d=grid.d, dt=dt, backend=backend
        )
        coeffs = coeffs[0] if coeffs.ndim == 3 else coeffs
        return coeffs[jnp.ix_(keep_time, keep_freq)]

    return signal_coeffs
