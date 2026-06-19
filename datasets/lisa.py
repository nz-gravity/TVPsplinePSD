"""Non-stationary LISA noise: instrument PSD + modulated Galactic confusion.

The observed noise is the sum of two independent components,

    x(t) = x_inst(t) + m(u) x_gal(t),   u = t / T_obs in [0, 1],

where ``x_inst`` is stationary instrument noise, ``x_gal`` is stationary
confusion noise, and ``m(u)`` is a slowly-varying seasonal envelope with
``<m^2> = 1``. Because the components are independent and ``m`` varies slowly,
the local (evolutionary) PSD is exactly the WDM estimation target,

    S(u, f) = S_inst(f) + m(u)^2 S_gal(f) = E[w_nm^2].

The stationary shapes use the analytic Robson, Cornish & Liu (2019) fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (alpha, beta, kappa, gamma, f_knee[Hz]) Galactic-confusion fit by observation time.
_GALACTIC_FIT_PARAMS: dict[str, tuple[float, float, float, float, float]] = {
    "0.5yr": (0.133, 243.0, 482.0, 917.0, 2.58e-3),
    "1yr": (0.171, 292.0, 1020.0, 1680.0, 2.15e-3),
    "2yr": (0.165, 299.0, 611.0, 1340.0, 1.73e-3),
    "4yr": (0.138, -221.0, 521.0, 1680.0, 1.13e-3),
}

_SPEED_OF_LIGHT = 299_792_458.0


@dataclass
class LISANoiseConfig:
    """Configuration for the non-stationary LISA noise generator."""

    tobs_key: str = "1yr"
    arm_length: float = 2.5e9
    instrument_scale: float = 1.0
    galactic_scale: float = 1.0
    galactic_amplitude: float = 9e-45
    modulation_depth: float = 0.85
    n_modulation_cycles: float = 3.0
    modulation_phase: float = 0.0
    modulation_harmonic2: float = 0.4
    normalize: bool = True


def lisa_instrument_psd(freq_hz: np.ndarray, *, arm_length: float = 2.5e9) -> np.ndarray:
    """Analytic LISA instrument noise PSD (Robson, Cornish & Liu 2019), in 1/Hz."""
    f = np.asarray(freq_hz, dtype=float)
    f = np.where(f > 0.0, f, np.finfo(float).tiny)
    f_star = _SPEED_OF_LIGHT / (2.0 * np.pi * arm_length)
    p_oms = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / f) ** 4)
    p_acc = (3.0e-15) ** 2 * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8.0e-3) ** 4)
    return (
        p_oms + 2.0 * (1.0 + np.cos(f / f_star) ** 2) * p_acc / (2.0 * np.pi * f) ** 4
    ) / arm_length**2


def lisa_galactic_confusion_psd(
    freq_hz: np.ndarray, *, tobs_key: str = "1yr", amplitude: float = 9e-45
) -> np.ndarray:
    """Time-averaged Galactic confusion PSD (Robson, Cornish & Liu 2019), in 1/Hz."""
    if tobs_key not in _GALACTIC_FIT_PARAMS:
        raise ValueError(
            f"Unknown tobs_key {tobs_key!r}; choose from {sorted(_GALACTIC_FIT_PARAMS)}."
        )
    f = np.asarray(freq_hz, dtype=float)
    f = np.where(f > 0.0, f, np.finfo(float).tiny)
    alpha, beta, kappa, gamma, f_knee = _GALACTIC_FIT_PARAMS[tobs_key]
    return (
        amplitude
        * f ** (-7.0 / 3.0)
        * np.exp(-(f**alpha) + beta * f * np.sin(kappa * f))
        * (1.0 + np.tanh(gamma * (f_knee - f)))
    )


def galactic_modulation(u: np.ndarray, config: LISANoiseConfig) -> np.ndarray:
    """Seasonal modulation envelope ``m(u) >= 0`` with analytic ``<m^2> = 1``."""
    u = np.asarray(u, dtype=float)
    depth = float(config.modulation_depth)
    h2 = float(config.modulation_harmonic2)
    theta = 2.0 * np.pi * config.n_modulation_cycles * u + config.modulation_phase
    raw = 0.5 * (1.0 + np.cos(theta)) + h2 * 0.5 * (1.0 + np.cos(2.0 * theta))
    raw_max = 1.0 + h2
    # mean(raw) over whole cycles = 0.5 * raw_max => mean(power) = 1 - 0.5 * depth.
    power = ((1.0 - depth) + depth * raw / raw_max) / (1.0 - 0.5 * depth)
    return np.sqrt(np.maximum(power, 0.0))


def _component_psds(
    freq_hz: np.ndarray, config: LISANoiseConfig
) -> tuple[np.ndarray, np.ndarray]:
    s_inst = config.instrument_scale * lisa_instrument_psd(
        freq_hz, arm_length=config.arm_length
    )
    s_gal = config.galactic_scale * lisa_galactic_confusion_psd(
        freq_hz, tobs_key=config.tobs_key, amplitude=config.galactic_amplitude
    )
    return s_inst, s_gal


def simulate_colored_noise(
    psd_onesided: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Synthesise a real stationary series with a given digital PSD.

    Convention: unit-variance white noise corresponds to a flat PSD equal to 1,
    and the squared WDM coefficients of the output have expectation equal to
    ``psd_onesided`` at the matching channel.
    """
    psd_onesided = np.asarray(psd_onesided, dtype=float)
    n_freq = n // 2 + 1
    if psd_onesided.shape[0] != n_freq:
        raise ValueError("psd_onesided must have length n // 2 + 1.")
    scale = np.sqrt(n * np.maximum(psd_onesided, 0.0))
    spectrum = np.zeros(n_freq, dtype=complex)
    spectrum[0] = scale[0] * rng.standard_normal()
    if n % 2 == 0:
        n_interior = n_freq - 2
        if n_interior > 0:
            noise = rng.standard_normal(n_interior) + 1j * rng.standard_normal(n_interior)
            spectrum[1:-1] = scale[1:-1] * noise / np.sqrt(2.0)
        spectrum[-1] = scale[-1] * rng.standard_normal()
    else:
        n_interior = n_freq - 1
        noise = rng.standard_normal(n_interior) + 1j * rng.standard_normal(n_interior)
        spectrum[1:] = scale[1:] * noise / np.sqrt(2.0)
    return np.fft.irfft(spectrum, n=n)


def normalization_constant(n: int, dt: float, config: LISANoiseConfig) -> float:
    """Median in-band level used to normalise the LISA PSD to ``O(1)``.

    Deterministic in ``(n, dt, config)`` so the simulator, the analytic truth,
    and the Monte Carlo reference all share the same scale.
    """
    if not config.normalize:
        return 1.0
    freq = np.fft.rfftfreq(n, d=dt)
    freq_eval = freq.copy()
    if freq_eval.size > 1:
        freq_eval[0] = freq_eval[1]
    s_inst, s_gal = _component_psds(freq_eval, config)
    return float(np.median((s_inst + s_gal)[1:]))


def simulate_tv_lisa_noise(
    n: int, *, dt: float, rng: np.random.Generator, config: LISANoiseConfig
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Simulate non-stationary LISA noise with a modulated Galactic foreground.

    Returns ``(data, meta)`` where ``meta`` carries the normalised component
    PSDs, the rfft frequencies, and ``norm_ref`` (needed to reproduce the
    analytic ``S(u, f)`` on the same scale).
    """
    freq_hz = np.fft.rfftfreq(n, d=dt)
    freq_eval = freq_hz.copy()
    if freq_eval.size > 1:
        freq_eval[0] = freq_eval[1]
    s_inst, s_gal = _component_psds(freq_eval, config)

    norm_ref = normalization_constant(n, dt, config)
    s_inst = s_inst / norm_ref
    s_gal = s_gal / norm_ref

    x_inst = simulate_colored_noise(s_inst, n, rng)
    x_gal = simulate_colored_noise(s_gal, n, rng)
    u = np.arange(n, dtype=float) / n
    data = x_inst + galactic_modulation(u, config) * x_gal

    meta = {"freq_hz": freq_hz, "s_inst": s_inst, "s_gal": s_gal, "norm_ref": norm_ref}
    return data, meta


def true_psd_lisa(
    time_grid: np.ndarray,
    freq_grid_hz: np.ndarray,
    config: LISANoiseConfig,
    *,
    norm_ref: float = 1.0,
) -> np.ndarray:
    """Analytic non-stationary LISA PSD surface ``S(u, f)`` on a WDM grid."""
    s_inst, s_gal = _component_psds(np.asarray(freq_grid_hz), config)
    s_inst = s_inst / norm_ref
    s_gal = s_gal / norm_ref
    m2 = galactic_modulation(time_grid, config) ** 2
    return s_inst[None, :] + m2[:, None] * s_gal[None, :]
