"""Consistent LISA TDI signal + noise generation (jaxGB + lisatools).

The galactic-binary signal (``jaxGB``) and the instrument noise PSD
(``lisatools``) MUST share the same TDI channel, TDI generation, and units, or
the signal-to-noise ratio is physically meaningless. We use TDI-X, **generation
2** for both (matching ``XYZ2SensitivityMatrix`` and ``jaxGB
tdi_generation=2.0``); mixing generations changes the SNR by a large,
frequency-dependent factor.

FFT convention (one realisation, real series of length ``n`` at spacing ``dt``,
``T = n dt``, ``df = 1/T``): the continuous Fourier transform is
``h(f_k) = dt * rfft(x)_k`` and the one-sided PSD is
``S(f_k) = (2 dt / n) |rfft(x)_k|^2``. Noise is drawn so its one-sided PSD is the
lisatools TDI PSD; the jaxGB signal ``h(f)`` is embedded as ``rfft_k = h/dt`` at
its absolute bins. With this single convention the matched-filter SNR
``SNR^2 = 4 df sum_k |h_k|^2 / S(f_k)`` is consistent between signal and noise.

Requires the optional ``[lisa]`` dependencies (``uv pip install -e '.[lisa]'``).


The observed noise is the sum of two independent components,

    x(t) = x_inst(t) + m(u) x_gal(t),   u = t / T_obs in [0, 1],

where ``x_inst`` is stationary instrument noise, ``x_gal`` is stationary
confusion noise, and ``m(u)`` is a slowly-varying seasonal envelope with
``<m^2> = 1``. Because the components are independent and ``m`` varies slowly,
the local (evolutionary) PSD is exactly the WDM estimation target,

    S(u, f) = S_inst(f) + m(u)^2 S_gal(f) = E[w_nm^2].

The stationary spectral shapes use the analytic Robson, Cornish & Liu (2019)
fits. The seasonal envelope follows the cyclostationary Galactic-background
model of Digman & Cornish (2022): the local confusion *power* is modulated by
``r(t) = 1 + sum_k A_k cos(2 pi k t / T_year - phi_k)`` -- the annual harmonics
of LISA's antenna pattern sweeping the anisotropic Galaxy -- so the amplitude
envelope is ``m(u) = sqrt(r(t))`` with ``<m^2> = <r> = 1`` over whole years.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_GB_PARAM_ORDER = ("f0", "fdot", "A", "ra", "dec", "psi", "iota", "phi0")
_SEC_PER_YEAR = 365.25 * 86400.0


# (alpha, beta, kappa, gamma, f_knee[Hz]) Galactic-confusion fit by observation time.
_GALACTIC_FIT_PARAMS: dict[str, tuple[float, float, float, float, float]] = {
    "0.5yr": (0.133, 243.0, 482.0, 917.0, 2.58e-3),
    "1yr": (0.171, 292.0, 1020.0, 1680.0, 2.15e-3),
    "2yr": (0.165, 299.0, 611.0, 1340.0, 1.73e-3),
    "4yr": (0.138, -221.0, 521.0, 1680.0, 1.13e-3),
}

# Annual-harmonic coefficients (A_k, phi_k), k = 1..5, of the cyclostationary
# Galactic-confusion modulation r(t) = 1 + sum_k A_k cos(2 pi k t/T_year - phi_k),
# from Digman & Cornish (2022), Table 1, for the A and E TDI channels at several
# observation durations. The dominant k = 2 harmonic gives two peaks per year.
_DC_MODULATION_HARMONICS: dict[tuple[str, str], tuple[tuple[float, float], ...]] = {
    ("A", "1yr"): ((0.183, 3.92), (0.616, 3.09), (0.012, 4.92), (0.004, 3.33), (0.005, 4.72)),
    ("E", "1yr"): ((0.212, 3.56), (0.462, 3.08), (0.022, 0.94), (0.027, 0.08), (0.006, 1.84)),
    ("A", "2yr"): ((0.177, 3.92), (0.622, 3.10), (0.012, 4.93), (0.003, 3.83), (0.004, 4.49)),
    ("E", "2yr"): ((0.211, 3.54), (0.458, 3.08), (0.023, 0.96), (0.023, 0.05), (0.004, 2.01)),
    ("A", "4yr"): ((0.181, 3.91), (0.625, 3.09), (0.016, 5.38), (0.006, 3.98), (0.004, 4.20)),
    ("E", "4yr"): ((0.209, 3.58), (0.462, 3.08), (0.022, 1.23), (0.023, 0.03), (0.002, 1.11)),
    ("A", "8yr"): ((0.183, 3.95), (0.630, 3.09), (0.016, 5.47), (0.008, 3.85), (0.005, 4.38)),
    ("E", "8yr"): ((0.207, 3.58), (0.467, 3.08), (0.023, 1.12), (0.024, 0.05), (0.003, 1.74)),
}

_SPEED_OF_LIGHT = 299_792_458.0

def _require_lisa():
    try:
        import jaxgb.jaxgb  # noqa: F401
        import lisaorbits  # noqa: F401
        from lisatools.sensitivity import (  # noqa: F401
            XYZ1SensitivityMatrix,
            XYZ2SensitivityMatrix,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The realistic LISA TDI demo needs the [lisa] extra: "
            "uv pip install -e '.[lisa]'"
        ) from exc







@dataclass
class LISANoiseConfig:
    """Configuration for the non-stationary LISA noise generator.

    The seasonal modulation defaults to the physical Digman & Cornish (2022)
    cyclostationary law (``modulation_model="digman_cornish"``); the legacy
    raised-cosine envelope is retained as ``modulation_model="raised_cosine"``
    for backward compatibility. ``n_year_cycles`` is the number of annual cycles
    spanned by the observation (= ``T_obs / 1 yr``), so a one-year run uses 1.0.
    """

    tobs_key: str = "1yr"
    arm_length: float = 2.5e9
    instrument_scale: float = 1.0
    galactic_scale: float = 1.0
    galactic_amplitude: float = 9e-45
    # Physical cyclostationary modulation (default).
    modulation_model: str = "digman_cornish"
    dc_channel: str = "A"
    dc_tobs_key: str = "1yr"
    n_year_cycles: float = 1.0
    # Legacy raised-cosine modulation parameters.
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


def digman_cornish_power_modulation(
    u: np.ndarray, *, channel: str = "A", tobs_key: str = "1yr", n_year_cycles: float = 1.0
) -> np.ndarray:
    """Cyclostationary confusion-power envelope ``r(t)`` (Digman & Cornish 2022).

    ``r(t) = 1 + sum_{k=1}^{5} A_k cos(2 pi k t/T_year - phi_k)`` with the
    tabulated annual-harmonic coefficients (their Table 1). The rescaled time
    ``u in [0, 1]`` maps to ``t/T_year = n_year_cycles * u``, so a one-year
    observation (``n_year_cycles = 1``) traverses one full annual cycle. The mean
    over whole years is unity, matching the ``<m^2> = 1`` normalisation.
    """
    key = (channel.upper(), tobs_key)
    if key not in _DC_MODULATION_HARMONICS:
        raise ValueError(
            f"No Digman & Cornish harmonics for {key!r}; "
            f"choose from {sorted(_DC_MODULATION_HARMONICS)}."
        )
    u = np.asarray(u, dtype=float)
    t_over_year = n_year_cycles * u
    r = np.ones_like(u)
    for k, (amp, phase) in enumerate(_DC_MODULATION_HARMONICS[key], start=1):
        r = r + amp * np.cos(2.0 * np.pi * k * t_over_year - phase)
    return np.maximum(r, 0.0)


def galactic_modulation(u: np.ndarray, config: LISANoiseConfig) -> np.ndarray:
    """Seasonal modulation envelope ``m(u) >= 0`` with analytic ``<m^2> = 1``.

    Dispatches on ``config.modulation_model``: the physical Digman & Cornish
    cyclostationary law (default) or the legacy raised cosine.
    """
    u = np.asarray(u, dtype=float)
    if config.modulation_model == "digman_cornish":
        power = digman_cornish_power_modulation(
            u, channel=config.dc_channel, tobs_key=config.dc_tobs_key,
            n_year_cycles=config.n_year_cycles,
        )
        return np.sqrt(power)
    if config.modulation_model != "raised_cosine":
        raise ValueError(f"Unknown modulation_model {config.modulation_model!r}.")
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


def lisa_tdi_noise_psd(
    freq_hz: np.ndarray, *, tdi_gen: int = 2, model: str = "scirdv1", channel: int = 0
) -> np.ndarray:
    """One-sided TDI auto-noise PSD (``Sxx`` by default) from lisatools."""
    _require_lisa()
    from lisatools.sensitivity import XYZ1SensitivityMatrix, XYZ2SensitivityMatrix

    cls = XYZ2SensitivityMatrix if tdi_gen == 2 else XYZ1SensitivityMatrix
    f = np.asarray(freq_hz, dtype=np.float64)
    f_safe = np.where(f > 0.0, f, f[f > 0.0].min())
    sens = cls(f_safe, model=model).sens_mat
    return np.transpose(sens, (2, 0, 1))[:, channel, channel].real


def gb_tdi_signal(
    n: int,
    dt: float,
    gb_params: dict[str, float],
    *,
    tdi_gen: float = 2.0,
    channel: int = 0,
    tdi_combination: str = "XYZ",
    n_slow: int = 128,
) -> np.ndarray:
    """Time-domain TDI galactic-binary signal via jaxGB (matched convention).

    Args:
        n: Number of samples (``T = n * dt`` is the jaxGB observation time).
        dt: Sampling interval.
        gb_params: Dict with keys ``f0, fdot, A, ra, dec, psi, iota, phi0``.
        tdi_gen: TDI generation (use ``2.0`` to match ``XYZ2``/``AE2`` noise).
        channel: channel index into ``tdi_combination`` (XYZ: 0/1/2 = X/Y/Z;
            AET: 0/1/2 = A/E/T).
        tdi_combination: ``"XYZ"`` or ``"AET"``.
    """
    _require_lisa()
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import lisaorbits
    from jaxgb.jaxgb import JaxGB

    T = n * dt
    gb = JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=T, t0=0.0, n=n_slow)
    params = jnp.array([float(gb_params[k]) for k in _GB_PARAM_ORDER])
    tdi = gb.get_tdi(params, tdi_generation=tdi_gen, tdi_combination=tdi_combination)
    h_band = np.asarray(tdi[channel]).astype(np.complex128)  # continuous FT, on a band
    kmin = int(np.asarray(gb.get_kmin(f0=jnp.array([float(gb_params["f0"])])))[0])

    n_freq = n // 2 + 1
    spectrum = np.zeros(n_freq, dtype=complex)
    bins = kmin + np.arange(h_band.size)
    valid = (bins >= 0) & (bins < n_freq)
    spectrum[bins[valid]] = h_band[valid] / dt  # continuous FT -> numpy rfft
    return np.fft.irfft(spectrum, n=n)


def lisa_tdi_confusion_psd(
    freq_hz: np.ndarray, *, tobs_years: float = 1.0, tdi_gen: int = 2,
    model: str = "scirdv1", channel: int = 0,
) -> np.ndarray:
    """One-sided TDI galactic-confusion PSD = (instrument+galactic) - instrument."""
    _require_lisa()
    from lisatools.sensitivity import XYZ1SensitivityMatrix, XYZ2SensitivityMatrix

    cls = XYZ2SensitivityMatrix if tdi_gen == 2 else XYZ1SensitivityMatrix
    f = np.asarray(freq_hz, dtype=np.float64)
    f_safe = np.where(f > 0.0, f, f[f > 0.0].min())
    total = np.transpose(
        cls(f_safe, model=model, stochastic_params=(tobs_years * _SEC_PER_YEAR,)).sens_mat,
        (2, 0, 1),
    )[:, channel, channel].real
    inst = lisa_tdi_noise_psd(f, tdi_gen=tdi_gen, model=model, channel=channel)
    return np.maximum(total - inst, 0.0)


def _draw_colored(psd_onesided: np.ndarray, n: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Real series whose one-sided PSD is ``psd_onesided`` (E|rfft|^2 = (n/2dt) S)."""
    spectrum = np.zeros(n // 2 + 1, dtype=complex)
    interior_scale = np.sqrt(n / (4.0 * dt) * np.maximum(psd_onesided, 0.0))
    if n % 2 == 0:
        ni = n // 2 - 1
        spectrum[1:-1] = interior_scale[1:-1] * (
            rng.standard_normal(ni) + 1j * rng.standard_normal(ni)
        )
        spectrum[-1] = np.sqrt(n / (2.0 * dt) * psd_onesided[-1]) * rng.standard_normal()
    else:
        ni = n // 2
        spectrum[1:] = interior_scale[1:] * (
            rng.standard_normal(ni) + 1j * rng.standard_normal(ni)
        )
    return np.fft.irfft(spectrum, n=n)


def simulate_tdi_noise(
    n: int,
    dt: float,
    rng: np.random.Generator,
    *,
    tdi_gen: int = 2,
    model: str = "scirdv1",
    channel: int = 0,
) -> np.ndarray:
    """Stationary TDI instrument-noise realisation with the lisatools PSD."""
    freq = np.fft.rfftfreq(n, d=dt)
    psd = lisa_tdi_noise_psd(freq, tdi_gen=tdi_gen, model=model, channel=channel)
    return _draw_colored(psd, n, dt, rng)


def simulate_tv_lisa_tdi(
    n: int,
    dt: float,
    rng: np.random.Generator,
    *,
    mod_config: LISANoiseConfig,
    tobs_years: float = 1.0,
    tdi_gen: int = 2,
    model: str = "scirdv1",
    channel: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Non-stationary TDI noise: instrument + seasonally modulated confusion.

    The instantaneous one-sided PSD is ``S_inst(f) + m(u)^2 S_conf(f)`` with the
    confusion amplitude modulated by ``galactic_modulation`` (``<m^2> = 1``). All
    components share the TDI-X, generation-``tdi_gen`` convention of the jaxGB
    signal, so the joint signal+noise problem is physically consistent.
    """
    instrument = simulate_tdi_noise(n, dt, rng, tdi_gen=tdi_gen, model=model, channel=channel)
    freq = np.fft.rfftfreq(n, d=dt)
    s_conf = lisa_tdi_confusion_psd(freq, tobs_years=tobs_years, tdi_gen=tdi_gen,
                                    model=model, channel=channel)
    confusion = _draw_colored(s_conf, n, dt, rng)
    u = np.arange(n, dtype=float) / n
    data = instrument + galactic_modulation(u, mod_config) * confusion
    return data, {"freq": freq, "s_conf": s_conf}


def true_tv_lisa_tdi_psd(
    time_grid: np.ndarray, freq_grid_hz: np.ndarray, *, mod_config: LISANoiseConfig,
    tobs_years: float = 1.0, tdi_gen: int = 2, model: str = "scirdv1", channel: int = 0,
) -> np.ndarray:
    """Analytic non-stationary TDI PSD surface ``S(u, f)`` (one-sided, physical)."""
    s_inst = lisa_tdi_noise_psd(freq_grid_hz, tdi_gen=tdi_gen, model=model, channel=channel)
    s_conf = lisa_tdi_confusion_psd(freq_grid_hz, tobs_years=tobs_years, tdi_gen=tdi_gen,
                                    model=model, channel=channel)
    m2 = galactic_modulation(time_grid, mod_config) ** 2
    return s_inst[None, :] + m2[:, None] * s_conf[None, :]


def optimal_snr(signal: np.ndarray, dt: float, *, tdi_gen: int = 2, model: str = "scirdv1",
                channel: int = 0) -> float:
    """Matched-filter optimal SNR of a time-domain signal against the TDI PSD."""
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=dt)
    h = dt * np.fft.rfft(signal)  # continuous FT
    psd = lisa_tdi_noise_psd(freq, tdi_gen=tdi_gen, model=model, channel=channel)
    df = 1.0 / (n * dt)
    integrand = np.zeros_like(psd)
    mask = freq > 0
    integrand[mask] = 4.0 * np.abs(h[mask]) ** 2 / psd[mask] * df
    return float(np.sqrt(np.sum(integrand)))


# --- A/E TDI channels (orthogonal combinations) --------------------------------
# The A and E channels are the standard noise-orthogonal TDI variables. Their
# instrument auto-spectra come from lisatools' AE sensitivity; the Galactic
# foreground is modulated by the channel-specific Digman & Cornish (2022) annual
# law (A and E have different harmonic coefficients), so the two channels carry
# genuinely different non-stationarity.

_AE_CHANNELS = {"A": 0, "E": 1}


def ae_tdi_noise_psd(
    freq_hz: np.ndarray, *, channel: str = "A", tdi_gen: int = 2, model: str = "scirdv1"
) -> np.ndarray:
    """One-sided A or E instrument-noise auto-PSD from lisatools."""
    _require_lisa()
    from lisatools.sensitivity import AE1SensitivityMatrix, AE2SensitivityMatrix

    cls = AE2SensitivityMatrix if tdi_gen == 2 else AE1SensitivityMatrix
    f = np.asarray(freq_hz, dtype=np.float64)
    f_safe = np.where(f > 0.0, f, f[f > 0.0].min())
    sens = np.asarray(cls(f_safe, model=model).sens_mat)  # (2, n_freq)
    return sens[_AE_CHANNELS[channel]].real


def ae_tdi_confusion_psd(
    freq_hz: np.ndarray, *, channel: str = "A", tobs_years: float = 1.0,
    tdi_gen: int = 2, model: str = "scirdv1",
) -> np.ndarray:
    """One-sided A or E Galactic-confusion PSD = (instrument+galactic) - instrument."""
    _require_lisa()
    from lisatools.sensitivity import AE1SensitivityMatrix, AE2SensitivityMatrix

    cls = AE2SensitivityMatrix if tdi_gen == 2 else AE1SensitivityMatrix
    f = np.asarray(freq_hz, dtype=np.float64)
    f_safe = np.where(f > 0.0, f, f[f > 0.0].min())
    total = np.asarray(
        cls(f_safe, model=model, stochastic_params=(tobs_years * _SEC_PER_YEAR,)).sens_mat
    )[_AE_CHANNELS[channel]].real
    inst = ae_tdi_noise_psd(f, channel=channel, tdi_gen=tdi_gen, model=model)
    return np.maximum(total - inst, 0.0)


def gb_ae_signal(
    n: int, dt: float, gb_params: dict[str, float], *, channel: str = "A",
    tdi_gen: float = 2.0, n_slow: int = 128,
) -> np.ndarray:
    """Time-domain A or E galactic-binary signal via jaxGB (matched convention)."""
    return gb_tdi_signal(
        n, dt, gb_params, tdi_gen=tdi_gen, channel=_AE_CHANNELS[channel],
        tdi_combination="AET", n_slow=n_slow,
    )


def simulate_tv_ae_tdi(
    n: int, dt: float, rng: np.random.Generator, *, channel: str,
    mod_config: LISANoiseConfig, tobs_years: float = 1.0, tdi_gen: int = 2,
    model: str = "scirdv1",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Non-stationary A or E noise: instrument + seasonally modulated confusion."""
    freq = np.fft.rfftfreq(n, d=dt)
    s_inst = ae_tdi_noise_psd(freq, channel=channel, tdi_gen=tdi_gen, model=model)
    s_conf = ae_tdi_confusion_psd(freq, channel=channel, tobs_years=tobs_years,
                                  tdi_gen=tdi_gen, model=model)
    instrument = _draw_colored(s_inst, n, dt, rng)
    confusion = _draw_colored(s_conf, n, dt, rng)
    u = np.arange(n, dtype=float) / n
    data = instrument + galactic_modulation(u, mod_config) * confusion
    return data, {"freq": freq, "s_inst": s_inst, "s_conf": s_conf}


def true_tv_ae_tdi_psd(
    time_grid: np.ndarray, freq_grid_hz: np.ndarray, *, channel: str,
    mod_config: LISANoiseConfig, tobs_years: float = 1.0, tdi_gen: int = 2,
    model: str = "scirdv1",
) -> np.ndarray:
    """Analytic non-stationary A or E PSD surface ``S(u, f)`` (one-sided)."""
    s_inst = ae_tdi_noise_psd(freq_grid_hz, channel=channel, tdi_gen=tdi_gen, model=model)
    s_conf = ae_tdi_confusion_psd(freq_grid_hz, channel=channel, tobs_years=tobs_years,
                                  tdi_gen=tdi_gen, model=model)
    m2 = galactic_modulation(time_grid, mod_config) ** 2
    return s_inst[None, :] + m2[:, None] * s_conf[None, :]


def optimal_snr_ae(
    signal_a: np.ndarray, signal_e: np.ndarray, dt: float, *, tdi_gen: int = 2,
    model: str = "scirdv1",
) -> float:
    """Combined A+E matched-filter optimal SNR of a time-domain signal pair."""
    n = len(signal_a)
    freq = np.fft.rfftfreq(n, d=dt)
    df = 1.0 / (n * dt)
    mask = freq > 0
    snr2 = 0.0
    for sig, ch in ((signal_a, "A"), (signal_e, "E")):
        h = dt * np.fft.rfft(sig)
        psd = ae_tdi_noise_psd(freq, channel=ch, tdi_gen=tdi_gen, model=model)
        snr2 += np.sum(4.0 * np.abs(h[mask]) ** 2 / psd[mask] * df)
    return float(np.sqrt(snr2))



