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
"""

from __future__ import annotations

import numpy as np

from .lisa import LISANoiseConfig, galactic_modulation

_GB_PARAM_ORDER = ("f0", "fdot", "A", "ra", "dec", "psi", "iota", "phi0")
_SEC_PER_YEAR = 365.25 * 86400.0


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
